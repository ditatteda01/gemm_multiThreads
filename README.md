# GEMM 效能調參 Benchmark：方法論與結果

BLIS 風格 GEMM 實作（C++ / NEON / ThreadPool）在 Apple M1（MacBook Air, 無風扇）上的
MC/NC/KC block size 調參結果，重點記錄如何在**持續高負載導致的熱節流**下取得可信的
效能排名。

矩陣規模：`C(5347, 4655) = A(5347, 4201) x B(4201, 4655)`，float32，8 threads。

---

## 背景：為什麼不能直接照原始 sweep 結果調參

第一版 benchmark（大區塊依序跑完每個 config 的所有 repetition）觀察到 KC=198 系統性地比其他 KC 值慢 22–30%。看起來像是 cache line 不對齊（198×4B=792B 無法被 128B cache line 整除）造成的硬體限制，但這個假說有一個沒被排除的混淆變因：**KC=198 剛好被排在 sweep 的最後一批**，而 Apple M1（無風扇）在連續高負載下會顯著降頻。

驗證方式：把 KC=198 的順序調到第一位，結果在多組 MC/NC 組合下變成最快，證實原本的排名差異主要是**執行順序造成的熱節流**，不是 cache 對齊的硬體限制。

## 方法論：round-robin + 已知熱穩態

為了得到不被執行順序污染的排名，最終採用的協定：

1. **Warm-up**：正式計時前，先用全部 60 組 config 跑 N 輪暖機（不計時），讓系統進入一個已知、可重現的熱狀態，而不是從冷機開始。

2. **Round-robin benchmark**：不是「洗牌一次、每個 config 連續跑 N 次」（那仍然是大區塊，只是區塊被打散了順序），而是**每一輪都重新洗牌**，60 個 config 在每輪內全部跑過一次再進下一輪。任何殘留的熱漂移會被均勻分攤到所有 config，不會集中在特定幾組上。

3. **溫度/頻率同步記錄**：全程用 `powermetrics --samplers cpu_power,thermal -i 1000` 記錄 P-Cluster 頻率與 thermal pressure level，事後可與 benchmark log 的時間戳對齊，驗證是否真的達到穩態、round-robin 是否有效抵銷順序效應。

4. **C++ 端獨立記錄 wall-clock 時間戳**（見 `logger.hpp`），與 `powermetrics` 使用同一個時鐘基準（`system_clock`，非計時用的 `high_resolution_clock`），確保兩份 log 能精確合併分析。

**最終跑法：20 輪 warm-up + 30 輪 round-robin benchmark，每輪 60 config，共 1800 次正式量測，全程同步記錄 thermal log。**

## 驗證結果：這次的 round-robin 有沒有真的消除順序偏差？

把 1800 筆 benchmark 記錄與同一小時的 thermal log（3600 筆，1Hz 取樣）依時間戳合併分析：

| 指標 | 結果 | 解讀 |
|:---|:---|:---|
| Thermal pressure level | 全部 1800 筆皆為 `Heavy` | benchmark 全程處於同一個節流等級，沒有中途回到 Nominal |
| P-Cluster 頻率 | 中位數 831 MHz，30 輪平均值都落在 832–837 MHz | 頻率打平，round 之間無趨勢 |
| position vs 頻率 相關性 | r = -0.024（p = 0.31，不顯著） | round-robin 成功消除了頻率上的順序效應 |
| position vs 時間 相關性（原始） | r = 0.082（p < 0.001，但解釋力極低） | 弱相關，多半是雜訊 |
| position vs 時間 相關性（扣除各 config 自身均值後） | r = 0.231，R² = 0.054，估計總漂移約 16ms / 1800 次 | 有殘留的系統性漂移，但量級遠小於 config 間差距（100–300ms），不足以翻轉排名 |

***另外發現一個量測假影：全場最慢的 config（MC=16, NC=128, KC=128）因為單次執行時間拖得比其他組長，1Hz 的 `powermetrics` 取樣有更高機率剛好落在單執行緒的 correctness-check / matrix-init 空檔，導致該組頻率讀數異常偏高（900–957MHz vs. 其他組穩定的 828–834MHz）。這只影響「拿頻率做事後分析」的準確性，不影響 C++ 內部計時本身的 `time_ms`，因此不影響排名結論。***

## 最終排名（深度節流穩態下）

| 排名 | MC | NC | KC | Median (ms) | IQR |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 32 | 2048 | 384 | 1245.9 | 27.7 |
| 2 | 64 | 2048 | 384 | 1246.6 | 31.9 |
| 3 | 32 | 1024 | 384 | 1248.4 | 26.4 |
| 4 | 16 | 1024 | 384 | 1252.0 | 21.5 |
| 5 | 64 | 1024 | 384 | 1253.6 | 16.1 |
| ⋯ | | | | | |
| 60 | 16 | 128 | 128 | 1554.0 | 18.6 |

60 組 config 量測結果見 `gemm_mthread_30rep_roundrobin`，以及逐次量測數據 `bench_runs.csv`。

**⚠️ 這份排名的適用範圍是「持續高負載、深度節流穩態」（P-Cluster ≈ 828MHz，約為 M1 峰值 3.2GHz 的 26%），不是峰值/冷機性能。** 同一份程式碼在冷機或輕度節流下量測，~~KC≈128~~ KC=384 系列反而更快（見前段 KC=198 異常排查過程）——**最優 block size 會隨熱力學狀態改變，不存在單一「正確」的 MC/NC/KC**。若之後有明確的部署場景（例如：間歇性單次呼叫 vs. 長時間批次運算），應針對該場景的實際熱力學條件重新用同一套 round-robin 協定量測，而不是直接套用這份結果。

## 檔案

| 檔案 | 用途 |
|:---|:---|
| `main.cpp` | benchmark driver，round-robin 主迴圈 |
| `logger.hpp` | 記錄每次執行的 wall-clock 時間戳，供事後與 thermal log 對齊 |
| `toolkit.hpp` | `correct_check`（相對誤差容忍度）、`get_median` 等工具函式 |
| `bench_runs.csv` | 1800 筆逐次原始量測數據 |
| `thermal_log.txt` | 同時段 `powermetrics` 完整輸出（1Hz） |

## 已知限制 / 尚待驗證

- 尚未驗證真正的「冷啟動 / 短暫呼叫」情境下的最優參數，目前只有「深度節流穩態」的完整數據。

- 16ms 的殘留位置漂移原因尚未完全定位（可能是比 828MHz 更緩慢的次要降頻趨勢，或量測雜訊），量級小到不影響排名但值得留意。

## 執行方法

1. 開啟terminal執行CPU監測紀錄

```bash
cd directory/to/save/your/thermal_log
sudo powermetrics --samplers cpu_power,thermal -i 1000 -n 3600 | tee thermal_log.txt | grep -E "P-Cluster HW active frequency|Thermal pressure state"
```

- `powermetrics`: A built-in tool in Mac that measures power and temperature,
- `--samplers cpu_power,thermal`: This tells the tool to only gather information about the CPU's power usage and the system's thermals.
- `-i 1000`: Take a sample every 1000 milliseconds.
- `-n 3600`: Take exactly 3600 samples.
- `| tee thermal_log.txt`: The `|` symbol takes the data from the previous step and passes it forward. `tee` (T-pipe) takes the data, saves a complete copy of it into a new text file called `thermal_log.txt`, and passes the data forward to the next step.
- `grep -E "P-Cluster HW active frequency|Thermal"`: This is a filter for the screen. While the text file gets all the data, `grep` ensures the screen only shows lines of text that contain the words "P-Cluster HW active frequency" or "Thermal".

*註：在記錄過程中，如果 Terminal 出現 "Second underflow occurred" 字樣，表示 powermetrics 在採集某個 sample 時，實際經過的時間比預期的取樣間隔還短（內部計時上發生了 underflow），原因可能是採樣本身耗時不穩定，也可能是輸出/IO（例如 pipe 阻塞）拖慢了下一輪取樣的啟動時間。這代表該筆記錄可能不是「乾淨」的一個時間切片——不是超過，而是這次取樣的實際時長跟你指定的 -i 對不上（通常是被壓縮或延後了）。這種情況在系統負載重、或有 tee/grep 等下游處理時更容易出現。解決方法：拉長 -i，或是若有接 pipe，改成單純寫檔（-o）事後再處理，減少 IO 造成的阻塞。*

2. 在project下執行程式

```bash
./build/gemm_mthreads
```
