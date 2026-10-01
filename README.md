# GEMM Multi-Threads

A multi-threaded, BLIS-style GEMM implementation for Apple Silicon (ARM NEON). It benchmarks different blocking configurations (`MC` / `NC` / `KC`) under **heavy thermal throttling** to obtain a fair, realistic ranking.

The project started as C++ only. Analysing the benchmark data turned out to be much easier with Python tooling, so it is now a hybrid project. To quantify the thermal-throttling effect, `powermetrics` is run separately in a terminal. Three parts produce data:

| Part | Output | Description |
|------|--------|-------------|
| Terminal | `thermal_log.txt` | CPU metrics from `powermetrics` (sampled per second) |
| C++ | `./log/bench_runs.csv`, `./log/gemm_mThds.log` | GEMM performance per run |
| Python | `./data/integrated_bench.csv`, figures under `./png/` | Merges `bench_runs.csv` with `thermal_log.txt`, then analyses and plots |

> **Note:** `powermetrics` must be started from `./log/` (it requires `sudo`), so that `thermal_log.txt` is written where the Python scripts expect it. See [Build & Run](#build--run).

---

## Requirements

- Python 3 (`matplotlib`, `numpy`, `pandas`, `seaborn`)
- C++17
- CMake ≥ 3.20
- ARM NEON (Apple Silicon)

---

## Project Structure

```text
gemm_multiThreads/
│
├── data/
│   └── integrated_bench.csv    # Produced by scripts/01_* (bench_runs.csv + thermal_log.txt)
│
├── log/
│   ├── bench_runs.csv          # Produced by src/main.cpp
│   ├── gemm_mThds.log          # Produced by src/main.cpp
│   └── thermal_log.txt         # Produced by powermetrics (terminal)
│
├── png/
│   ├── config_score.png        # Produced by scripts/02_*
│   └── pcluster_freq.png       # Produced by scripts/02_*
│
├── scripts/
│   ├── 01_integrate_bench.py   # Merge thermal_log.txt into bench_runs.csv
│   ├── 02_analyze_bench.py     # Analyse integrated_bench.csv and plot figures
│   └── toolkits.py             # Utilities
│
├── include/
│   ├── gemm.hpp                # GEMM interface
│   ├── kernel_neon.hpp         # GEMM tool
│   ├── packing.hpp             # GEMM tool
│   ├── ThreadPool.hpp          # Thread pool used by GEMM
│   ├── logger.hpp              # Writes results to bench_runs.csv
│   └── toolkit.hpp             # Utilities
│
├── src/
│   ├── main.cpp                # Benchmark entry point
│   └── gemm.cpp                # GEMM implementation
│
├── CMakeLists.txt
│
└── README.md
```

---

## Build & Run

1. Build the C++ executable

```bash
cmake -S . -B build
cmake --build build
```

2. Run the system measurement and the benchmark

Open a separate terminal and run `powermetrics` from `./log/`:

```bash
cd path/to/gemm_multiThreads/log
sudo powermetrics --samplers cpu_power,thermal -i 1000 -n 3600 | tee thermal_log.txt | grep -E "P-Cluster HW active frequency|Thermal"
```

Then run the executable in the original terminal:

```bash
path/to/gemm_multiThreads/build/gemm_mthreads
```

3. Set up the Python environment and run the analysis

Once the benchmark has finished:

```bash
python3 -m venv .venv
source ./.venv/bin/activate
which python    # make sure the venv Python is in use
pip install matplotlib numpy pandas seaborn
python3 ./scripts/01_integrate_bench.py
python3 ./scripts/02_analyze_bench.py
```

---

## Algorithm

Computes $C = A \times B$, where

- $C \in \mathbb{R}^{M \times N}$
- $A \in \mathbb{R}^{M \times K}$
- $B \in \mathbb{R}^{K \times N}$

### Single-thread baseline

C, A and B are partitioned into blocks (macro-panels) and tiles (micro-panels), and the innermost tile update is accelerated with SIMD:

```text
for jp : N
    for kp : K
        pack blockB[KC x NC]
        for ip : M
            pack blockA[MC x KC]
            for ir : MC
                for jr : NC
                    SIMD on tileC[MR x NR] with blockA, blockB
```

### Multi-threaded GEMM

**Step 1: parallelize the `ip` loop.**
Packing is the bottleneck. `blockA` is packed inside the innermost of the three outer loops, so it is packed far more often than `blockB` and dominates the latency. Moreover, the `blockA` panels are non-overlapping across different `ip`, which makes the `ip` loop an ideal candidate for parallelization.

```text
for jp : N
    for kp : K
        pack blockB[KC x NC]
        for ip : M
            parallelize {
                pack blockA[MC x KC]
                for ir : MC
                    for jr : NC
                        SIMD on tileC[MR x NR] with blockA, blockB
            }

        main thread sync
```

**Step 2: overlap `blockB` packing with worker computation (double buffering).**
While the workers are busy packing `blockA` and running SIMD, the main thread would otherwise sit idle. Since `blockB` is being read by the workers, we allocate twice the space (`blockB[KC x NC x 2]`) and let the main thread pack the *next* `blockB` into the unused half.

The `ip` tasks are submitted to the thread pool asynchronously: after dispatching them, the main thread moves on to the next iteration and packs the next `blockB` while the workers are still computing. The main thread only waits for the workers right before it would reuse a buffer half that they might still be reading. The two halves alternate continuously, including across `jp` boundaries.

```text
for jp : N
    for kp : K
        // blockB[KC x NC x 2]
        main thread packs the unused half of blockB
        main thread syncs: wait for workers to finish with the in-use half

        for ip : M
            parallelize {
                pack blockA[MC x KC]
                for ir : MC
                    for jr : NC
                        SIMD on tileC[MR x NR] with blockA, blockB
            }

main thread sync    // wait for all workers to finish before returning
```

### Round-robin / shuffled sweep

To avoid a first-run advantage (e.g. the first configuration running while the chip is still cool), every repetition sweeps through the entire set of `MC` / `NC` / `KC` configurations in a freshly shuffled order:

```text
for round : repetitions
    shuffle config set
    for config : config set
        run gemm(config)
```

---

## Analysis

- Problem size: $M / N / K = 5347 / 4655 / 4201$
- Data type: `float32`

### Methodology

The analysis has two stages: (1) detect when the CPU has settled into a stable thermal state, and (2) rank the `MC` / `NC` / `KC` configurations using only the runs from that steady state.

#### Stage 1: Finding the steady state

After warmup, the Thermal Pressure Level stays at `Heavy`. The `P-Cluster Frequency` first declines and then flattens out at around the 800th sample.

![P-Cluster Frequency](./png/pcluster_freq.png)

To locate this plateau objectively, the frequency series is smoothed with a rolling median, and the absolute slope between consecutive smoothed points is averaged over a sliding window. The first position where this average drops below a threshold $\varepsilon$ is taken as the start of the steady state.

```text
stable_pos  = first index where mean_slope < ε
```

#### Stage 2: Scoring the configurations

Only runs after `stable_pos` are used, so every configuration is compared under the same thermal condition. Among those runs, only the fastest 10% are kept. The score of a configuration combines *how highly* its runs rank and *how often* they appear in that top group.

```text
steady          = runs[stable_pos:], sorted by performance
top_10percent   = top 10% runs of steady

// 1. Rank weight: the fastest run gets 1, decaying exponentially
for i = 0 .. n-1:
    top_10percent[i].w = exp(-i / n)      // n = |top_10percent|

// 2. Aggregate per (MC, NC, KC)
for each config c:
    rank_mass[c] = sum of w over c's runs in top_10percent
    freq[c]      = number of c's runs in top_10percent

// 3. Normalize against the best config
rank_norm[c] = rank_mass[c] / max(rank_mass)
freq_norm[c] = log(1 + freq[c]) / log(1 + max(freq))

// 4. Final score
score[c] = W_RANK * rank_norm[c] + W_FREQ * freq_norm[c]    // W_RANK = 0.7, W_FREQ = 0.3
```

Or equivalently:

$$
\text{score}(c) = W_{\text{rank}} \cdot \frac{\sum_{r \in c} e^{-i_r/n}}{\max_{c'} \sum_{r \in c'} e^{-i_r/n}} + W_{\text{freq}} \cdot \frac{\ln(1 + f_c)}{\ln(1 + \max_{c'} f_{c'})}
$$

where $i_r$ is the rank of run $r$ (0 = fastest), $n$ is the number of top runs, and $f_c$ is the number of runs of configuration $c$ in the top group.

Notes:

- **Rank weight:** the exponential decay rewards faster runs without letting a single lucky run dominate.
- **Frequency term:** a configuration that shows up repeatedly in the top group is more trustworthy than one that appears once. The logarithm keeps a high count from overwhelming the rank term.
- **Max normalization:** each term answers "how close is this configuration to the best one?", so both terms lie in $[0, 1]$ and can be mixed with the weights above.
- **Scope of the score:** it is built from the fastest 10% of steady-state runs, so it ranks configurations by how consistently they reach the best observed performance, not by their typical performance. It carries no confidence interval, so small score gaps should not be over-read. The weights 0.7 / 0.3 are a choice, not a derived value.

---

### Inference

![Config Score](./png/config_score.png)

The heat map shows `MC` vs `NC` for each `KC`. A higher score means the configuration lands among the fastest steady-state runs more consistently. The top configurations are:

| Rank | (MC, NC, KC) | Score |
|------|--------------|-------|
| 1 | (16, 1024, 512) | 0.998 |
| 2 | (16, 1024, 384) | 0.976 |
| 3 | (32, 1024, 512) | 0.847 |
| 4 | (32, 1024, 384) | 0.828 |
| 5 | (32, 2048, 384) | 0.807 |

#### Observations

- **KC:** `KC = 384` and `KC = 512` dominate the top 10%. `KC = 256` reaches at most 0.786 (at MC = 16, NC = 2048), and `KC = 198` at most 0.471.
- **MC:** smaller is better. At `NC = 1024` the score drops monotonically as MC grows: 0.998 → 0.847 → 0.304 for `KC = 512`, and 0.976 → 0.828 → 0.666 for `KC = 384`.
- **NC:** `NC ≤ 256` almost never reaches the top group. For `KC ≥ 384` the peak sits at `NC = 1024`, and `NC = 2048` is competitive but lower (e.g. 0.767 vs 0.998 at MC = 16, KC = 512). For `KC ≤ 256`, `NC = 2048` is the better choice, which hints that the footprint of blockB (`KC x NC`) matters and not `NC` alone. It is not the whole story, though: (16, 2048, 256) and (16, 1024, 512) have the same `KC x NC` but score 0.786 vs 0.998.
- **Difference from the single-thread version:** there, a larger `NC` is always better. A is repacked once per `jp` iteration, i.e. $N / NC$ times, so its total packing volume is $M \cdot K \cdot N / NC$ and shrinks as `NC` grows; a larger `NC` also means fewer loop iterations. With multiple threads the `blockA` packing is spread over the workers, so this saving is worth less, and the optimum settles at `NC = 1024` instead of growing further.

#### Possible explanation

In Apple M1 chips, **L1 cache is per-core** (private to each core), while **L2 cache is shared** across a cluster of cores.

```text
[ P-Cluster (Performance / Firestorm) ]
┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   P-Core #1   │  │   P-Core #2   │  │   P-Core #3   │  │   P-Core #4   │
│ ┌───────────┐ │  │ ┌───────────┐ │  │ ┌───────────┐ │  │ ┌───────────┐ │
│ │ L1 Cache  │ │  │ │ L1 Cache  │ │  │ │ L1 Cache  │ │  │ │ L1 Cache  │ │
│ │ 192KB Inst│ │  │ │ 192KB Inst│ │  │ │ 192KB Inst│ │  │ │ 192KB Inst│ │
│ │ 128KB Data│ │  │ │ 128KB Data│ │  │ │ 128KB Data│ │  │ │ 128KB Data│ │
│ └───────────┘ │  │ └───────────┘ │  │ └───────────┘ │  │ └───────────┘ │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │                  │
        └──────────────────┴────────┬─────────┴──────────────────┘
                                    ▼
                        ┌───────────────────────┐
                        │ Shared 12 MB L2 Cache │
                        └───────────────────────┘


[ E-Cluster (Efficiency / Icestorm) ]
┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   E-Core #1   │  │   E-Core #2   │  │   E-Core #3   │  │   E-Core #4   │
│ ┌───────────┐ │  │ ┌───────────┐ │  │ ┌───────────┐ │  │ ┌───────────┐ │
│ │ L1 Cache  │ │  │ │ L1 Cache  │ │  │ │ L1 Cache  │ │  │ │ L1 Cache  │ │
│ │ 128KB Inst│ │  │ │ 128KB Inst│ │  │ │ 128KB Inst│ │  │ │ 128KB Inst│ │
│ │  64KB Data│ │  │ │  64KB Data│ │  │ │  64KB Data│ │  │ │  64KB Data│ │
│ └───────────┘ │  │ └───────────┘ │  │ └───────────┘ │  │ └───────────┘ │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │                  │
        └──────────────────┴────────┬─────────┴──────────────────┘
                                    ▼
                        ┌───────────────────────┐
                        │ Shared 4 MB L2 Cache  │
                        └───────────────────────┘

```
![M1 chip](https://assets.toptal.io/images?url=https%3A%2F%2Fbs-uploads.toptal.io%2Fblackfish-uploads%2Fuploaded_file%2Ffile%2F463051%2Fimage-1606946154481-a6ace29657b8c2432ba5573f441aa452.png&width=1440)
*Figures from https://www.toptal.com/developers/ios/apple-m1-processor-compatibility-overview*

Based on the Apple M1 architecture, the following are hypotheses. They have not been verified with hardware counters, and they are not mutually exclusive:

- **Small blockA stays cache-hot between packing and use, one copy per thread.** `MC` matters for the whole `blockA` (`MC x KC` floats: 24 KB at (16, 384), 32 KB at (16, 512)), which a task packs and then consumes right away. A small blockA is more likely to still sit in L1 when it is read after packing, instead of being served from L2. If the 8 workers are spread over the 4 P-cores (128 KB L1D each) and the 4 E-cores (64 KB L1D each), larger `MC` loses this first on the E-cores (`MC = 32, KC = 512` and `MC = 64, KC ≥ 256` are already ≥ 64 KB, the whole E-core L1D) and then on the P-cores (`MC = 64, KC = 512` is 128 KB). Because a parallel region only finishes when its slowest worker does, this is consistent with the monotonic `MC` trend.
- **blockB lives in L2, but each cluster has its own L2.** Each `ip` task sweeps the whole in-use `blockB` once per A micro-panel (`MC / MR` times), and it is served from L2. The P-cluster has a 12 MB L2 and the E-cluster a 4 MB L2, so each cluster pulls its own copy of the in-use blockB into its L2. At `KC x NC = 512 x 1024` that is 2 MB, far larger than L1. At `KC x NC = 512 x 2048` it is 4 MB, the entire E-cluster L2 before counting blockA and C traffic, and the P-cluster also has to hold the next blockB half being packed (about 8 MB in total). This may explain the drop at `NC = 2048` for `KC = 512`.
- **Smaller `MC` gives finer tasks and better load balance.** The cores are heterogeneous, and one `ip` task covers `MC` rows of C. For M = 5347, `MC = 16` gives about 335 tasks per parallel region, while `MC = 64` gives about 84. With finer tasks, the fast P-cores can take over more of the work, so they spend less time waiting for the E-cores at the end of a region. This effect could contribute to the `MC` trend independently of cache capacity, and the two are not separated by the current data.
- **Small NC gives each parallel region little work.** With `NC ≤ 256`, synchronization and dispatch overhead take a larger share of the runtime, and blockB is repacked more often.

---

## Further Exploration & Study

### Exploration

The best configurations sit on the edge of the search grid (`MC = 16`, `KC = 512`), so the true optimum may lie outside it.

Suggestion:
- MC Exploration: Try MC $\in$ {8, 12, 16} (a multiple of MR) to test whether a smaller, more L1-friendly blockA and finer task granularity yield further gains.
- NC Resolution: Test intermediate values between 512, 1024, and 2048 (e.g., NC $\in$ {768, 1280, 1536}) to pinpoint the threshold at which blockB (and its double buffer) starts to exceed L2 capacity.
- KC Upper Bound: Extend KC $\in$ {640, 768, 1024} to see where the footprints of the A micro-panel (`MR x KC`), the B micro-panel (`KC x NR`) and blockA (`MC x KC`) outgrow L1. A larger KC amortizes the load/store traffic of the C tile, so the optimum is limited by cache capacity.
- Separating the hypotheses: compare static vs. dynamic task scheduling, or pin the workers to P-cores only, to tell load balance apart from cache pressure.
- Robustness: report the number of runs per configuration, and check that a median-based ranking agrees with the top-10% score.

### Study

- https://www.toptal.com/developers/ios/apple-m1-processor-compatibility-overview
- https://chipsandcheese.com/p/igpu-cache-setups-compared-including-m1
