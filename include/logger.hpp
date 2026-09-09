#pragma once
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>


/**
 * @brief 記錄每次 GEMM 執行的時間戳與 config 資訊，供事後跟 powermetrics
 *        thermal log 對齊分析用。
 *
 * 設計重點：
 *   - 用 std::chrono::system_clock（不是 steady_clock）取得 wall-clock
 *     時間，因為 powermetrics 印出的時間戳也是 wall-clock，兩份 log
 *     才能用同一個時間軸對齊。
 *   - 記錄的是「執行前」跟「執行後」兩個時間戳，而不是只記錄耗時，
 *     這樣事後才能把某次 GEMM run 精準對應到 thermal log 裡對應時段
 *     的頻率/thermal pressure 讀數。
 *   - 每筆記錄立即 flush，避免程式中途中斷（比如 correctness check
 *     拋例外）導致緩衝區資料遺失。
 *   - CSV 格式，方便丟進 Python/Excel 跟 thermal_log.txt 一起處理。
 *
 * 用法：
 *   RunLogger logger("bench_runs.csv");
 *   for (...) {
 *       logger.record_start();
 *       auto t0 = high_resolution_clock::now();
 *       gemm(...);
 *       auto t1 = high_resolution_clock::now();
 *       logger.record_end(round, position, config.mc, config.nc, config.kc,
 *                          config.workers, duration_ms);
 *   }
 */
class Logger {
public:
    explicit Logger(const std::string& path) : out(path, std::ios::out) {
        if (!out.is_open()) {
            std::cerr   << "Logger: failed to open "
                        << path
                        << " for writing\n";
        }
        out << "wall_start,wall_end,round,position,mc,nc,kc,workers,time_ms\n";
        out.flush();
    }

    void record_start() {
        pending_start = std::chrono::system_clock::now();
    }

    void record_end(
        int round, size_t position, size_t mc, size_t nc, size_t kc, size_t workers, double time_ms
    ) {
        auto wall_end = std::chrono::system_clock::now();
        out << iso8601(pending_start) << ","
            << iso8601(wall_end) << ","
            << round << ","
            << position << ","
            << mc << ","
            << nc << ","
            << kc << ","
            << workers << ","
            << time_ms << "\n";
        out.flush();
    }

    ~Logger() {
        if (out.is_open()) {
            out.close();
        }
    }

private:
    std::ofstream out;
    std::chrono::system_clock::time_point pending_start;

    static std::string iso8601(std::chrono::system_clock::time_point tp) {
        auto t = std::chrono::system_clock::to_time_t(tp);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()) % 1000;
        
        std::tm tm_buf{};
        localtime_r(&t, &tm_buf);

        std::ostringstream oss;
        oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
            << "." << std::setfill('0')
            << std::setw(3)
            << ms.count();
        
        return oss.str();
    }
};