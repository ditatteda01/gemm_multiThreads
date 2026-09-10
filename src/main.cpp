#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include "gemm.hpp"
#include "toolkit.hpp"
#include "logger.hpp"


int main() {

    try {
        int M = 5347;
        int N = 4655;
        int K = 4201;
        float* A = new float[M * K];
        float* B = new float[K * N];
        float* C = new float[M * N];
        float* GroundTruth = new float[M * N];
        std::vector<float*> ptr_handler{A, B, C, GroundTruth};

        std::cout << "Initialize...";
        {
            std::random_device rdv;
            std::mt19937 rng(rdv());
            std::uniform_real_distribution<float> dis(-5.0f, 5.0f);
    
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < K; j++) {
                    A[i * K + j] = dis(rng);
                }
            }
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < N; j++) {
                    B[i * N + j] = dis(rng);
                }
            }
        }

        ToolKit_GEMM::initMat(C, M * N);
        ToolKit_GEMM::initMat(GroundTruth, M * N);
        std::cout << "Done\n";
        std::cout << "C(" << M << ", " << N << ") = ";
        std::cout << "A(" << M << ", " << K << ") x B(" << K << ", " << N << ")\n";

        std::__1::chrono::steady_clock::time_point start;
        std::__1::chrono::steady_clock::time_point end;
        std::chrono::duration<double, std::milli> alg_elapsed;

        std::cout << "Calculate A x B...";
        start = std::chrono::high_resolution_clock::now();
        ToolKit_GEMM::matmul(A, B, GroundTruth, M, N, K);
        end = std::chrono::high_resolution_clock::now();
        alg_elapsed = end - start;
        std::cout << "Done. Time Elapsed: " << alg_elapsed.count() << "ms\n";

        std::vector<ToolKit_GEMM::Config> configs{
            // {16, 128, 198, 8},
            // {32, 128, 198, 8},
            // {64, 128, 198, 8},
            // {16, 256, 198, 8},
            // {32, 256, 198, 8},
            // {64, 256, 198, 8},
            // {16, 512, 198, 8},
            // {32, 512, 198, 8},
            // {64, 512, 198, 8},
            // {16, 1024, 198, 8},
            // {32, 1024, 198, 8},
            // {64, 1024, 198, 8},
            // {16, 2048, 198, 8},
            // {32, 2048, 198, 8},
            // {64, 2048, 198, 8},

            // {16, 128, 128, 8},
            // {32, 128, 128, 8},
            // {64, 128, 128, 8},
            // {16, 256, 128, 8},
            // {32, 256, 128, 8},
            // {64, 256, 128, 8},
            // {16, 512, 128, 8},
            // {32, 512, 128, 8},
            // {64, 512, 128, 8},
            // {16, 1024, 128, 8},
            // {32, 1024, 128, 8},
            // {64, 1024, 128, 8},
            // {16, 2048, 128, 8},
            // {32, 2048, 128, 8},
            // {64, 2048, 128, 8},

            // {16, 128, 256, 8},
            // {32, 128, 256, 8},
            // {64, 128, 256, 8},
            // {16, 256, 256, 8},
            // {32, 256, 256, 8},
            // {64, 256, 256, 8},
            // {16, 512, 256, 8},
            // {32, 512, 256, 8},
            // {64, 512, 256, 8},
            // {16, 1024, 256, 8},
            // {32, 1024, 256, 8},
            // {64, 1024, 256, 8},
            // {128, 1024, 256, 8},
            // {256, 1024, 256, 8},
            // {16, 2048, 256, 8},
            // {32, 2048, 256, 8},
            // {64, 2048, 256, 8},

            // {16, 128, 384, 8},
            // {32, 128, 384, 8},
            // {64, 128, 384, 8},
            // {16, 256, 384, 8},
            // {32, 256, 384, 8},
            // {64, 256, 384, 8},
            // {16, 512, 384, 8},
            // {32, 512, 384, 8},
            // {64, 512, 384, 8},
            {32, 2048, 256, 8},
            {32, 2048, 384, 8},
            {32, 2048, 512, 8},
            // {128, 1024, 384, 8},
            // {256, 1024, 384, 8},
            // {512, 1024, 384, 8},
            // {1024, 1024, 384, 8},
            // {2048, 1024, 384, 8},
            // {4096, 1024, 384, 8},
            // {16, 2048, 384, 8},
            // {32, 2048, 384, 8},
            // {64, 2048, 384, 8},
            // {2048, 2048, 384, 8},
            // {2048, 4096, 256, 8},
            // {2048, 4096, 384, 8},
            // {2048, 4096, 384, 8},
            // {3072, 4096, 384, 8},
            // {4096, 4096, 384, 8},

        };

        // int rep_warmup = 20;
        int rep_warmup = 1;
        // int rep_bench = 30;
        int rep_bench = 5;

        // Gemm gemm(configs.back().workers);
        Gemm_DoubleBuffer gemm(configs.back().workers);
        std::cout << "Start GEMM Warm-up...";
        for (int i = 0; i < rep_warmup; i++) {
            for (auto& config : configs) {
                gemm.set_blockDims(config.mc, config.nc, config.kc);
                gemm(A, B, C, M, N, K);
            }
        }
        std::cout << "Done\n";

        // --- Round-robin benchmark ---
        // Instead of shuffling once and then running each config for `rep`
        // consecutive repetitions (which re-introduces the same
        // "large contiguous block" ordering issue as an un-shuffled sweep,
        // just with the blocks themselves permuted), we shuffle the config
        // order fresh on every round and interleave all configs within each
        // round. Any residual thermal drift across the whole benchmark phase
        // is then spread evenly across all configs instead of concentrated
        // on whichever configs happen to land in the same contiguous block.
        std::vector<std::vector<ToolKit_GEMM::Benchmark>> results(configs.size());
        std::vector<size_t> indices(configs.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::random_device rdv;
        std::mt19937 rng(rdv());

        Logger logger("log/bench_runs.csv");
        size_t global_position = 0;

        std::cout << "Start GEMM benchmark...\n";
        for (int round = 0; round < rep_bench; round++) {
            std::shuffle(indices.begin(), indices.end(), rng);

            for (size_t idx : indices) {
                auto& config = configs[idx];
                gemm.set_blockDims(config.mc, config.nc, config.kc);

                ToolKit_GEMM::initMat(C, M * N);

                logger.record_start();

                std::atomic_thread_fence(std::memory_order_seq_cst);
                start = std::chrono::high_resolution_clock::now();
                std::atomic_thread_fence(std::memory_order_seq_cst);

                gemm(A, B, C, M, N, K);

                std::atomic_thread_fence(std::memory_order_seq_cst);
                end = std::chrono::high_resolution_clock::now();
                std::atomic_thread_fence(std::memory_order_seq_cst);

                alg_elapsed = end - start;
                results[idx].push_back({alg_elapsed.count()});
                ++global_position;

                logger.record_end(
                    round, global_position,
                    config.mc, config.nc, config.kc, config.workers,
                    alg_elapsed.count()
                );

                if (!ToolKit_GEMM::correct_check(GroundTruth, C, M * N)) {
                    throw std::runtime_error(
                        "Correctness check failed at (MC, NC, KC, workers) = ("
                        + std::to_string(config.mc) + ", "
                        + std::to_string(config.nc) + ", "
                        + std::to_string(config.kc) + ", "
                        + std::to_string(config.workers) + "), round "
                        + std::to_string(round)
                    );
                }
            }
            std::cout << "Round " << round + 1 << "/" << rep_bench << " done." << std::endl;
        }
        std::cout << "All done\n";

        std::vector<std::pair<ToolKit_GEMM::Config, ToolKit_GEMM::Benchmark>> perfs;
        for (size_t idx = 0; idx < configs.size(); idx++) {
            perfs.emplace_back(configs[idx], ToolKit_GEMM::get_median(results[idx]));
        }

        for (auto& perf : perfs) {
            std::cout   << "(MC, NC, KC, workers) = ("
                        << perf.first.mc << ", "
                        << perf.first.nc << ", "
                        << perf.first.kc << ", "
                        << perf.first.workers << ") | "
                        << "Time Elapsed Alg = "
                        << perf.second.alg_ms << "ms\n";
        }
        auto best_perf = ToolKit_GEMM::get_best_perf(perfs);
        std::cout << "Best Performance (MC, NC, KC) = ("
                  << best_perf.first.mc << ", "
                  << best_perf.first.nc << ", "
                  << best_perf.first.kc << ", "
                  << best_perf.first.workers << ") | "
                  << "Time Spanned Alg = "
                  << best_perf.second.alg_ms << "ms\n";

        std::string logname{"gemm_multiThreads_" +std::to_string(rep_bench) + "rep_roundrobin.log"};
        std::ofstream ofile("log/" + logname);
        if (ofile.is_open()) {
            ofile << "C(" << M << ", " << N << ") = ";
            ofile << "A(" << M << ", " << K << ") x B(" << K << ", " << N << ")\n";
            ofile << "With Warm-up x " << rep_warmup << " rep | "
                  << "Algorithm x " << rep_bench << " rep and take median.\n";
            for (auto& perf : perfs) {
                ofile   << "(MC, NC, KC, workers) = ("
                        << perf.first.mc << ", "
                        << perf.first.nc << ", "
                        << perf.first.kc << ", "
                        << perf.first.workers << ") | "
                        << "Time Elapsed Alg = "
                        << perf.second.alg_ms << "ms\n";
            }
            ofile   << "Best Performance (MC, NC, KC) = ("
                    << best_perf.first.mc << ", "
                    << best_perf.first.nc << ", "
                    << best_perf.first.kc << ", "
                    << best_perf.first.workers << ") | "
                    << "Time Spanned Alg = "
                    << best_perf.second.alg_ms << "ms\n";
        } else {
            std::cerr << "Error: Could not open the file!" << std::endl;
        }

        for (auto ptr : ptr_handler) {
            delete[] ptr;
        }
    }
    catch (const std::runtime_error& e) {
        std::cout << "Caught error: " << e.what() << "\n";
    }

}