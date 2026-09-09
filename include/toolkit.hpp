#pragma once
#include <cstddef>
#include <algorithm>
#include <functional>
#include <random>
#include <vector>
#include <limits>
#include <chrono>

namespace ToolKit_GEMM {
    struct Config {
        size_t mc, nc, kc, workers;
    };

    struct Benchmark {
        double alg_ms;
    };

    std::pair<Config, Benchmark> get_best_perf(const std::vector<std::pair<Config, Benchmark>>& vec) {
        auto it = std::min_element(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
            return a.second.alg_ms < b.second.alg_ms;
        });
        return *it;
    }

    void initMat(float* mat, size_t dim) {
        std::fill(mat, mat + dim, 0.0f);
    }
    
    /**
     * @brief Check that `candidate` matches `target` within float32-appropriate tolerance.
     *
     * The naive reference matmul and the blocked/packed/NEON-FMA implementation
     * accumulate in different orders, so bit-exact equality is not expected even
     * when both are "correct". We use a combined relative+absolute tolerance
     * (similar in spirit to numpy.allclose) instead of comparing the sum of
     * absolute differences against std::numeric_limits<double>::epsilon(), which
     * is far too strict for ~M*N float32 accumulations and would reject correct
     * results (or silently pass only by coincidence).
     *
     * @param rtol  Relative tolerance (default 1e-4, consistent with prior
     *              float32 validation tolerances used on the Python side).
     * @param atol  Absolute tolerance floor for near-zero values (default 1e-5).
     */
    bool correct_check(const float* target, const float* candidate, size_t cnt) {
        double rtol = 1e-4, atol = 1e-5;
        for (size_t i = 0; i < cnt; i++) {
            double t = target[i];
            double c = candidate[i];
            double diff = std::abs(t - c);
            double allowed = atol + rtol * std::abs(t);
            if (diff > allowed) {
                return false;
            }
        }
        return true;
    }
    
    void matmul(const float* __restrict A, const float* __restrict B, float* __restrict C, size_t M, size_t N, size_t K) {
        for (size_t i = 0; i < M; i++) {
            for (size_t k = 0; k < K; k++) {
                for (size_t j = 0; j < N; j++) {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }
    }
    
    double time_pack_block(
        std::function<void(const float*, float*, size_t, size_t, size_t)> func,
        const float* inMat, float* outMat, size_t stride, size_t rows, size_t cols, size_t spec
    ) {
        auto start = std::chrono::high_resolution_clock::now();
        func(inMat, outMat, stride, rows, cols);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count();
    }
    
    Benchmark get_median(std::vector<Benchmark>& nums) {
        size_t size = nums.size();
        auto mid_it = nums.begin() + size / 2;
        std::nth_element(
            nums.begin(),
            mid_it,
            nums.end(),
            [](const Benchmark& a, const Benchmark& b) {
                return a.alg_ms < b.alg_ms;
            }
        );
        if (size % 2 != 0) {
            return *mid_it;
        } else {
            auto left_mid_it = std::max_element(nums.begin(), mid_it, [](const Benchmark& a, const Benchmark& b) {
                return a.alg_ms < b.alg_ms;}
            );
            return {(mid_it->alg_ms + left_mid_it->alg_ms) / 2};
        }
    }
}