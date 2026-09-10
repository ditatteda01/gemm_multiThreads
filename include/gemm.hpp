#pragma once
#include "ThreadPool.hpp"

constexpr size_t MR = 4;
constexpr size_t NR = 16;

class Gemm {
public:
    explicit Gemm(size_t workers = 4, size_t iMC = 16, size_t iNC = 1024, size_t iKC = 128)
        : TPool(workers)
    {
        set_blockDims(iMC, iNC, iKC);
    }

    void operator()(
        const float* __restrict A,
        const float* __restrict B,
        float* __restrict C,
        size_t M,
        size_t N,
        size_t K
    );

    void set_blockDims(size_t iMC, size_t iNC, size_t iKC) {
        MC = iMC;
        NC = iNC;
        KC = iKC;
        if (blockB_packed.size() < KC * NC) {
            blockB_packed.resize(KC * NC);
        }
    }
    
private:
    ThreadPool TPool;
    std::vector<float> blockB_packed;
    size_t MC;
    size_t NC;
    size_t KC;
};

class Gemm_DoubleBuffer {
public:
    explicit Gemm_DoubleBuffer(size_t workers = 4, size_t iMC = 16, size_t iNC = 1024, size_t iKC = 128)
        : TPool(workers)
    {
        set_blockDims(iMC, iNC, iKC);
    }

    void operator()(
        const float* __restrict A,
        const float* __restrict B,
        float* __restrict C,
        size_t M,
        size_t N,
        size_t K
    );

    void set_blockDims(size_t iMC, size_t iNC, size_t iKC) {
        MC = iMC;
        NC = iNC;
        KC = iKC;
        if (blockB_packed.size() < 2 * KC * NC) {
            blockB_packed.resize(2 * KC * NC);
        }
    }
    
private:
    ThreadPool TPool;
    std::vector<float> blockB_packed;
    size_t MC;
    size_t NC;
    size_t KC;
};

