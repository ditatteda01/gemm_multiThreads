#include <stdexcept>
#include "gemm.hpp"
#include "kernel_neon.hpp"
#include "packing.hpp"

void Gemm::operator()(
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict C,
    size_t M, size_t N, size_t K
) {
    for (size_t jc = 0; jc < N; jc += NC) {
        size_t nc = std::min(N - jc, NC);

        for (size_t kp = 0; kp < K; kp += KC) {
            size_t kc = std::min(K - kp, KC);
            Packing::pack_blockB(
                B + kp * N + jc,
                blockB_packed.data(),
                N, kc, nc, NR
            );

            for (size_t ic = 0; ic < M; ic += MC) {
                size_t mc = std::min(M - ic, MC);

                // Capture local variables, and implicitly 'this'
                TPool.submit([=]() {
                    thread_local std::vector<float> blockA_packed;
                    thread_local std::vector<float> tileC_packed(MR * NR);
                    
                    if (blockA_packed.size() < MC * KC) {
                        blockA_packed.resize(MC * KC);
                    }

                    Packing::pack_blockA(
                        A + ic * K + kp,
                        blockA_packed.data(),
                        K, mc, kc, MR
                    );

                    for (size_t ir = 0; ir < mc; ir += MR) {
                        size_t mr = std::min(mc - ir, MR);

                        for (size_t jr = 0; jr < nc; jr += NR) {
                            size_t nr = std::min(nc - jr, NR);

                            if (mr == MR && nr == NR) {
                                Kernel::kernel_fma_4x16(
                                    blockA_packed.data() + ir * kc,
                                    blockB_packed.data() + jr * kc,
                                    C + (ic + ir) * N + jc + jr,
                                    N, kc
                                );
                            } else {
                                float* C_ = C + (ic + ir) * N + jc + jr;
                                Packing::pack_tileC(
                                    C_,
                                    tileC_packed.data(),
                                    N, mr, nr, MR, NR
                                );
                                Kernel::kernel_fma_4x16_edge(
                                    blockA_packed.data() + ir * kc,
                                    blockB_packed.data() + jr * kc,
                                    tileC_packed.data(),
                                    kc
                                );
                                Packing::unpack_tileC(
                                    C_,
                                    tileC_packed.data(),
                                    N, mr, nr, NR
                                );
                            }
                        }
                    }
                });
            }
            TPool.wait_allworks_done();
        }
    }
}

void Gemm_DoubleBuffer::operator()(
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict C,
    size_t M, size_t N, size_t K
) {
    size_t offset;
    short bufferID = 0;

    for (size_t jc = 0; jc < N; jc += NC) {
        size_t nc = std::min(N - jc, NC);

        for (size_t kp = 0; kp < K; kp += KC) {
            size_t kc = std::min(K - kp, KC);

            offset = bufferID * (KC * NC);
            bufferID ^= 1;

            Packing::pack_blockB(
                B + kp * N + jc,
                blockB_packed.data() + offset,
                N, kc, nc, NR
            );

            TPool.wait_allworks_done();

            for (size_t ic = 0; ic < M; ic += MC) {
                size_t mc = std::min(M - ic, MC);

                // Capture local variables, and implicitly 'this'
                TPool.submit([=]() {
                    thread_local std::vector<float> blockA_packed;
                    
                    if (blockA_packed.size() < MC * KC) {
                        blockA_packed.resize(MC * KC);
                    }

                    Packing::pack_blockA(
                        A + ic * K + kp,
                        blockA_packed.data(),
                        K, mc, kc, MR
                    );

                    for (size_t ir = 0; ir < mc; ir += MR) {
                        size_t mr = std::min(mc - ir, MR);

                        for (size_t jr = 0; jr < nc; jr += NR) {
                            size_t nr = std::min(nc - jr, NR);

                            if (mr == MR && nr == NR) {
                                Kernel::kernel_fma_4x16(
                                    blockA_packed.data() + ir * kc,
                                    blockB_packed.data() + offset + jr * kc,
                                    C + (ic + ir) * N + jc + jr,
                                    N, kc
                                );
                            } else {
                                alignas(128) thread_local float tileC_packed[MR * NR];
                                float* C_ = C + (ic + ir) * N + jc + jr;
                                Packing::pack_tileC(
                                    C_,
                                    tileC_packed,
                                    N, mr, nr, MR, NR
                                );
                                Kernel::kernel_fma_4x16_edge(
                                    blockA_packed.data() + ir * kc,
                                    blockB_packed.data() + offset + jr * kc,
                                    tileC_packed,
                                    kc
                                );
                                Packing::unpack_tileC(
                                    C_,
                                    tileC_packed,
                                    N, mr, nr, NR
                                );
                            }
                        }
                    }
                });
            }
        }
    }
    TPool.wait_allworks_done();
}