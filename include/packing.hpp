#pragma once
#include <cstddef>
#include <algorithm>


namespace Packing {
    /**
     * @brief Assign values from a submatrix of A into corresponding tile in blockA_packed.
     * 
     * @param A             Points to the begining of the submatrix.
     * @param tileA         Points to the beginning of the tile.
     * @param stride_a(lda) Number of elements to skip in the submatrix to access next row.
     * @param mr            Number of rows in the submatrix (1 <= mr <= MR).
     * @param kc            Number of columns in the submatrix (1 <= kc <= KC).
     * @param MR            Number of rows of kernel matrix.
     */
    void pack_tileA(const float* __restrict A, float* __restrict tileA,
                    size_t stride_a, size_t mr, size_t kc, size_t MR
    ) {
        for (size_t j = 0; j < kc; j++) {
            for (size_t i = 0; i < MR; i++) {
                if (i < mr) {
                    *tileA++ = A[i * stride_a + j];
                } else {
                    *tileA++ = 0.0f;
                }
            }
        }
    }

    /**
     * @brief Assign values from a submatrix of A into blockA_packed.
     * 
     * @param A             Points to the begining of the submatrix.
     * @param blockA        Points to the beginning of the tile.
     * @param stride_a(lda) Number of elements to skip in the submatrix to access next row.
     * @param mc            Number of rows in the submatrix (1 <= mr <= MC).
     * @param kc            Number of columns in the submatrix (1 <= kc <= KC).
     * @param MR            Number of rows of kernel matrix.
     * 
     */
    void pack_blockA(const float* __restrict A, float* __restrict blockA,
                    size_t stride_a, size_t mc, size_t kc, size_t MR
    ) {
        for (size_t i = 0; i < mc; i += MR) {
            const size_t mr = std::min(mc - i, MR);

            pack_tileA(A + i * stride_a, blockA, stride_a, mr, kc, MR);

            blockA += MR * kc;
        }
    }

    /**
     * @brief Assign values from a submatrix of B into corresponding tile in blockB_packed.
     * 
     * @param B             Points to the begining of the submatrix.
     * @param tileB         Points to the beginning of the tile.
     * @param stride_b(ldb) Number of elements to skip in the submatrix to access next row.
     * @param kc            Number of rows in the submatrix (1 <= kc <= KC).
     * @param nr            Number of columns in the submatrix (1 <= nr <= NR).
     * @param NR            Number of columns of kernel matrix.
     * 
     * Note: the submatrix of B has actual dimension (kc, nr), but the tile in blockB_packed
     *       will takes up to kc x NR instead, for register size reason.
     */
    void pack_tileB(const float* __restrict B, float* __restrict tileB,
                    size_t stride_b, size_t kc, size_t nr, size_t NR
    ) {
        for (size_t i = 0; i < kc; i++) {
            for (size_t j = 0; j < NR; j++) {
                if (j < nr) {
                    *tileB++ = B[i * stride_b + j];
                } else {
                    *tileB++ = 0;
                }
            }
        }
    }

    /**
     * @brief Assign values from a submatrix of B into blockB_packed.
     * 
     * @param B             Points to the begining of the submatrix.
     * @param blockB        Points to the beginning of the blockB_packed.
     * @param stride_b(ldb) Number of elements to skip in the submatrix to access next row.
     * @param kc            Number of rows in the submatrix (1 <= kc <= KC).
     * @param nc            Number of columns in the submatrix (1 <= nc <= NC).
     * @param NR            Number of columns of kernel matrix.
     */
    void pack_blockB(const float* __restrict B, float* __restrict blockB,
                    size_t stride_b, size_t kc, size_t nc, size_t NR
    ) {
        for (size_t j = 0; j < nc; j += NR) {
            size_t nr = std::min(nc - j, NR);

            pack_tileB(B + j, blockB, stride_b, kc, nr, NR);

            blockB += kc * NR;
        }
    }

    /**
     * @brief Assign values from a submatrix of C into tileC_packed.
     * 
     * @param C             Points to the begining of the submatrix.
     * @param tileC         Points to the beginning of the tileC_packed.
     * @param stride_c(ldc) Number of elements to skip in the submatrix to access next row.
     * @param mr            Number of rows in the submatrix (1 <= mr <= MR).
     * @param nr            Number of columns in the submatrix (1 <= nr <= NR).
     * @param MR            Number of rows of kernel matrix.
     * @param NR            Number of columns of kernel matrix.
     * 
     * Note: the submatrix of C has actual dimension (mr, nr), but the tileC_packed
     *       will takes up to MR x NR instead, for register size reason.
     */
    void pack_tileC(float* __restrict C, float* __restrict tileC,
                    size_t stride_c, size_t mr, size_t nr, size_t MR, size_t NR
    ) {
        std::fill(tileC, tileC + MR * NR, 0);
        for (size_t i = 0; i < mr; i++) {
            for (size_t j = 0; j < nr; j++) {
                tileC[i * NR + j] = C[i * stride_c + j];
            }
        }
    }

    /**
     * @brief Accumulate values of tileC_packed back to the submatrix of C.
     * 
     * @param C             Points to the begining of the submatrix.
     * @param tileC         Points to the beginning of the tileC_packed.
     * @param stride_c(ldc) Number of elements to skip in the submatrix to access next row.
     * @param mr            Number of rows in the submatrix (1 <= mr <= MR).
     * @param nr            Number of columns in the submatrix (1 <= nr <= NR).
     * @param NR            Number of columns of kernel matrix.
     */
    void unpack_tileC(float* __restrict C, float* __restrict tileC,
                    size_t stride_c, size_t mr, size_t nr, size_t NR
    ) {
        for (size_t i = 0; i < mr; i++) {
            for (size_t j = 0; j < nr; j++) {
                C[i * stride_c + j] = tileC[i * NR + j];
            }
        }
    }
}