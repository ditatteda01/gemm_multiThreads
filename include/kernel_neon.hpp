#pragma once
#include <cstddef>
#include <arm_neon.h>

namespace Kernel {
    void kernel_fma_4x16_edge(
        float* __restrict tileA_packed,
        float* __restrict tileB_packed,
        float* __restrict tileC_packed,
        size_t kc
    ){
        float32x4_t c00 = vld1q_f32(tileC_packed);
        float32x4_t c01 = vld1q_f32(tileC_packed + 4);
        float32x4_t c02 = vld1q_f32(tileC_packed + 8);
        float32x4_t c03 = vld1q_f32(tileC_packed + 12);
        float32x4_t c10 = vld1q_f32(tileC_packed + 16);
        float32x4_t c11 = vld1q_f32(tileC_packed + 20);
        float32x4_t c12 = vld1q_f32(tileC_packed + 24);
        float32x4_t c13 = vld1q_f32(tileC_packed + 28);
        float32x4_t c20 = vld1q_f32(tileC_packed + 32);
        float32x4_t c21 = vld1q_f32(tileC_packed + 36);
        float32x4_t c22 = vld1q_f32(tileC_packed + 40);
        float32x4_t c23 = vld1q_f32(tileC_packed + 44);
        float32x4_t c30 = vld1q_f32(tileC_packed + 48);
        float32x4_t c31 = vld1q_f32(tileC_packed + 52);
        float32x4_t c32 = vld1q_f32(tileC_packed + 56);
        float32x4_t c33 = vld1q_f32(tileC_packed + 60);

        float32x4_t a = {};
        float32x4_t b0 = {};
        float32x4_t b1 = {};
        float32x4_t b2 = {};
        float32x4_t b3 = {};

        for (size_t k = 0; k < kc; k++) {
            a = vld1q_f32(tileA_packed);

            b0 = vld1q_f32(tileB_packed);
            b1 = vld1q_f32(tileB_packed + 4);
            b2 = vld1q_f32(tileB_packed + 8);
            b3 = vld1q_f32(tileB_packed + 12);

            c00 = vfmaq_laneq_f32(c00, b0, a, 0);
            c01 = vfmaq_laneq_f32(c01, b1, a, 0);
            c02 = vfmaq_laneq_f32(c02, b2, a, 0);
            c03 = vfmaq_laneq_f32(c03, b3, a, 0);
            
            c10 = vfmaq_laneq_f32(c10, b0, a, 1);
            c11 = vfmaq_laneq_f32(c11, b1, a, 1);
            c12 = vfmaq_laneq_f32(c12, b2, a, 1);
            c13 = vfmaq_laneq_f32(c13, b3, a, 1);

            c20 = vfmaq_laneq_f32(c20, b0, a, 2);
            c21 = vfmaq_laneq_f32(c21, b1, a, 2);
            c22 = vfmaq_laneq_f32(c22, b2, a, 2);
            c23 = vfmaq_laneq_f32(c23, b3, a, 2);

            c30 = vfmaq_laneq_f32(c30, b0, a, 3);
            c31 = vfmaq_laneq_f32(c31, b1, a, 3);
            c32 = vfmaq_laneq_f32(c32, b2, a, 3);
            c33 = vfmaq_laneq_f32(c33, b3, a, 3);

            tileA_packed += 4;
            tileB_packed += 16;
        }

        vst1q_f32(tileC_packed      , c00);
        vst1q_f32(tileC_packed + 4  , c01);
        vst1q_f32(tileC_packed + 8  , c02);
        vst1q_f32(tileC_packed + 12 , c03);
        vst1q_f32(tileC_packed + 16 , c10);
        vst1q_f32(tileC_packed + 20 , c11);
        vst1q_f32(tileC_packed + 24 , c12);
        vst1q_f32(tileC_packed + 28 , c13);
        vst1q_f32(tileC_packed + 32 , c20);
        vst1q_f32(tileC_packed + 36 , c21);
        vst1q_f32(tileC_packed + 40 , c22);
        vst1q_f32(tileC_packed + 44 , c23);
        vst1q_f32(tileC_packed + 48 , c30);
        vst1q_f32(tileC_packed + 52 , c31);
        vst1q_f32(tileC_packed + 56 , c32);
        vst1q_f32(tileC_packed + 60 , c33);
    }

    void kernel_fma_4x16(
        float* __restrict tileA_packed,
        float* __restrict tileB_packed,
        float* __restrict C,
        size_t stride_c, size_t kc
    ){
        float32x4_t c00 = vld1q_f32(C);
        float32x4_t c01 = vld1q_f32(C + 4);
        float32x4_t c02 = vld1q_f32(C + 8);
        float32x4_t c03 = vld1q_f32(C + 12);
        float32x4_t c10 = vld1q_f32(C + stride_c);
        float32x4_t c11 = vld1q_f32(C + stride_c + 4);
        float32x4_t c12 = vld1q_f32(C + stride_c + 8);
        float32x4_t c13 = vld1q_f32(C + stride_c + 12);
        float32x4_t c20 = vld1q_f32(C + 2 * stride_c);
        float32x4_t c21 = vld1q_f32(C + 2 * stride_c + 4);
        float32x4_t c22 = vld1q_f32(C + 2 * stride_c + 8);
        float32x4_t c23 = vld1q_f32(C + 2 * stride_c + 12);
        float32x4_t c30 = vld1q_f32(C + 3 * stride_c);
        float32x4_t c31 = vld1q_f32(C + 3 * stride_c + 4);
        float32x4_t c32 = vld1q_f32(C + 3 * stride_c + 8);
        float32x4_t c33 = vld1q_f32(C + 3 * stride_c + 12);

        float32x4_t a = {};
        float32x4_t b0 = {};
        float32x4_t b1 = {};
        float32x4_t b2 = {};
        float32x4_t b3 = {};

        for (size_t k = 0; k < kc; k++) {
            a = vld1q_f32(tileA_packed);

            b0 = vld1q_f32(tileB_packed);
            b1 = vld1q_f32(tileB_packed + 4);
            b2 = vld1q_f32(tileB_packed + 8);
            b3 = vld1q_f32(tileB_packed + 12);

            c00 = vfmaq_laneq_f32(c00, b0, a, 0);
            c01 = vfmaq_laneq_f32(c01, b1, a, 0);
            c02 = vfmaq_laneq_f32(c02, b2, a, 0);
            c03 = vfmaq_laneq_f32(c03, b3, a, 0);
            
            c10 = vfmaq_laneq_f32(c10, b0, a, 1);
            c11 = vfmaq_laneq_f32(c11, b1, a, 1);
            c12 = vfmaq_laneq_f32(c12, b2, a, 1);
            c13 = vfmaq_laneq_f32(c13, b3, a, 1);

            c20 = vfmaq_laneq_f32(c20, b0, a, 2);
            c21 = vfmaq_laneq_f32(c21, b1, a, 2);
            c22 = vfmaq_laneq_f32(c22, b2, a, 2);
            c23 = vfmaq_laneq_f32(c23, b3, a, 2);

            c30 = vfmaq_laneq_f32(c30, b0, a, 3);
            c31 = vfmaq_laneq_f32(c31, b1, a, 3);
            c32 = vfmaq_laneq_f32(c32, b2, a, 3);
            c33 = vfmaq_laneq_f32(c33, b3, a, 3);

            tileA_packed += 4;
            tileB_packed += 16;
        }

        vst1q_f32(C , c00);
        vst1q_f32(C + 4, c01);
        vst1q_f32(C + 8, c02);
        vst1q_f32(C + 12, c03);
        vst1q_f32(C + stride_c, c10);
        vst1q_f32(C + stride_c + 4, c11);
        vst1q_f32(C + stride_c + 8, c12);
        vst1q_f32(C + stride_c + 12, c13);
        vst1q_f32(C + 2 * stride_c, c20);
        vst1q_f32(C + 2 * stride_c + 4, c21);
        vst1q_f32(C + 2 * stride_c + 8, c22);
        vst1q_f32(C + 2 * stride_c + 12, c23);
        vst1q_f32(C + 3 * stride_c, c30);
        vst1q_f32(C + 3 * stride_c + 4, c31);
        vst1q_f32(C + 3 * stride_c + 8, c32);
        vst1q_f32(C + 3 * stride_c + 12, c33);
    }
}