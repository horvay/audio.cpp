//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "conv.hpp"

static  void conv_transpose_1d_kernel(
        const int s0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2,
        const int src1_ne0, const int dst_ne0,
        const float * src0, const float * src1,  float * dst,
        const sycl::nd_item<3> &item_ct1) {
    int global_index = item_ct1.get_local_id(2) +
                       item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (global_index >= output_size) {
        return;
    }

    int out_index = global_index / dst_ne0;
    const int idx = global_index % dst_ne0;

    // Only inputs i with i*s0 <= idx < i*s0 + K touch this output: i in [ceil((idx - K + 1) / s0), idx / s0].
    // (The upstream kernel scanned all src1_ne0 inputs per output and skipped the rest: O(length) per output,
    // which made long codec decodes take minutes.)
    const int i_lo = idx >= src0_ne0 ? (idx - src0_ne0) / s0 + 1 : 0;
    const int i_hi = sycl::min(idx / s0, src1_ne0 - 1);

    float accumulator = 0;

    for (int c = 0; c < src0_ne2; c++) {
        const int kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        const int input_offset = src1_ne0 * c;

        for (int i = i_lo; i <= i_hi; i++) {
            accumulator += src0[kernel_offset + idx - i*s0] * src1[input_offset + i];
        }
    }
    dst[global_index] = accumulator;
}

static void conv_transpose_1d_f32_f32_sycl(
    const int s0, const int output_size,
    const int src0_ne0, const int src0_ne1, const int src0_ne2,
    const int src1_ne0, const int dst_ne0,
    const float *src0, const float *src1, float *dst,
    const queue_ptr& stream) {

    const int num_blocks = (output_size + SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE;
    const sycl::range<3> block_dims(1, 1, SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, 1, num_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(
            block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv_transpose_1d_kernel(
                s0, output_size,
                src0_ne0, src0_ne1, src0_ne2,
                src1_ne0, dst_ne0,
                src0, src1, dst, item_ct1);
        });
}

void ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];
    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;

    float * dst_d = (float *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const int32_t * opts = (const int32_t *)dst->op_params;

    const int s0 = opts[0];

    const int64_t output_size = ggml_nelements(dst);

    // As a GEMM plus overlap-add: col[i][co*K + t] = sum_ci x[ci][i] * w[ci][co][t] (oneMKL, one call for every tap and
    // input position), then out[co][j] = sum over i*s0 + t == j of col[i][co*K + t]. The direct kernel below re-reads
    // the weights for every output sample; the decoders of audio codecs (Fish DAC) spent seconds per take in it.
    // GGML_SYCL_CONVT_NAIVE=1 forces the direct kernel.
    static const bool naive = getenv("GGML_SYCL_CONVT_NAIVE") != nullptr;
    const int64_t K = src0->ne[0], Cout = src0->ne[1], Cin = src0->ne[2], L = src1->ne[0], OL = dst->ne[0];
    if (!naive && src1->ne[1] == Cin && src1->ne[2] == 1 && src1->ne[3] == 1 && dst->ne[1] == Cout && dst->ne[2] == 1 &&
        K * Cout * L <= (int64_t) 1 << 31) {
        ggml_sycl_pool_alloc<float> col(ctx.pool(), (size_t) K * Cout * L);
        float * colp = col.get();
        // column-major: col (L x K*Cout, ld L) = x^T (L x Cin, ld L) * w^T, w viewed as (K*Cout x Cin, ld K*Cout)
        const float alpha = 1.f, beta = 0.f;
        SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
            *stream, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, L, K * Cout, Cin, alpha, src1_d, L,
            src0_d, K * Cout, beta, colp, L)));
        stream->parallel_for(sycl::range<2>(Cout, OL), [=](sycl::id<2> id) {
            const int64_t co = id[0], j = id[1];
            float acc = 0.f;
            for (int64_t t = j % s0; t < K; t += s0) {
                const int64_t i = (j - t) / s0;
                if (i >= 0 && i < L) acc += colp[(co * K + t) * L + i];
            }
            dst_d[co * OL + j] = acc;
        });
        return;
    }

    conv_transpose_1d_f32_f32_sycl(s0, output_size,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

