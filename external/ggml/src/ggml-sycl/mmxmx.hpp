// Small-batch matmuls on the Intel matrix engines (XMX) through ESIMD DPAS, for the batched decode steps of models
// such as Drama 3 (2..32 activation rows): the weights are read once from memory for all rows, instead of once per
// row (mat-vec paths) or through tiles sized for big batches.
//
// F16: f16 x f16 DPAS accumulating in f32; activations scaled per row by an exact power of two to stay in f16 range.
// Q8_0: int8 x int8 DPAS. The weights must be in the reordered layout (all qs, then all d; see reorder_qw_q8_0).
// Activations are quantized per 32 values into the VNNI-packed layout DPAS reads (same activation rounding as the
// q8_1 paths). Disable with GGML_SYCL_DISABLE_XMX=1.
#pragma once

#include "common.hpp"

bool ggml_sycl_xmx_enabled();

// True if dst = src0 x src1 can take ggml_sycl_mul_mat_q8_0_xmx (shape and type checks only; the caller reorders src0).
bool ggml_sycl_xmx_q8_0_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// src0: Q8_0, reordered; src1: f32 [K, n<=32]; dst: f32 [rows, n].
void ggml_sycl_mul_mat_q8_0_xmx(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst);

bool ggml_sycl_xmx_f16_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// src0: F16 [K, rows]; src1: f32 [K, n<=32]; dst: f32 [rows, n].
void ggml_sycl_mul_mat_f16_xmx(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               ggml_tensor * dst);
