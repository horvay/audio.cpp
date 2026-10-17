#include "mmxmx.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

#include <cstdlib>

namespace esimd = sycl::ext::intel::esimd;
namespace xmx   = sycl::ext::intel::esimd::xmx;

namespace {

constexpr int XQK = 32;  // values per Q8_0 block = DPAS K for int8
constexpr int XTN = 16;  // activation rows per tile = DPAS execution size on Xe2
constexpr int XTM = 8;   // weight rows per DPAS (repeat count)
constexpr int XKB = 2;   // Q8_0 blocks per weight load (64 bytes per row)

// x[n][k] f32 (row stride sx floats) -> per 16-row tile: xq[tile][b][k/4][16][4] int8, xd[tile][b][16] f32.
// Rows past n are zero.
void quantize_x_vnni(const float * x, size_t sx, int n, int K, int8_t * xq, float * xd, dpct::queue_ptr stream) {
    const int nb     = K / XQK;
    const int ntiles = (n + XTN - 1) / XTN;
    stream->parallel_for(sycl::range<3>(ntiles, nb, XTN), [=](sycl::id<3> id) {
        const int tile = id[0], b = id[1], r = id[2];
        const int row  = tile * XTN + r;
        const float * xr = x + (size_t) row * sx + b * XQK;
        float amax = 0.f;
        if (row < n) {
            for (int i = 0; i < XQK; i++) amax = sycl::fmax(amax, sycl::fabs(xr[i]));
        }
        const float d = amax / 127.f, inv = d > 0.f ? 1.f / d : 0.f;
        xd[((size_t) tile * nb + b) * XTN + r] = d;
        int8_t * q = xq + ((size_t) tile * nb + b) * XQK * XTN;
        for (int i = 0; i < XQK; i++) {
            q[(i / 4) * XTN * 4 + r * 4 + (i % 4)] = row < n ? (int8_t) sycl::round(xr[i] * inv) : (int8_t) 0;
        }
    });
}

// One hardware thread: 8 weight rows x 16 activation rows of one tile, over blocks [split * nbps, (split+1) * nbps).
// out[split][row of x][weight row] (row stride nrows); nsplit > 1 writes partials that a second kernel sums in order.
void q8_0_xmx_kernel(const int8_t * wq, const sycl::half * wd, const int8_t * xq, const float * xd, float * out,
                     int nrows, int K, int n, int nsplit, size_t out_split_stride, dpct::queue_ptr stream) {
    const int nb      = K / XQK;
    const int nbps    = nb / nsplit;
    const int ngroups = nrows / XTM;
    const int ntiles  = (n + XTN - 1) / XTN;
    stream->parallel_for(sycl::range<3>(ntiles, nsplit, ngroups), [=](sycl::id<3> tid) SYCL_ESIMD_KERNEL {
        using namespace esimd;
        const int tile  = (int) tid[0];
        const int split = (int) tid[1];
        const int row0  = (int) tid[2] * XTM;
        const int8_t * xqt = xq + (size_t) tile * nb * XQK * XTN;
        const float *  xdt = xd + (size_t) tile * nb * XTN;
        simd<float, XTM * XTN> acc = 0.f;
        simd<uint32_t, XTM> d_off = simd<uint32_t, XTM>(0, 1) * (uint32_t) (nb * sizeof(sycl::half));
        for (int b0 = split * nbps; b0 < (split + 1) * nbps; b0 += XKB) {
            simd<int8_t, XTM * XQK * XKB> A;
#pragma unroll
            for (int m = 0; m < XTM; m++) {
                A.template select<XQK * XKB, 1>(m * XQK * XKB) =
                    block_load<int8_t, XQK * XKB>(wq + (size_t) (row0 + m) * K + (size_t) b0 * XQK);
            }
#pragma unroll
            for (int kb = 0; kb < XKB; kb++) {
                const int b = b0 + kb;
                simd<int8_t, XQK * XTN> B  = block_load<int8_t, XQK * XTN>(xqt + (size_t) b * XQK * XTN);
                simd<float, XTN>        dx = block_load<float, XTN>(xdt + b * XTN);
                simd<float, XTM>        dw = gather<sycl::half, XTM>(wd + (size_t) row0 * nb + b, d_off);
                simd<int8_t, XTM * XQK> a  = A.template bit_cast_view<int8_t, XTM, XQK * XKB>()
                                                 .template select<XTM, 1, XQK, 1>(0, kb * XQK);
                simd<int, XTM * XTN>   c  = xmx::dpas<8, XTM, int, int8_t, int8_t>(B, a);
                simd<float, XTM * XTN> cf = c;
#pragma unroll
                for (int m = 0; m < XTM; m++) {
                    const float sw = dw[m];
                    acc.template select<XTN, 1>(m * XTN) += cf.template select<XTN, 1>(m * XTN) * (dx * sw);
                }
            }
        }
        // acc row m = 16 activation rows for weight row row0 + m -> out[x row][weight row]
        float * dst = out + (size_t) split * out_split_stride + (size_t) tile * XTN * nrows;
        simd<uint32_t, XTN> off  = simd<uint32_t, XTN>(0, 1) * (uint32_t) (nrows * sizeof(float));
        simd_mask<XTN>      mask = simd<int, XTN>(0, 1) < (n - tile * XTN);
#pragma unroll
        for (int m = 0; m < XTM; m++) {
            scatter<float, XTN>(dst + row0 + m, off, acc.template select<XTN, 1>(m * XTN).read(), mask);
        }
    });
}

constexpr int XHK  = 16;  // DPAS K for f16
constexpr int XHKB = 4;   // DPAS K steps per weight load (64 halves = 128 bytes per row)

// x[n][k] f32 -> per 16-row tile: xh[tile][k/2][16][2] f16 (VNNI pairs) and xs[tile][16] f32, where each row is divided
// by a power of two (exact) that brings its max |x| to at most 2^15, so no value overflows f16.
void convert_x_f16_vnni(const float * x, size_t sx, int n, int K, sycl::half * xh, float * xs, dpct::queue_ptr stream) {
    const int ntiles = (n + XTN - 1) / XTN;
    constexpr int WG = 256;
    stream->parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) ntiles * XTN, WG), sycl::range<2>(1, WG)),
                         [=](sycl::nd_item<2> it) {
        const int row  = (int) it.get_global_id(0);
        const int tile = row / XTN, r = row % XTN;
        const int lid  = (int) it.get_local_id(1);
        const float * xr = x + (size_t) row * sx;
        float amax = 0.f;
        if (row < n) {
            for (int k = lid; k < K; k += WG) amax = sycl::fmax(amax, sycl::fabs(xr[k]));
        }
        amax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
        float scale = 1.f;
        while (amax / scale > 32768.f && scale < 0x1p100f) scale *= 2.f;  // inf/NaN: give up at 2^100
        const float inv = 1.f / scale;
        if (lid == 0) xs[(size_t) tile * XTN + r] = scale;
        sycl::half * h = xh + (size_t) tile * K * XTN;
        for (int k = lid; k < K; k += WG) {
            h[(size_t) (k / 2) * XTN * 2 + r * 2 + (k % 2)] = row < n ? (sycl::half) (xr[k] * inv) : (sycl::half) 0.f;
        }
    });
}

// f16 weights: same tiling as q8_0_xmx_kernel, f16 x f16 DPAS accumulating in f32; ksteps of 16 over
// [split * kps, (split + 1) * kps).
void f16_xmx_kernel(const sycl::half * w, const sycl::half * xh, const float * xs, float * out, int nrows, int K, int n,
                    int nsplit, size_t out_split_stride, dpct::queue_ptr stream) {
    const int nk      = K / XHK;
    const int kps     = nk / nsplit;
    const int ngroups = nrows / XTM;
    const int ntiles  = (n + XTN - 1) / XTN;
    stream->parallel_for(sycl::range<3>(ntiles, nsplit, ngroups), [=](sycl::id<3> tid) SYCL_ESIMD_KERNEL {
        using namespace esimd;
        const int tile  = (int) tid[0];
        const int split = (int) tid[1];
        const int row0  = (int) tid[2] * XTM;
        const sycl::half * xht = xh + (size_t) tile * K * XTN;
        simd<float, XTM * XTN> acc = 0.f;
        for (int k0 = split * kps; k0 < (split + 1) * kps; k0 += XHKB) {
            simd<sycl::half, XTM * XHK * XHKB> A;
#pragma unroll
            for (int m = 0; m < XTM; m++) {
                A.template select<XHK * XHKB, 1>(m * XHK * XHKB) =
                    block_load<sycl::half, XHK * XHKB>(w + (size_t) (row0 + m) * K + (size_t) k0 * XHK);
            }
#pragma unroll
            for (int kk = 0; kk < XHKB; kk++) {
                simd<sycl::half, XHK * XTN> B = block_load<sycl::half, XHK * XTN>(xht + (size_t) (k0 + kk) * XHK * XTN);
                simd<sycl::half, XTM * XHK> a = A.template bit_cast_view<sycl::half, XTM, XHK * XHKB>()
                                                    .template select<XTM, 1, XHK, 1>(0, kk * XHK);
                acc = xmx::dpas<8, XTM, float, float, sycl::half, sycl::half>(acc, B, a);
            }
        }
        simd<float, XTN> s = block_load<float, XTN>(xs + (size_t) tile * XTN);
        float * dst = out + (size_t) split * out_split_stride + (size_t) tile * XTN * nrows;
        simd<uint32_t, XTN> off  = simd<uint32_t, XTN>(0, 1) * (uint32_t) (nrows * sizeof(float));
        simd_mask<XTN>      mask = simd<int, XTN>(0, 1) < (n - tile * XTN);
#pragma unroll
        for (int m = 0; m < XTM; m++) {
            simd<float, XTN> v = acc.template select<XTN, 1>(m * XTN);
            scatter<float, XTN>(dst + row0 + m, off, v * s, mask);
        }
    });
}

// Pick nsplit for about 2,048 threads, keeping at least min_steps K steps per split.
int pick_nsplit(int nrows, int ntiles, int nsteps, int step_align, int min_steps) {
    int nsplit = 1;
    while ((size_t) (nrows / XTM) * ntiles * nsplit < 2048 && nsteps % (nsplit * 2 * step_align) == 0 &&
           nsteps / (nsplit * 2) >= min_steps) {
        nsplit *= 2;
    }
    return nsplit;
}

void reduce_splits(const float * p, float * y, size_t count, int nsplit, size_t split_stride, dpct::queue_ptr stream) {
    stream->parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
        float s = 0.f;
        for (int k = 0; k < nsplit; k++) s += p[(size_t) k * split_stride + i[0]];
        y[i[0]] = s;
    });
}

}  // namespace

bool ggml_sycl_xmx_enabled() {
    static const bool enabled = [] {
        const char * v = getenv("GGML_SYCL_DISABLE_XMX");
        return v == nullptr || atoi(v) == 0;
    }();
    return enabled;
}

bool ggml_sycl_xmx_q8_0_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return ggml_sycl_xmx_enabled() && src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 &&
           dst->type == GGML_TYPE_F32 && src1->ne[1] >= 2 && src1->ne[1] <= 32 && src0->ne[2] == 1 &&
           src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 && ggml_is_contiguous(src0) &&
           src1->nb[0] == sizeof(float) && src0->ne[0] % (XQK * XKB) == 0 && src0->ne[1] % XTM == 0 &&
           ggml_is_contiguous(dst);
}

void ggml_sycl_mul_mat_q8_0_xmx(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst) {
    const int K     = (int) src0->ne[0];
    const int nrows = (int) src0->ne[1];
    const int n     = (int) src1->ne[1];
    const int nb    = K / XQK;
    const int ntiles = (n + XTN - 1) / XTN;
    dpct::queue_ptr stream = ctx.stream();

    // reordered Q8_0: qs (nrows * K int8), then d (nrows * nb half)
    const int8_t *     wq = (const int8_t *) src0->data;
    const sycl::half * wd = (const sycl::half *) (wq + (size_t) nrows * K);

    ggml_sycl_pool_alloc<int8_t> xq(ctx.pool(), (size_t) ntiles * nb * XQK * XTN);
    ggml_sycl_pool_alloc<float>  xd(ctx.pool(), (size_t) ntiles * nb * XTN);
    quantize_x_vnni((const float *) src1->data, src1->nb[1] / sizeof(float), n, K, xq.get(), xd.get(), stream);

    const int nsplit = pick_nsplit(nrows, ntiles, nb, XKB, 4);
    float * y = (float *) dst->data;
    if (nsplit == 1) {
        q8_0_xmx_kernel(wq, wd, xq.get(), xd.get(), y, nrows, K, n, 1, 0, stream);
        return;
    }
    const size_t split_stride = (size_t) ntiles * XTN * nrows;
    ggml_sycl_pool_alloc<float> part(ctx.pool(), split_stride * nsplit);
    q8_0_xmx_kernel(wq, wd, xq.get(), xd.get(), part.get(), nrows, K, n, nsplit, split_stride, stream);
    reduce_splits(part.get(), y, (size_t) n * nrows, nsplit, split_stride, stream);
}

bool ggml_sycl_xmx_f16_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return ggml_sycl_xmx_enabled() && src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 &&
           dst->type == GGML_TYPE_F32 && src1->ne[1] >= 2 && src1->ne[1] <= 32 && src0->ne[2] == 1 &&
           src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 && ggml_is_contiguous(src0) &&
           src1->nb[0] == sizeof(float) && src0->ne[0] % (XHK * XHKB) == 0 && src0->ne[1] % XTM == 0 &&
           ggml_is_contiguous(dst);
}

void ggml_sycl_mul_mat_f16_xmx(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               ggml_tensor * dst) {
    const int K      = (int) src0->ne[0];
    const int nrows  = (int) src0->ne[1];
    const int n      = (int) src1->ne[1];
    const int ntiles = (n + XTN - 1) / XTN;
    dpct::queue_ptr stream = ctx.stream();

    ggml_sycl_pool_alloc<sycl::half> xh(ctx.pool(), (size_t) ntiles * XTN * K);
    ggml_sycl_pool_alloc<float>      xs(ctx.pool(), (size_t) ntiles * XTN);
    convert_x_f16_vnni((const float *) src1->data, src1->nb[1] / sizeof(float), n, K, xh.get(), xs.get(), stream);

    const sycl::half * w = (const sycl::half *) src0->data;
    const int nsplit = pick_nsplit(nrows, ntiles, K / XHK, XHKB, 8);
    float * y = (float *) dst->data;
    if (nsplit == 1) {
        f16_xmx_kernel(w, xh.get(), xs.get(), y, nrows, K, n, 1, 0, stream);
        return;
    }
    const size_t split_stride = (size_t) ntiles * XTN * nrows;
    ggml_sycl_pool_alloc<float> part(ctx.pool(), split_stride * nsplit);
    f16_xmx_kernel(w, xh.get(), xs.get(), part.get(), nrows, K, n, nsplit, split_stride, stream);
    reduce_splits(part.get(), y, (size_t) n * nrows, nsplit, split_stride, stream);
}
