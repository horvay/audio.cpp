// Prototype: small-batch Q8_0 matmul on the Arc Pro B70's matrix engines (ESIMD int8 DPAS), y[n][r] = sum_k W[r][k] x[n][k]
// for n <= 16 tokens, the shape of Drama 3's batched decode steps. Weights stay int8 with one f16 scale per 32 values
// (Q8_0, split into a qs array and a transposed scale array); activations are quantized per 32 values into the
// VNNI-packed layout DPAS reads. Each hardware thread computes RT row tiles of 8 rows x 16 tokens over the whole K.
// Checks against a CPU reference and prints time and effective weight bandwidth per shape.
//   icpx -O2 -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" xmx_q8_bench.cpp
// Run under b70-lease (it uses the B70).
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;
namespace xmx   = sycl::ext::intel::esimd::xmx;

constexpr int QK = 32;   // values per scale block = DPAS K for int8
constexpr int TN = 16;   // tokens per tile = DPAS execution size
constexpr int TM = 8;    // rows per DPAS (repeat count)

// x[n][k] f32 (n < ntok, rows padded with zeros to 16) -> xq[b][k/4][n][k%4] int8 + xd[b][n] f32
static void quantize_x(sycl::queue & q, const float * x, int ntok, int K, int8_t * xq, float * xd) {
    const int nb = K / QK;
    q.parallel_for(sycl::range<2>(nb, TN), [=](sycl::id<2> id) {
        const int b = id[0], n = id[1];
        float amax = 0.f;
        if (n < ntok) for (int i = 0; i < QK; i++) amax = sycl::fmax(amax, sycl::fabs(x[(size_t) n * K + b * QK + i]));
        const float d = amax / 127.f, inv = d > 0.f ? 1.f / d : 0.f;
        xd[b * TN + n] = d;
        for (int i = 0; i < QK; i++) {
            const float v = n < ntok ? x[(size_t) n * K + b * QK + i] : 0.f;
            xq[(size_t) b * QK * TN + (i / 4) * TN * 4 + n * 4 + (i % 4)] = (int8_t) sycl::round(v * inv);
        }
    });
}

// One hardware thread: RT row tiles (8 rows each) x 16 tokens over the K range [split * nbps, (split + 1) * nbps) blocks,
// KB blocks (KB * 32 bytes per weight row) per load. nsplit > 1 writes partial sums part[split][n][row]; a second
// kernel adds them in a fixed order (deterministic, unlike atomics).
template <int RT, int KB>
static sycl::event matmul_xmx(sycl::queue & q, const int8_t * wq, const sycl::half * wd_t, const int8_t * xq,
                              const float * xd, float * y, float * part, int nrows, int K, int ntok, int nsplit) {
    const int nb       = K / QK;
    const int nbps     = nb / nsplit;  // blocks per split (nb % (nsplit * KB) == 0 is required)
    const int ngroups  = nrows / (TM * RT);
    float *   out      = nsplit > 1 ? part : y;
    auto e = q.parallel_for(sycl::range<2>(nsplit, ngroups), [=](sycl::id<2> tid) SYCL_ESIMD_KERNEL {
        using namespace esimd;
        const int split = (int) tid[0];
        const int row0  = (int) tid[1] * TM * RT;
        simd<float, TM * TN> acc[RT];
        for (int t = 0; t < RT; t++) acc[t] = 0.f;
        for (int b0 = split * nbps; b0 < (split + 1) * nbps; b0 += KB) {
            simd<int8_t, TM * QK * KB> A[RT];
#pragma unroll
            for (int t = 0; t < RT; t++)
#pragma unroll
                for (int m = 0; m < TM; m++)
                    A[t].template select<QK * KB, 1>(m * QK * KB) =
                        block_load<int8_t, QK * KB>(wq + (size_t) (row0 + t * TM + m) * K + b0 * QK);
#pragma unroll
            for (int kb = 0; kb < KB; kb++) {
                const int b = b0 + kb;
                simd<int8_t, QK * TN> B  = block_load<int8_t, QK * TN>(xq + (size_t) b * QK * TN);
                simd<float, TN>       dx = block_load<float, TN>(xd + b * TN);
                simd<float, TM * RT>  dw = block_load<sycl::half, TM * RT>(wd_t + (size_t) b * nrows + row0);
#pragma unroll
                for (int t = 0; t < RT; t++) {
                    // rows m of A[t] hold KB blocks each: pick block kb of every row -> 8 x 32 tile
                    simd<int8_t, TM * QK> a = A[t].template bit_cast_view<int8_t, TM, QK * KB>()
                                                  .template select<TM, 1, QK, 1>(0, kb * QK);
                    simd<int, TM * TN> c = xmx::dpas<8, TM, int, int8_t, int8_t>(B, a);
                    simd<float, TM * TN> cf = c;
#pragma unroll
                    for (int m = 0; m < TM; m++) {
                        const float sw = dw[t * TM + m];
                        acc[t].template select<TN, 1>(m * TN) += cf.template select<TN, 1>(m * TN) * (dx * sw);
                    }
                }
            }
        }
        float * dst = out + (size_t) split * TN * nrows;
#pragma unroll
        for (int t = 0; t < RT; t++)
#pragma unroll
            for (int m = 0; m < TM; m++) {
                simd<uint32_t, TN> off(0, (uint32_t) (nrows * sizeof(float)));
                simd_mask<TN> mask = simd<uint32_t, TN>(0, 1) < (uint32_t) ntok;
                scatter<float, TN>(dst + row0 + t * TM + m, off, acc[t].template select<TN, 1>(m * TN).read(), mask);
            }
    });
    if (nsplit == 1) return e;
    return q.parallel_for(sycl::range<1>((size_t) ntok * nrows), e, [=](sycl::id<1> i) {
        float s = 0.f;
        for (int k = 0; k < nsplit; k++) s += part[(size_t) k * TN * nrows + i[0]];
        y[i[0]] = s;
    });
}

int main(int argc, char ** argv) {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    struct Shape { int K, rows; };
    const Shape shapes[] = {{2048, 12352}, {2048, 12288}, {2048, 9216}, {4096, 2048}, {2048, 4096}, {2048, 1024}, {512, 2048}};
    const int ntoks[] = {16, 8, 1};
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (const Shape & s : shapes) {
        const int K = s.K, R = s.rows, nb = K / QK;
        std::vector<int8_t> hwq((size_t) R * K);
        std::vector<sycl::half> hwd_t((size_t) nb * R);
        for (auto & v : hwq) v = (int8_t) (std::lround(nd(rng) * 40) % 127);
        for (auto & v : hwd_t) v = (sycl::half) (0.001f + 0.0005f * std::fabs(nd(rng)));
        std::vector<float> hx((size_t) TN * K);
        for (auto & v : hx) v = nd(rng);
        int8_t *     wq = sycl::malloc_device<int8_t>(hwq.size(), q);
        sycl::half * wd = sycl::malloc_device<sycl::half>(hwd_t.size(), q);
        float *      x  = sycl::malloc_device<float>(hx.size(), q);
        int8_t *     xq = sycl::malloc_device<int8_t>((size_t) nb * QK * TN, q);
        float *      xd = sycl::malloc_device<float>((size_t) nb * TN, q);
        float *      y  = sycl::malloc_device<float>((size_t) TN * R, q);
        q.memcpy(wq, hwq.data(), hwq.size()).wait();
        q.memcpy(wd, hwd_t.data(), hwd_t.size() * 2).wait();
        q.memcpy(x, hx.data(), hx.size() * 4).wait();
        for (int ntok : ntoks) {
            quantize_x(q, x, ntok, K, xq, xd);
            q.memset(y, 0, (size_t) TN * R * 4).wait();
            float * part = sycl::malloc_device<float>((size_t) 16 * TN * R, q);
            struct Cfg { int rt, kb, ns; };
            std::vector<Cfg> cfgs;
            for (int rt : {1, 2, 4}) for (int kb : {1, 2, 4}) for (int ns : {1, 2, 4, 8, 16})
                if (R % (TM * rt) == 0 && nb % (ns * kb) == 0) cfgs.push_back({rt, kb, ns});
            auto launch = [&](const Cfg & c) -> sycl::event {
#define L(RT, KB) if (c.rt == RT && c.kb == KB) return matmul_xmx<RT, KB>(q, wq, wd, xq, xd, y, part, R, K, ntok, c.ns);
                L(1,1) L(1,2) L(1,4) L(2,1) L(2,2) L(2,4) L(4,1) L(4,2) L(4,4)
#undef L
                return {};
            };
            Cfg best{}; double best_us = 1e30;
            for (const Cfg & c : cfgs) {
                launch(c).wait();
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 20; i++) launch(c);
                q.wait();
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 20;
                if (us < best_us) { best_us = us; best = c; }
            }
            auto run = [&]() { return launch(best); };
            run().wait();
            // CPU reference with f32 activations (so the error includes the activation quantization)
            std::vector<float> hy((size_t) TN * R);
            q.memcpy(hy.data(), y, hy.size() * 4).wait();
            double err = 0, ref2 = 0;
            for (int n = 0; n < ntok; n++)
                for (int r = 0; r < R; r += 37) {
                    double acc = 0;
                    for (int k = 0; k < K; k++) acc += (double) hwq[(size_t) r * K + k] * (float) hwd_t[(size_t) (k / QK) * R + r] * hx[(size_t) n * K + k];
                    err += (hy[(size_t) n * R + r] - acc) * (hy[(size_t) n * R + r] - acc);
                    ref2 += acc * acc;
                }
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 50; i++) run();
            q.wait();
            const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
            const int iters = 50;
            const double us = ns / iters / 1e3, bytes = (double) R * K * (1 + 2.0 / QK);
            std::printf("K=%5d rows=%6d ntok=%2d  %8.1f us  %6.1f GB/s weights  rel_rms %.2e  (rt %d kb %d split %d)\n", K, R, ntok, us,
                        bytes / us / 1e3, std::sqrt(err / ref2), best.rt, best.kb, best.ns);
            sycl::free(part, q);
        }
        for (void * p : {(void *) wq, (void *) wd, (void *) x, (void *) xq, (void *) xd, (void *) y}) sycl::free(p, q);
    }
    return 0;
}
