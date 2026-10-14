// mm_bench: time ggml mul_mat / mul_mat_id on the Vulkan backend for Drama 3 decode shapes.
// usage: mm_bench [reps]   (prints us per op for each shape and column count)

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <ggml.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static ggml_backend_t vk() {
    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { std::fprintf(stderr, "no GPU device\n"); std::exit(1); }
    std::fprintf(stderr, "device: %s\n", ggml_backend_dev_description(dev));
    return ggml_backend_dev_init(dev, nullptr);
}

static void fill(ggml_tensor * t, std::mt19937 & rng) {
    const int64_t n = ggml_nelements(t);
    std::normal_distribution<float> d(0.0F, 0.05F);
    std::vector<float> f(static_cast<size_t>(n));
    for (auto & x : f) x = d(rng);
    if (t->type == GGML_TYPE_F32) { ggml_backend_tensor_set(t, f.data(), 0, n * 4); return; }
    std::vector<uint8_t> q(ggml_nbytes(t));
    const int64_t row = t->ne[0];
    ggml_quantize_chunk(t->type, f.data(), q.data(), 0, n / row, row, nullptr);
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

struct Case {
    std::string name;
    ggml_type type;
    int64_t m, k, experts;  // experts > 0: mul_mat_id with 8 selected per token
};

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
    const int copies = 16;  // independent ops per graph, distinct weights so nothing stays in cache
    auto * backend = vk();
    std::mt19937 rng(1);
    std::vector<Case> cases = {
        {"q8 512x2048", GGML_TYPE_Q8_0, 512, 2048, 0},
        {"q8 8192x2048", GGML_TYPE_Q8_0, 8192, 2048, 0},
        {"q8 2048x4096", GGML_TYPE_Q8_0, 2048, 4096, 0},
        {"q8 32x2048", GGML_TYPE_Q8_0, 32, 2048, 0},
        {"f16 6144x2048", GGML_TYPE_F16, 6144, 2048, 0},
        {"f16 2048x6144", GGML_TYPE_F16, 2048, 6144, 0},
        {"q4k moe 512x2048", GGML_TYPE_Q4_K, 512, 2048, 256},
        {"q4k moe 2048x512", GGML_TYPE_Q4_K, 2048, 512, 256},
    };
    const char * only = std::getenv("MMB_ONLY");
    for (const auto & c : cases) {
        if (only && c.name.find(only) == std::string::npos) continue;
        for (int n : {1, 8, 16, 32, 48, 64}) {
            ggml_init_params p{256 * ggml_tensor_overhead() + ggml_graph_overhead_custom(4096, false), nullptr, true};
            auto * ctx = ggml_init(p);
            const int ncopy = c.experts ? 6 : copies;
            std::vector<ggml_tensor *> w(ncopy);
            for (auto & t : w) t = c.experts ? ggml_new_tensor_3d(ctx, c.type, c.k, c.m, c.experts)
                                              : ggml_new_tensor_2d(ctx, c.type, c.k, c.m);
            static const bool colbatch = std::getenv("MMB_COLBATCH") != nullptr;  // x as [K, 1, n]: batched mat-vec
            auto * x = (c.experts || colbatch) ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.k, 1, n)
                                               : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.k, n);
            ggml_tensor * ids = c.experts ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, n) : nullptr;
            auto * buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            for (auto * t : w) fill(t, rng);
            fill(x, rng);
            if (ids) {
                std::vector<int32_t> v(8 * n);
                for (int j = 0; j < n; ++j) {
                    std::vector<int> pool(c.experts);
                    for (int e = 0; e < c.experts; ++e) pool[e] = e;
                    std::shuffle(pool.begin(), pool.end(), rng);
                    for (int s = 0; s < 8; ++s) v[j * 8 + s] = pool[s];
                }
                ggml_backend_tensor_set(ids, v.data(), 0, v.size() * 4);
            }
            ggml_init_params gp{1024 * ggml_tensor_overhead() + ggml_graph_overhead_custom(4096, false), nullptr, true};
            auto * gctx = ggml_init(gp);
            auto * g = ggml_new_graph_custom(gctx, 4096, false);
            for (auto * t : w) {
                auto * y = c.experts ? ggml_mul_mat_id(gctx, t, x, ids) : ggml_mul_mat(gctx, t, x);
                ggml_build_forward_expand(g, y);
            }
            auto * ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            ggml_gallocr_alloc_graph(ga, g);
            ggml_backend_graph_compute(backend, g);
            if (!c.experts && std::getenv("MMB_CHECK")) {
                // host reference for the first copy
                const int64_t K = c.k, M = c.m;
                std::vector<uint8_t> wq(ggml_nbytes(w[0]));
                ggml_backend_tensor_get(w[0], wq.data(), 0, wq.size());
                std::vector<float> wf(K * M), xf(K * n), yf(M * n);
                if (c.type == GGML_TYPE_F32) std::memcpy(wf.data(), wq.data(), wq.size());
                else ggml_get_type_traits(c.type)->to_float(wq.data(), wf.data(), K * M);
                ggml_backend_tensor_get(x, xf.data(), 0, xf.size() * 4);
                ggml_backend_tensor_get(ggml_graph_node(g, 0), yf.data(), 0, yf.size() * 4);
                double maxerr = 0, maxref = 0;
                for (int j = 0; j < n; ++j) for (int64_t i = 0; i < M; ++i) {
                    double ref = 0;
                    for (int64_t k = 0; k < K; ++k) ref += (double) wf[i * K + k] * xf[j * K + k];
                    maxerr = std::max(maxerr, std::fabs(ref - yf[j * M + i])); maxref = std::max(maxref, std::fabs(ref));
                }
                std::printf("  check n=%d: max abs err %.3g (max |ref| %.3g)\n", n, maxerr, maxref);
            }
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) ggml_backend_graph_compute(backend, g);
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps / ncopy;
            const double bytes = static_cast<double>(ggml_nbytes(w[0])) /
                (c.experts ? static_cast<double>(c.experts) / std::min<double>(c.experts, c.experts * (1 - std::pow(1 - 8.0 / c.experts, n))) : 1.0);
            std::printf("%-18s n=%3d  %8.1f us   %6.0f GB/s (weights touched)\n", c.name.c_str(), n, us, bytes / us / 1e3);
            std::fflush(stdout);
            ggml_gallocr_free(ga);
            ggml_free(gctx);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
    }
    ggml_backend_free(backend);
    return 0;
}
