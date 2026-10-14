#include "engine/community_models/fish_drama3/batch_engine.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/models/fish_audio/tokenizer_text.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace engine::models::fish_drama3 {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct CtxDeleter {
    void operator()(ggml_context * ctx) const { if (ctx) ggml_free(ctx); }
};
using CtxPtr = std::unique_ptr<ggml_context, CtxDeleter>;

constexpr int kRanks = 32;          // sampler candidate list (top_k <= kRanks)
constexpr int kSamplers = 11;       // semantic, semantic RAS, nine residual codebooks
constexpr int kRasWindow = 16;
constexpr float kDropped = -1.0e9F;

// Small f32/i32 tensors created on the host: norm weights with the Gemma +1
// folded in, gate constants, sampler tables.
class HostTensors {
public:
    explicit HostTensors(ggml_backend_t backend) : backend_(backend) {
        ggml_init_params params{4096 * ggml_tensor_overhead(), nullptr, true};
        ctx_.reset(ggml_init(params));
    }
    ~HostTensors() { if (buffer_) ggml_backend_buffer_free(buffer_); }
    ggml_tensor * f32(std::vector<float> values, std::initializer_list<int64_t> ne) {
        auto * t = make(GGML_TYPE_F32, ne);
        f32_.emplace_back(t, std::move(values));
        return t;
    }
    ggml_tensor * i32(std::vector<int32_t> values, std::initializer_list<int64_t> ne) {
        auto * t = make(GGML_TYPE_I32, ne);
        i32_.emplace_back(t, std::move(values));
        return t;
    }
    ggml_tensor * raw(ggml_type type, std::initializer_list<int64_t> ne, std::vector<std::byte> bytes) {
        auto * t = make(type, ne);
        raw_.emplace_back(t, std::move(bytes));
        return t;
    }
    void upload() {
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_);
        if (!buffer_) throw std::runtime_error("Drama 3 batch: constant buffer allocation failed");
        for (auto & [t, v] : f32_) ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
        for (auto & [t, v] : i32_) ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(int32_t));
        for (auto & [t, v] : raw_) {
            if (v.size() != ggml_nbytes(t)) throw std::runtime_error("Drama 3 batch: raw constant size mismatch");
            ggml_backend_tensor_set(t, v.data(), 0, v.size());
        }
        raw_.clear();
        f32_.clear();
        i32_.clear();
    }
private:
    ggml_tensor * make(ggml_type type, std::initializer_list<int64_t> ne) {
        std::vector<int64_t> dims(ne);
        auto * t = ggml_new_tensor(ctx_.get(), type, static_cast<int>(dims.size()), dims.data());
        int64_t count = 1;
        for (auto d : dims) count *= d;
        (void) count;
        return t;
    }
    ggml_backend_t backend_;
    CtxPtr ctx_;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::vector<std::pair<ggml_tensor *, std::vector<float>>> f32_;
    std::vector<std::pair<ggml_tensor *, std::vector<int32_t>>> i32_;
    std::vector<std::pair<ggml_tensor *, std::vector<std::byte>>> raw_;
};

struct BackboneLayer {
    bool full = false;
    ggml_tensor * in_norm = nullptr;
    ggml_tensor * post_norm = nullptr;
    // full attention
    ggml_tensor * wq = nullptr;
    ggml_tensor * wk = nullptr;
    ggml_tensor * wv = nullptr;
    ggml_tensor * wo = nullptr;
    ggml_tensor * q_norm = nullptr;
    ggml_tensor * k_norm = nullptr;
    // gated delta net
    ggml_tensor * w_qkv = nullptr;
    ggml_tensor * w_z = nullptr;
    ggml_tensor * w_b = nullptr;
    ggml_tensor * w_a = nullptr;
    ggml_tensor * conv = nullptr;
    ggml_tensor * dt_bias = nullptr;
    ggml_tensor * neg_a = nullptr;
    ggml_tensor * lin_norm = nullptr;
    ggml_tensor * w_out = nullptr;
    // fused projections (rows concatenated, same input): nullptr when the parts differ in type
    ggml_tensor * dn_in = nullptr;      // [in_proj_qkv | in_proj_z | in_proj_b | in_proj_a]
    ggml_tensor * att_in = nullptr;     // [q_proj | k_proj | v_proj]
    ggml_tensor * route = nullptr;      // [router | shared_expert_gate]
    ggml_tensor * sh_gate_up = nullptr; // [shared gate | shared up]
    ggml_tensor * exp_gate_up = nullptr;// per expert [gate | up]
    // mixture of experts
    ggml_tensor * router = nullptr;
    ggml_tensor * exp_gate = nullptr;
    ggml_tensor * exp_up = nullptr;
    ggml_tensor * exp_down = nullptr;
    ggml_tensor * sh_gate = nullptr;
    ggml_tensor * sh_up = nullptr;
    ggml_tensor * sh_down = nullptr;
    ggml_tensor * sh_scale = nullptr;
    // per-slot state
    ggml_tensor * key_cache = nullptr;    // f16 [kv_width, capacity, slots]
    ggml_tensor * value_cache = nullptr;
    ggml_tensor * conv_tail = nullptr;    // f32 [kernel - 1, conv_dim, slots]
    ggml_tensor * state = nullptr;        // f32 [S_v * S_v * H, slots]
};

struct TowerLayer {
    ggml_tensor * embedding = nullptr;
    ggml_tensor * enorm = nullptr;
    ggml_tensor * hnorm = nullptr;
    ggml_tensor * eh = nullptr;
    ggml_tensor * in_norm = nullptr;
    ggml_tensor * qkv = nullptr;
    ggml_tensor * q_norm = nullptr;
    ggml_tensor * k_norm = nullptr;
    ggml_tensor * wo = nullptr;
    ggml_tensor * post_norm = nullptr;
    ggml_tensor * w1 = nullptr;
    ggml_tensor * w3 = nullptr;
    ggml_tensor * w2 = nullptr;
    ggml_tensor * final_norm = nullptr;
    ggml_tensor * head = nullptr;
    ggml_tensor * w13 = nullptr;        // fused [w1 | w3]
    ggml_tensor * key_cache = nullptr;
    ggml_tensor * value_cache = nullptr;
};

// A step graph over S sequences of T tokens each, starting at slot `base`.
struct StepGraph {
    CtxPtr ctx;
    ggml_cgraph * graph = nullptr;
    int64_t tokens = 0, seqs = 0, base = 0, keys = 0;
    bool decode = false;
    bool packed = false;               // decode attention with GQA heads packed as query rows
    int64_t mask_rows = 1;
    ggml_tensor * tok = nullptr;       // I32 [N]
    ggml_tensor * add = nullptr;       // f32 [hidden, N]
    ggml_tensor * scale = nullptr;     // f32 [1, N]
    ggml_tensor * extra = nullptr;     // f32 [hidden, N]
    ggml_tensor * label = nullptr;     // f32 [1, N]
    ggml_tensor * pos = nullptr;       // I32 [N]
    ggml_tensor * kv_index = nullptr;  // I32 [N]
    ggml_tensor * mask = nullptr;      // f16 [keys, T, 1, S]
    ggml_tensor * codes = nullptr;     // I32 [N, 10] (prefill: next-frame codes)
    ggml_tensor * noise = nullptr;     // f32 [kRanks, N, kSamplers]
    ggml_tensor * recent = nullptr;    // f32 [V, N]
    ggml_tensor * out = nullptr;       // I32 [10, N] (decode) or tower hidden (prefill)
};

struct Row {
    bool active = false;
    Drama3BatchJob job;
    std::vector<Drama3Frame> suffix;   // forced text tokens; the last one starts generation
    size_t forced = 0;                 // index of the current input token in suffix
    int64_t position = 0;
    int64_t prompt_tokens = 0;
    int64_t limit = 0;
    Drama3Frame input{};
    std::vector<Drama3Frame> output;
    std::deque<int32_t> recent;
    std::mt19937 random;
    bool generating() const { return forced + 1 >= suffix.size(); }
};

bool is_semantic(const Drama3Frame & frame, const Drama3TowerConfig & c) {
    return frame[0] >= c.semantic_begin && frame[0] < c.semantic_begin + c.codebook_size;
}

}  // namespace

class Drama3BatchEngine::Impl {
public:
    Impl(std::shared_ptr<const Drama3Assets> assets, core::ExecutionContext & execution, Drama3BatchOptions options)
        : assets_(std::move(assets)), execution_(execution), backend_(execution.backend()), opt_(options),
          tokenizer_(assets_->tokenizer_assets), constants_(execution.backend()) {
        if (opt_.batch <= 0 || opt_.capacity <= 0 || opt_.voice_slots <= 0 || opt_.top_k <= 0 ||
            opt_.top_k > kRanks || opt_.kv_bucket <= 0) {
            throw std::runtime_error("invalid Drama 3 batch options");
        }
        slots_ = opt_.batch + opt_.voice_slots;
        load_weights();
        allocate_state();
        load_codec();
        assets_->model_weights->release_storage();
        assets_->codec_weights->release_storage();
        galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        rows_.resize(static_cast<size_t>(opt_.batch));
        voice_lru_.clear();
        for (int s = 0; s < opt_.voice_slots; ++s) voice_slot_key_.push_back("");
    }

    ~Impl() {
        graphs_.clear();
        prefill_.reset();
        copy_graph_.reset();
        if (galloc_) ggml_gallocr_free(galloc_);
        if (copy_galloc_) ggml_gallocr_free(copy_galloc_);
        if (state_buffer_) ggml_backend_buffer_free(state_buffer_);
        for (auto & o : owned_) if (o.buffer) ggml_backend_buffer_free(o.buffer);
    }

    codecs::FishDacCodes encode_reference(const runtime::AudioBuffer & audio) {
        auto result = codec_->encode_codes(audio);
        codec_->release_encode_graph();
        validate_drama3_codes(result, assets_->tower.codebook_size);
        return result;
    }

    runtime::AudioBuffer decode(const codecs::FishDacCodes & codes) {
        auto audio = codec_->decode_codes(codes);
        return audio;
    }

    void run(const std::vector<Drama3BatchJob> & jobs, const std::function<void(Drama3BatchResult &&)> & done) {
        std::deque<const Drama3BatchJob *> queue;
        for (const auto & job : jobs) queue.push_back(&job);
        int active = 0;
        while (!queue.empty() || active > 0) {
            // Fill free rows.
            for (int r = 0; r < opt_.batch && !queue.empty(); ++r) {
                if (rows_[r].active) continue;
                admit(r, *queue.front());
                queue.pop_front();
                ++active;
            }
            step();
            for (int r = 0; r < opt_.batch; ++r) {
                auto & row = rows_[r];
                if (!row.active || !row_finished_[r]) continue;
                Drama3BatchResult result;
                result.id = row.job.id;
                result.reached_limit = row_limited_[r];
                result.prompt_tokens = row.prompt_tokens;
                result.codes.codebooks = 10;
                result.codes.frames = static_cast<int64_t>(row.output.size());
                result.codes.codes.resize(row.output.size() * 10);
                for (int64_t book = 0; book < 10; ++book) {
                    for (size_t f = 0; f < row.output.size(); ++f) {
                        result.codes.codes[book * row.output.size() + f] = row.output[f][book + 1];
                    }
                }
                row.active = false;
                row.output.clear();
                --active;
                done(std::move(result));
            }
        }
    }

    Drama3BatchStats stats_;

private:
    // ------------------------------------------------------------------ loading
    struct OwnedTensor {
        CtxPtr ctx;
        ggml_backend_buffer_t buffer = nullptr;
    };
    // Loads tensors that share an input as one weight with their rows concatenated, so one matmul replaces
    // several. With experts > 0 the parts are [experts, rows, inner] banks, concatenated per expert.
    // Returns nullptr (caller loads the parts separately) when the parts differ in type or shape.
    ggml_tensor * load_fused(const char * kind, const std::vector<std::string> & names, int64_t inner, int64_t experts = 0) {
        if (getenv("D3_NO_FUSE")) return nullptr;
        if (const char * only = getenv("D3_FUSE")) {
            if (std::string(",") .append(only).append(",").find(std::string(",") + kind + ",") == std::string::npos) return nullptr;
        }
        const auto & source = *assets_->model_weights;
        ggml_type type = GGML_TYPE_COUNT;
        for (const auto & n : names) {
            const auto t = assets::ggml_type_for_tensor_dtype(source.require_metadata(n).dtype);
            if (type == GGML_TYPE_COUNT) type = t;
            else if (t != type) return nullptr;
        }
        const int64_t E = experts > 0 ? experts : 1;
        const size_t row = ggml_row_size(type, inner);
        std::vector<int64_t> rows;
        int64_t total = 0;
        for (const auto & n : names) {
            const auto meta = source.require_metadata(n);
            if (meta.shape.empty() || meta.shape.back() != inner) return nullptr;
            int64_t count = 1;
            for (size_t d = 0; d + 1 < meta.shape.size(); ++d) count *= meta.shape[d];
            if (count % E) return nullptr;
            rows.push_back(count / E);
            total += count / E;
        }
        OwnedTensor owned;
        ggml_init_params params{2 * ggml_tensor_overhead(), nullptr, true};
        owned.ctx.reset(ggml_init(params));
        auto * t = experts > 0 ? ggml_new_tensor_3d(owned.ctx.get(), type, inner, total, E)
                               : ggml_new_tensor_2d(owned.ctx.get(), type, inner, total);
        owned.buffer = ggml_backend_alloc_ctx_tensors(owned.ctx.get(), backend_);
        if (!owned.buffer) throw std::runtime_error("Drama 3 batch: fused weight allocation failed");
        std::vector<std::byte> out(row * static_cast<size_t>(total * E));
        for (size_t p = 0; p < names.size(); ++p) {
            const auto raw = source.require_tensor_data(names[p]);
            const size_t part = row * static_cast<size_t>(rows[p]);
            if (raw.bytes.size() != part * static_cast<size_t>(E)) throw std::runtime_error("Drama 3 fused weight size mismatch: " + names[p]);
            size_t offset = 0;
            for (size_t q = 0; q < p; ++q) offset += row * static_cast<size_t>(rows[q]);
            for (int64_t e = 0; e < E; ++e) {
                std::memcpy(out.data() + e * row * total + offset, raw.bytes.data() + e * part, part);
            }
        }
        ggml_backend_tensor_set(t, out.data(), 0, out.size());
        owned_.push_back(std::move(owned));
        fused_count_++;
        return t;
    }

    void load_weights() {
        const auto & source = *assets_->model_weights;
        const auto & b = assets_->backbone;
        const auto & t = assets_->tower;
        store_ = std::make_unique<core::BackendWeightStore>(backend_, execution_.backend_type(),
                                                            "drama3_batch.weights", 16 * 1024 * 1024);
        auto load = [&](const std::string & name, std::initializer_list<int64_t> shape) {
            return store_->load_tensor(source, name, assets::TensorStorageType::Native, shape).tensor;
        };
        auto norm = [&](const std::string & name, int64_t n, bool gemma) {
            auto v = source.require_f32(name, {n});
            if (gemma) for (auto & x : v) x += 1.0F;
            return constants_.f32(std::move(v), {n});
        };
        const std::string root = b.weight_prefix;
        token_embedding_ = load(root + ".embed_tokens.weight", {b.vocab_size, b.hidden_size});
        const int64_t key_dim = b.linear_key_head_dim * b.linear_num_key_heads;
        const int64_t value_dim = b.linear_value_head_dim * b.linear_num_value_heads;
        const int64_t conv_dim = 2 * key_dim + value_dim;
        for (int64_t l = 0; l < b.layers; ++l) {
            const std::string p = root + ".layers." + std::to_string(l);
            BackboneLayer L;
            L.full = (l + 1) % b.full_attention_interval == 0;
            L.in_norm = norm(p + ".input_layernorm.weight", b.hidden_size, true);
            L.post_norm = norm(p + ".post_attention_layernorm.weight", b.hidden_size, true);
            if (L.full) {
                const std::string a = p + ".self_attn";
                L.att_in = load_fused("att", {a + ".q_proj.weight", a + ".k_proj.weight", a + ".v_proj.weight"}, b.hidden_size);
                if (!L.att_in) {
                    L.wq = load(a + ".q_proj.weight", {b.heads * b.head_dim * 2, b.hidden_size});
                    L.wk = load(a + ".k_proj.weight", {b.kv_heads * b.head_dim, b.hidden_size});
                    L.wv = load(a + ".v_proj.weight", {b.kv_heads * b.head_dim, b.hidden_size});
                }
                L.wo = load(a + ".o_proj.weight", {b.hidden_size, b.heads * b.head_dim});
                L.q_norm = norm(a + ".q_norm.weight", b.head_dim, true);
                L.k_norm = norm(a + ".k_norm.weight", b.head_dim, true);
            } else {
                const std::string a = p + ".linear_attn";
                L.dn_in = load_fused("dn", {a + ".in_proj_qkv.weight", a + ".in_proj_z.weight", a + ".in_proj_b.weight",
                                      a + ".in_proj_a.weight"}, b.hidden_size);
                if (!L.dn_in) {
                    L.w_qkv = load(a + ".in_proj_qkv.weight", {conv_dim, b.hidden_size});
                    L.w_z = load(a + ".in_proj_z.weight", {value_dim, b.hidden_size});
                    L.w_b = load(a + ".in_proj_b.weight", {b.linear_num_value_heads, b.hidden_size});
                    L.w_a = load(a + ".in_proj_a.weight", {b.linear_num_value_heads, b.hidden_size});
                }
                auto conv = source.require_f32(a + ".conv1d.weight", {conv_dim, 1, b.linear_conv_kernel_dim});
                L.conv = constants_.f32(std::move(conv), {b.linear_conv_kernel_dim, conv_dim});
                L.dt_bias = constants_.f32(source.require_f32(a + ".dt_bias", {b.linear_num_value_heads}),
                                           {b.linear_num_value_heads});
                auto a_log = source.require_f32(a + ".A_log", {b.linear_num_value_heads});
                for (auto & x : a_log) x = -std::exp(x);
                L.neg_a = constants_.f32(std::move(a_log), {b.linear_num_value_heads});
                L.lin_norm = norm(a + ".norm.weight", b.linear_value_head_dim, false);
                L.w_out = load(a + ".out_proj.weight", {b.hidden_size, value_dim});
            }
            const std::string m = p + ".mlp";
            L.sh_gate_up = load_fused("sh", {m + ".shared_expert.gate_proj.weight", m + ".shared_expert.up_proj.weight"}, b.hidden_size);
            if (!L.sh_gate_up) {
                L.sh_gate = load(m + ".shared_expert.gate_proj.weight", {b.intermediate_size, b.hidden_size});
                L.sh_up = load(m + ".shared_expert.up_proj.weight", {b.intermediate_size, b.hidden_size});
            }
            L.sh_down = load(m + ".shared_expert.down_proj.weight", {b.hidden_size, b.intermediate_size});
            L.route = load_fused("route", {m + ".gate.weight", m + ".shared_expert_gate.weight"}, b.hidden_size);
            if (!L.route) {
                L.router = load(m + ".gate.weight", {b.experts, b.hidden_size});
                L.sh_scale = load(m + ".shared_expert_gate.weight", {1, b.hidden_size});
            }
            L.exp_gate_up = load_fused("exp", {m + ".experts.gate_proj.weight", m + ".experts.up_proj.weight"}, b.hidden_size, b.experts);
            if (!L.exp_gate_up) {
                L.exp_gate = load(m + ".experts.gate_proj.weight", {b.experts, b.intermediate_size, b.hidden_size});
                L.exp_up = load(m + ".experts.up_proj.weight", {b.experts, b.intermediate_size, b.hidden_size});
            }
            L.exp_down = load(m + ".experts.down_proj.weight", {b.experts, b.hidden_size, b.intermediate_size});
            layers_.push_back(L);
        }
        final_norm_ = norm(root + ".norm.weight", b.hidden_size, true);
        {
            // Only the 4096 semantic rows and the EOS row (last) of the LM head are ever sampled.
            const auto raw = source.require_tensor_data(b.tie_word_embeddings ? root + ".embed_tokens.weight"
                                                                              : b.lm_head_prefix + ".weight");
            const ggml_type type = assets::ggml_type_for_tensor_dtype(raw.metadata.dtype);
            const size_t row = ggml_row_size(type, b.hidden_size);
            if (raw.bytes.size() != row * static_cast<size_t>(b.vocab_size)) throw std::runtime_error("Drama 3 LM head size mismatch");
            std::vector<std::byte> rows(row * static_cast<size_t>(t.codebook_size + 1));
            std::memcpy(rows.data(), raw.bytes.data() + row * t.semantic_begin, row * t.codebook_size);
            std::memcpy(rows.data() + row * t.codebook_size, raw.bytes.data() + row * assets_->eos, row);
            lm_head_ = constants_.raw(type, {b.hidden_size, t.codebook_size + 1}, std::move(rows));
        }

        for (int64_t index = 0; index < t.shared_layers + t.residual_layers; ++index) {
            const bool shared = index < t.shared_layers;
            const int64_t local = shared ? index : index - t.shared_layers;
            const std::string p = std::string(shared ? "audio_decoder_mtp_shared.layers." : "audio_decoder_mtp.layers.") +
                std::to_string(local);
            TowerLayer w;
            w.embedding = load(p + (shared ? ".semantic_embedding.weight" : ".codebook_embedding.weight"),
                               {t.codebook_size, t.hidden_size});
            w.enorm = norm(p + ".enorm.weight", t.hidden_size, true);
            w.hnorm = norm(p + ".hnorm.weight", t.hidden_size, true);
            w.eh = load(p + ".eh_proj.weight", {t.hidden_size, 2 * t.hidden_size});
            w.final_norm = norm(p + ".final_layernorm.weight", t.hidden_size, true);
            const std::string x = p + ".transformer_layer";
            w.in_norm = norm(x + ".attention_norm.weight", t.hidden_size, true);
            w.post_norm = norm(x + ".ffn_norm.weight", t.hidden_size, true);
            const int64_t qw = t.heads * t.head_dim, kw = t.kv_heads * t.head_dim;
            w.qkv = load(x + ".attention.wqkv.weight", {2 * qw + 2 * kw, t.hidden_size});
            w.q_norm = norm(x + ".attention.q_norm.weight", t.head_dim, true);
            w.k_norm = norm(x + ".attention.k_norm.weight", t.head_dim, true);
            w.wo = load(x + ".attention.wo.weight", {t.hidden_size, qw});
            w.w13 = load_fused("tower", {x + ".feed_forward.w1.weight", x + ".feed_forward.w3.weight"}, t.hidden_size);
            if (!w.w13) {
                w.w1 = load(x + ".feed_forward.w1.weight", {t.intermediate_size, t.hidden_size});
                w.w3 = load(x + ".feed_forward.w3.weight", {t.intermediate_size, t.hidden_size});
            }
            w.w2 = load(x + ".feed_forward.w2.weight", {t.hidden_size, t.intermediate_size});
            if (!shared) w.head = load("audio_decoder_mtp.heads." + std::to_string(local) + ".weight",
                                       {t.codebook_size, t.hidden_size});
            tower_.push_back(w);
        }
        code_embeddings_ = source.require_f32("audio_decoder.codebook_embeddings.weight",
                                              {10 * t.codebook_size, t.hidden_size});
        store_->upload();
        if (opt_.verbose) std::cerr << "[drama3_batch] fused weights: " << fused_count_ << "\n";

        // Sampler constants.
        vocab_ = t.codebook_size + 1;  // semantic codes, then EOS
        std::vector<int32_t> lut(static_cast<size_t>(vocab_));
        for (int64_t i = 0; i < vocab_; ++i) lut[i] = i < t.codebook_size ? static_cast<int32_t>(i) : 0;
        semantic_lut_ = constants_.i32(std::move(lut), {1, vocab_});
        std::vector<float> not_first(kRanks, 1.0F);
        not_first[0] = 0.0F;
        not_first_ = constants_.f32(std::move(not_first), {kRanks, 1});
        std::vector<float> drop_main(kRanks, 0.0F), drop_ras(kRanks, 0.0F);
        for (int r = 0; r < kRanks; ++r) {
            drop_main[r] = r >= opt_.top_k ? 1.0F : 0.0F;
            drop_ras[r] = r >= 30 ? 1.0F : 0.0F;
        }
        drop_main_ = constants_.f32(std::move(drop_main), {kRanks, 1});
        drop_ras_ = constants_.f32(std::move(drop_ras), {kRanks, 1});
        constants_.upload();
    }

    void allocate_state() {
        const auto & b = assets_->backbone;
        const auto & t = assets_->tower;
        ggml_init_params params{1024 * ggml_tensor_overhead(), nullptr, true};
        state_ctx_.reset(ggml_init(params));
        auto * ctx = state_ctx_.get();
        const int64_t key_dim = b.linear_key_head_dim * b.linear_num_key_heads;
        const int64_t value_dim = b.linear_value_head_dim * b.linear_num_value_heads;
        const int64_t conv_dim = 2 * key_dim + value_dim;
        const int64_t state_values = b.linear_value_head_dim * b.linear_value_head_dim * b.linear_num_value_heads;
        for (auto & L : layers_) {
            if (L.full) {
                L.key_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, b.kv_heads * b.head_dim, opt_.capacity, slots_);
                L.value_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, b.kv_heads * b.head_dim, opt_.capacity, slots_);
            } else {
                L.conv_tail = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, b.linear_conv_kernel_dim - 1, conv_dim, slots_);
                L.state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, state_values, slots_);
            }
        }
        for (auto & w : tower_) {
            w.key_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, t.kv_heads * t.head_dim, opt_.capacity, slots_);
            w.value_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, t.kv_heads * t.head_dim, opt_.capacity, slots_);
        }
        state_buffer_ = ggml_backend_alloc_ctx_tensors(ctx, backend_);
        if (!state_buffer_) throw std::runtime_error("Drama 3 batch: state allocation failed (lower --batch or --capacity)");
        ggml_backend_buffer_clear(state_buffer_, 0);
        if (opt_.verbose) {
            std::cerr << "[drama3_batch] state " << ggml_backend_buffer_get_size(state_buffer_) / 1048576.0
                      << " MiB for " << slots_ << " slots\n";
        }
    }

    void load_codec() {
        auto codec_backend = execution_.config();
        codec_execution_ = std::make_unique<core::ExecutionContext>(codec_backend);
        codecs::FishDacCodecRuntimeOptions codec_options;
        codec_options.weight_context_bytes = 16 * 1024 * 1024;
        const auto component = codecs::FishDacCodecComponent::load_from_tensor_source(assets_->codec_weights,
            assets_->codec, {}, codec_execution_->backend(), codec_backend.type, codec_options);
        codec_ = std::make_unique<codecs::FishDacCodecRuntime>(component, codec_backend,
            codec_backend.threads, codec_options.graph_arena_bytes);
        if (opt_.codec_cpu_workers > 0) {
            core::BackendConfig cpu;
            cpu.type = core::BackendType::Cpu;
            cpu.threads = opt_.codec_cpu_threads;
            cpu_codec_execution_ = std::make_unique<core::ExecutionContext>(cpu);
            const auto cpu_component = codecs::FishDacCodecComponent::load_from_tensor_source(assets_->codec_weights,
                assets_->codec, {}, cpu_codec_execution_->backend(), core::BackendType::Cpu, codec_options);
            for (int w = 0; w < opt_.codec_cpu_workers; ++w) {
                cpu_codecs_.push_back(std::make_unique<codecs::FishDacCodecRuntime>(cpu_component, cpu, cpu.threads,
                                                                                    codec_options.graph_arena_bytes));
            }
        }
    }
public:
    runtime::AudioBuffer decode_cpu(int worker, const codecs::FishDacCodes & codes) {
        auto audio = cpu_codecs_.at(static_cast<size_t>(worker))->decode_codes(codes);
        return audio;
    }
private:

    // ------------------------------------------------------------------ graph
    ggml_tensor * norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
        return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
    }

    // FA over per-slot caches. q: [D, H, N] (N = T*S rows, sequence-major).
    ggml_tensor * attention(ggml_context * ctx, StepGraph & g, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                            ggml_tensor * key_cache, ggml_tensor * value_cache, int64_t head_dim, int64_t heads,
                            int64_t kv_heads) {
        const int64_t T = g.tokens, S = g.seqs, N = T * S;
        const int64_t width = kv_heads * head_dim;
        auto * kc = ggml_reshape_2d(ctx, key_cache, width, opt_.capacity * slots_);
        auto * vc = ggml_reshape_2d(ctx, value_cache, width, opt_.capacity * slots_);
        kc = ggml_set_rows(ctx, kc, ggml_reshape_2d(ctx, k, width, N), g.kv_index);
        vc = ggml_set_rows(ctx, vc, ggml_reshape_2d(ctx, v, width, N), g.kv_index);
        const size_t row = static_cast<size_t>(width) * ggml_type_size(GGML_TYPE_F16);
        const size_t slot = row * opt_.capacity;
        auto * keys = ggml_view_4d(ctx, kc, head_dim, g.keys, kv_heads, S, row, head_dim * ggml_type_size(GGML_TYPE_F16),
                                   slot, g.base * slot);
        auto * values = ggml_view_4d(ctx, vc, head_dim, g.keys, kv_heads, S, row, head_dim * ggml_type_size(GGML_TYPE_F16),
                                     slot, g.base * slot);
        const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
        if (g.packed) {
            // One query per sequence: pack the heads that share a KV head as query rows of that head, so each
            // K/V tile is read once per KV head instead of once per query head.
            const int64_t group = heads / kv_heads;
            auto * qp = ggml_reshape_4d(ctx, q, head_dim, group, kv_heads, S);
            auto * out = ggml_flash_attn_ext(ctx, qp, keys, values, g.mask, scale, 0, 0);  // [D, kv_heads, group, S]
            ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
            out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));                       // [D, group, kv_heads, S]
            return ggml_reshape_2d(ctx, out, head_dim * heads, N);
        }
        q = ggml_permute(ctx, ggml_reshape_4d(ctx, q, head_dim, heads, T, S), 0, 2, 1, 3);
        auto * out = ggml_flash_attn_ext(ctx, q, keys, values, g.mask, scale, 0, 0);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        return ggml_reshape_2d(ctx, out, head_dim * heads, N);
    }

    ggml_tensor * full_attention(ggml_context * ctx, StepGraph & g, const BackboneLayer & L, ggml_tensor * x) {
        const auto & b = assets_->backbone;
        const int64_t N = g.tokens * g.seqs;
        const int64_t D = b.head_dim;
        const int64_t qw = 2 * D * b.heads, kw = D * b.kv_heads;
        ggml_tensor * qg = nullptr;
        ggml_tensor * k = nullptr;
        ggml_tensor * v = nullptr;
        if (L.att_in) {
            auto * y = ggml_mul_mat(ctx, L.att_in, x);  // [qw + 2 kw, N]
            qg = y;
            k = ggml_view_3d(ctx, y, D, b.kv_heads, N, D * sizeof(float), y->nb[1], qw * sizeof(float));
            v = ggml_cont(ctx, ggml_view_2d(ctx, y, kw, N, y->nb[1], (qw + kw) * sizeof(float)));
        } else {
            qg = ggml_mul_mat(ctx, L.wq, x);  // [2D * heads, N], per head [q | gate]
            k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wk, x), D, b.kv_heads, N);
            v = ggml_mul_mat(ctx, L.wv, x);
        }
        auto * q = ggml_view_3d(ctx, qg, D, b.heads, N, 2 * D * sizeof(float), qg->nb[1], 0);  // rms_norm takes strided input
        auto * gate = ggml_cont(ctx, ggml_view_3d(ctx, qg, D, b.heads, N, 2 * D * sizeof(float), qg->nb[1], D * sizeof(float)));
        const int rot = static_cast<int>(std::llround(D * static_cast<double>(b.partial_rotary_factor)));
        q = ggml_rope_ext(ctx, norm(ctx, q, L.q_norm, b.rms_norm_eps), g.pos, nullptr, rot, GGML_ROPE_TYPE_NEOX, 0,
                          b.rope_theta, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
        k = ggml_rope_ext(ctx, norm(ctx, k, L.k_norm, b.rms_norm_eps), g.pos, nullptr, rot, GGML_ROPE_TYPE_NEOX, 0,
                          b.rope_theta, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
        auto * att = attention(ctx, g, q, k, v, L.key_cache, L.value_cache, D, b.heads, b.kv_heads);
        att = ggml_mul(ctx, att, ggml_sigmoid(ctx, ggml_reshape_2d(ctx, gate, D * b.heads, N)));
        return ggml_mul_mat(ctx, L.wo, att);
    }

    ggml_tensor * delta_net(ggml_context * ctx, StepGraph & g, const BackboneLayer & L, ggml_tensor * x) {
        const auto & b = assets_->backbone;
        const int64_t T = g.tokens, S = g.seqs, N = T * S;
        const int64_t dk = b.linear_key_head_dim, hk = b.linear_num_key_heads;
        const int64_t dv = b.linear_value_head_dim, hv = b.linear_num_value_heads;
        const int64_t key_dim = dk * hk, value_dim = dv * hv, conv_dim = 2 * key_dim + value_dim;
        const int64_t kernel = b.linear_conv_kernel_dim;
        const int64_t state_values = dv * dv * hv;
        ggml_tensor * qkv3 = nullptr;
        ggml_tensor * z = nullptr;
        ggml_tensor * beta_in = nullptr;
        ggml_tensor * a_in = nullptr;
        if (L.dn_in) {
            auto * y = ggml_mul_mat(ctx, L.dn_in, x);  // [conv_dim + value_dim + 2 hv, N]
            qkv3 = ggml_view_3d(ctx, y, conv_dim, T, S, y->nb[1], T * y->nb[1], 0);
            z = ggml_cont(ctx, ggml_view_2d(ctx, y, value_dim, N, y->nb[1], conv_dim * sizeof(float)));
            beta_in = ggml_cont(ctx, ggml_view_2d(ctx, y, hv, N, y->nb[1], (conv_dim + value_dim) * sizeof(float)));
            a_in = ggml_cont(ctx, ggml_view_2d(ctx, y, hv, N, y->nb[1], (conv_dim + value_dim + hv) * sizeof(float)));
        } else {
            qkv3 = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.w_qkv, x), conv_dim, T, S);
            z = ggml_mul_mat(ctx, L.w_z, x);
            beta_in = ggml_mul_mat(ctx, L.w_b, x);
            a_in = ggml_mul_mat(ctx, L.w_a, x);
        }
        auto * qkv_t = ggml_cont(ctx, ggml_transpose(ctx, qkv3));  // [T, conv_dim, S]
        const size_t tail_slot = static_cast<size_t>((kernel - 1) * conv_dim) * sizeof(float);
        auto * tail = ggml_view_3d(ctx, L.conv_tail, kernel - 1, conv_dim, S, L.conv_tail->nb[1], L.conv_tail->nb[2],
                                   g.base * tail_slot);
        auto * sx = ggml_concat(ctx, tail, qkv_t, 0);  // [kernel - 1 + T, conv_dim, S]
        ggml_build_forward_expand(g.graph, ggml_cpy(ctx,
            ggml_view_3d(ctx, sx, kernel - 1, conv_dim, S, sx->nb[1], sx->nb[2], T * sizeof(float)), tail));
        auto * conv = ggml_silu(ctx, ggml_ssm_conv(ctx, sx, L.conv));  // [conv_dim, T, S]
        auto * q = ggml_view_4d(ctx, conv, dk, hk, T, S, dk * sizeof(float), conv->nb[1], conv->nb[2], 0);
        auto * k = ggml_view_4d(ctx, conv, dk, hk, T, S, dk * sizeof(float), conv->nb[1], conv->nb[2], key_dim * sizeof(float));
        auto * v = ggml_view_4d(ctx, conv, dv, hv, T, S, dv * sizeof(float), conv->nb[1], conv->nb[2], 2 * key_dim * sizeof(float));
        q = ggml_l2_norm(ctx, q, 1.0e-6F);  // rows are contiguous inside the strided view
        k = ggml_l2_norm(ctx, k, 1.0e-6F);
        const int64_t rep = hv / hk;
        if (rep > 1) {  // value head h reads key head h / rep
            q = ggml_reshape_4d(ctx, ggml_repeat_4d(ctx, ggml_reshape_4d(ctx, q, dk, 1, hk, N), dk, rep, hk, N), dk, hv, T, S);
            k = ggml_reshape_4d(ctx, ggml_repeat_4d(ctx, ggml_reshape_4d(ctx, k, dk, 1, hk, N), dk, rep, hk, N), dk, hv, T, S);
        }
        auto * beta = ggml_reshape_4d(ctx, ggml_sigmoid(ctx, beta_in), 1, hv, T, S);
        auto * a = ggml_softplus(ctx, ggml_add(ctx, a_in, L.dt_bias));
        auto * gate = ggml_reshape_4d(ctx, ggml_mul(ctx, a, L.neg_a), 1, hv, T, S);
        const size_t state_slot = static_cast<size_t>(state_values) * sizeof(float);
        auto * state = ggml_view_3d(ctx, L.state, state_values, 1, S, state_slot, state_slot, g.base * state_slot);
        auto * delta = ggml_gated_delta_net(ctx, q, k, v, gate, beta, state);
        auto * att = ggml_view_2d(ctx, delta, value_dim, N, delta->nb[1], 0);
        ggml_build_forward_expand(g.graph, ggml_cpy(ctx,
            ggml_view_1d(ctx, delta, state_values * S, N * value_dim * sizeof(float)),
            ggml_view_1d(ctx, L.state, state_values * S, g.base * state_slot)));
        auto * o = norm(ctx, ggml_reshape_2d(ctx, att, dv, hv * N), L.lin_norm, b.rms_norm_eps);
        o = ggml_swiglu_split(ctx, ggml_reshape_2d(ctx, z, dv, hv * N), o);
        return ggml_mul_mat(ctx, L.w_out, ggml_reshape_2d(ctx, o, value_dim, N));
    }

    ggml_tensor * moe(ggml_context * ctx, const BackboneLayer & L, ggml_tensor * x, int64_t N) {
        const auto & b = assets_->backbone;
        const int64_t E = b.experts, K = b.experts_per_token;
        ggml_tensor * logits = nullptr;
        ggml_tensor * shared_scale = nullptr;
        if (L.route) {
            auto * y = ggml_mul_mat(ctx, L.route, x);  // [E + 1, N]
            logits = ggml_cont(ctx, ggml_view_2d(ctx, y, E, N, y->nb[1], 0));
            shared_scale = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_2d(ctx, y, 1, N, y->nb[1], E * sizeof(float))));
        } else {
            logits = ggml_mul_mat(ctx, L.router, x);           // [E, N]
            shared_scale = ggml_sigmoid(ctx, ggml_mul_mat(ctx, L.sh_scale, x));
        }
        auto * probs = ggml_soft_max(ctx, logits);
        auto * probs3 = ggml_reshape_3d(ctx, probs, 1, E, N);
        auto * selected = ggml_argsort_top_k(ctx, probs, static_cast<int>(K));  // [K, N]
        auto * weights = ggml_get_rows(ctx, probs3, selected);    // [1, K, N]
        weights = ggml_reshape_2d(ctx, weights, K, N);
        auto * sum = ggml_clamp(ctx, ggml_sum_rows(ctx, weights), 6.103515625e-5F, INFINITY);
        weights = ggml_reshape_3d(ctx, ggml_div(ctx, weights, sum), 1, K, N);
        auto * x3 = ggml_reshape_3d(ctx, x, b.hidden_size, 1, N);
        ggml_tensor * act = nullptr;
        if (L.exp_gate_up) {
            act = ggml_swiglu(ctx, ggml_mul_mat_id(ctx, L.exp_gate_up, x3, selected));  // [inter, K, N]
        } else {
            auto * gate = ggml_mul_mat_id(ctx, L.exp_gate, x3, selected);
            auto * up = ggml_mul_mat_id(ctx, L.exp_up, x3, selected);
            act = ggml_swiglu_split(ctx, gate, up);
        }
        auto * experts = ggml_mul_mat_id(ctx, L.exp_down, act, selected);  // [hidden, K, N]
        experts = ggml_mul(ctx, experts, weights);
        // Expand the expert views first so the adds are consecutive and ggml-vulkan fuses them into one multi_add.
        std::vector<ggml_tensor *> parts;
        for (int64_t e = 0; e < K; ++e) {
            parts.push_back(ggml_view_2d(ctx, experts, b.hidden_size, N, experts->nb[2], e * experts->nb[1]));
            ggml_build_forward_expand(graph_, parts.back());
        }
        ggml_tensor * out = parts[0];
        for (int64_t e = 1; e < K; ++e) out = ggml_add(ctx, out, parts[e]);
        auto * shared_act = L.sh_gate_up ? ggml_swiglu(ctx, ggml_mul_mat(ctx, L.sh_gate_up, x))
                                         : ggml_swiglu_split(ctx, ggml_mul_mat(ctx, L.sh_gate, x), ggml_mul_mat(ctx, L.sh_up, x));
        auto * shared = ggml_mul(ctx, ggml_mul_mat(ctx, L.sh_down, shared_act), shared_scale);
        return ggml_add(ctx, out, shared);
    }

    ggml_tensor * tower_layer(ggml_context * ctx, StepGraph & g, int64_t index, ggml_tensor * hidden, ggml_tensor * codes,
                              ggml_tensor ** logits) {
        const auto & c = assets_->tower;
        const auto & w = tower_[index];
        const int64_t N = g.tokens * g.seqs;
        const int64_t D = c.head_dim, qw = c.heads * D, kw = c.kv_heads * D;
        auto * e = ggml_mul(ctx, ggml_get_rows(ctx, w.embedding, codes), g.label);
        if (index == 0) e = ggml_add(ctx, e, g.extra);
        auto * mixed = ggml_mul_mat(ctx, w.eh, ggml_concat(ctx, norm(ctx, e, w.enorm, c.norm_eps),
                                                           norm(ctx, hidden, w.hnorm, c.norm_eps), 0));
        auto * qkv = ggml_mul_mat(ctx, w.qkv, norm(ctx, mixed, w.in_norm, c.norm_eps));  // [2qw + 2kw, N]
        auto * q = ggml_view_3d(ctx, qkv, D, c.heads, N, D * sizeof(float), qkv->nb[1], 0);
        auto * gate = ggml_cont(ctx, ggml_view_2d(ctx, qkv, qw, N, qkv->nb[1], qw * sizeof(float)));
        auto * k = ggml_view_3d(ctx, qkv, D, c.kv_heads, N, D * sizeof(float), qkv->nb[1], 2 * qw * sizeof(float));
        auto * v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kw, N, qkv->nb[1], (2 * qw + kw) * sizeof(float)));
        const int rot = static_cast<int>(D * c.rotary_fraction);
        q = ggml_rope_ext(ctx, norm(ctx, q, w.q_norm, c.norm_eps), g.pos, nullptr, rot, GGML_ROPE_TYPE_NORMAL, 0,
                          c.rope_theta, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
        k = ggml_rope_ext(ctx, norm(ctx, k, w.k_norm, c.norm_eps), g.pos, nullptr, rot, GGML_ROPE_TYPE_NORMAL, 0,
                          c.rope_theta, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
        auto * att = attention(ctx, g, q, k, v, w.key_cache, w.value_cache, D, c.heads, c.kv_heads);
        att = ggml_mul(ctx, att, ggml_sigmoid(ctx, gate));
        auto * residual = ggml_add(ctx, mixed, ggml_mul_mat(ctx, w.wo, att));
        auto * ff = norm(ctx, residual, w.post_norm, c.norm_eps);
        auto * act = w.w13 ? ggml_swiglu(ctx, ggml_mul_mat(ctx, w.w13, ff))
                           : ggml_swiglu_split(ctx, ggml_mul_mat(ctx, w.w1, ff), ggml_mul_mat(ctx, w.w3, ff));
        auto * out = norm(ctx, ggml_add(ctx, residual, ggml_mul_mat(ctx, w.w2, act)), w.final_norm, c.norm_eps);
        if (logits && w.head) *logits = ggml_mul_mat(ctx, w.head, out);
        return out;
    }

    // Ranked Gumbel-max sampler over [V, N] logits. Returns candidate indices [N] (I32).
    struct Ranked {
        ggml_tensor * index = nullptr;   // I32 [kRanks, N] candidate ids, best first
        ggml_tensor * logits = nullptr;  // f32 [kRanks, N]
    };
    Ranked rank(ggml_context * ctx, ggml_tensor * logits) {
        const int64_t V = logits->ne[0], N = logits->ne[1];
        Ranked r;
        r.index = ggml_cont(ctx, ggml_argsort_top_k(ctx, logits, kRanks));
        r.logits = ggml_reshape_2d(ctx, ggml_get_rows(ctx, ggml_reshape_3d(ctx, logits, 1, V, N), r.index), kRanks, N);
        return r;
    }
    ggml_tensor * scores(ggml_context * ctx, ggml_tensor * logits, const Ranked & r, float temperature, float top_p,
                         ggml_tensor * drop_k, ggml_tensor * noise) {
        const int64_t V = logits->ne[0], N = logits->ne[1];
        auto * probs = ggml_soft_max_ext(ctx, logits, nullptr, 1.0F / temperature, 0.0F);
        auto * ranked = ggml_reshape_2d(ctx, ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, V, N), r.index), kRanks, N);
        auto * cum = ggml_cumsum(ctx, ranked);
        // 1 where the cumulative probability exceeds top_p (rank 0 is always kept).
        auto * over = ggml_clamp(ctx, ggml_scale_bias(ctx, cum, 1.0e7F, -top_p * 1.0e7F), 0.0F, 1.0F);
        auto * drop = ggml_clamp(ctx, ggml_add(ctx, ggml_mul(ctx, over, not_first_), drop_k), 0.0F, 1.0F);
        auto * s = ggml_add(ctx, ggml_scale(ctx, r.logits, 1.0F / temperature), noise);
        return ggml_add(ctx, s, ggml_scale(ctx, drop, kDropped));
    }
    ggml_tensor * pick(ggml_context * ctx, const Ranked & r, ggml_tensor * score) {
        const int64_t N = score->ne[1];
        auto * best = ggml_argmax(ctx, score);  // [N]
        auto * id = ggml_get_rows(ctx, ggml_reshape_3d(ctx, r.index, 1, kRanks, N), ggml_reshape_2d(ctx, best, 1, N));
        return ggml_reshape_1d(ctx, id, N);
    }
    ggml_tensor * noise_view(ggml_context * ctx, StepGraph & g, int sampler) {
        const int64_t N = g.tokens * g.seqs;
        return ggml_view_2d(ctx, g.noise, kRanks, N, g.noise->nb[1], sampler * g.noise->nb[2]);
    }

    std::unique_ptr<StepGraph> build(int64_t T, int64_t S, int64_t base, int64_t keys, bool decode) {
        const auto start = Clock::now();
        const auto & b = assets_->backbone;
        const auto & c = assets_->tower;
        auto g = std::make_unique<StepGraph>();
        g->tokens = T; g->seqs = S; g->base = base; g->keys = keys; g->decode = decode;
        ggml_init_params params{48ull * 1024 * 1024, nullptr, true};
        g->ctx.reset(ggml_init(params));
        auto * ctx = g->ctx.get();
        g->graph = ggml_new_graph_custom(ctx, 32768, false);
        graph_ = g->graph;
        const int64_t N = T * S, H = b.hidden_size;
        auto input = [&](ggml_type type, std::initializer_list<int64_t> ne) {
            std::vector<int64_t> d(ne);
            auto * x = ggml_new_tensor(ctx, type, static_cast<int>(d.size()), d.data());
            ggml_set_input(x);
            return x;
        };
        g->tok = input(GGML_TYPE_I32, {N});
        g->add = input(GGML_TYPE_F32, {H, N});
        g->scale = input(GGML_TYPE_F32, {1, N});
        g->extra = input(GGML_TYPE_F32, {H, N});
        g->label = input(GGML_TYPE_F32, {1, N});
        g->pos = input(GGML_TYPE_I32, {N});
        g->kv_index = input(GGML_TYPE_I32, {N});
        g->packed = decode && T == 1 && !(getenv("D3_FA_PACK") && std::string(getenv("D3_FA_PACK")) == "0");
        g->mask_rows = g->packed ? b.heads / b.kv_heads : T;
        if (g->packed && (b.heads / b.kv_heads != c.heads / c.kv_heads)) throw std::runtime_error("Drama 3 GQA packing needs equal group sizes");
        g->mask = input(GGML_TYPE_F16, {keys, g->mask_rows, 1, S});
        if (decode) {
            g->noise = input(GGML_TYPE_F32, {kRanks, N, kSamplers});
            g->recent = input(GGML_TYPE_F32, {vocab_, N});
        } else {
            g->codes = input(GGML_TYPE_I32, {N, 10});
        }

        auto * x = ggml_get_rows(ctx, token_embedding_, g->tok);
        x = ggml_mul(ctx, ggml_add(ctx, x, g->add), g->scale);
        for (const auto & L : layers_) {
            auto * h = norm(ctx, x, L.in_norm, b.rms_norm_eps);
            h = L.full ? full_attention(ctx, *g, L, h) : delta_net(ctx, *g, L, h);
            x = ggml_add(ctx, x, h);
            h = norm(ctx, x, L.post_norm, b.rms_norm_eps);
            x = ggml_add(ctx, x, moe(ctx, L, h, N));
        }
        auto * hidden = norm(ctx, x, final_norm_, b.rms_norm_eps);

        if (!decode) {
            auto * th = hidden;
            for (int64_t l = 0; l < static_cast<int64_t>(tower_.size()); ++l) {
                const int64_t book = l < c.shared_layers ? 0 : l - c.shared_layers;
                auto * codes = ggml_view_1d(ctx, g->codes, N, book * g->codes->nb[1]);
                th = tower_layer(ctx, *g, l, th, codes, nullptr);
            }
            g->out = th;
            ggml_set_output(g->out);
            ggml_build_forward_expand(g->graph, g->out);
        } else {
            auto * logits = ggml_mul_mat(ctx, lm_head_, hidden);  // [V, N]: semantic codes, then EOS
            const auto ranked = rank(ctx, logits);
            auto * s1 = scores(ctx, logits, ranked, opt_.temperature, opt_.top_p, drop_main_, noise_view(ctx, *g, 0));
            auto * s2 = scores(ctx, logits, ranked, 1.0F, 0.95F, drop_ras_, noise_view(ctx, *g, 1));
            auto * first = pick(ctx, ranked, s1);
            auto * repeat = ggml_reshape_2d(ctx, ggml_get_rows(ctx, ggml_reshape_3d(ctx, g->recent, 1, vocab_, N),
                                                               ggml_reshape_2d(ctx, first, 1, N)), 1, N);
            auto * keep = ggml_scale_bias(ctx, repeat, -1.0F, 1.0F);
            auto * chosen = ggml_add(ctx, ggml_mul(ctx, s1, keep), ggml_mul(ctx, s2, repeat));
            auto * semantic = pick(ctx, ranked, chosen);  // [N], vocab_ - 1 = EOS
            auto * code = ggml_reshape_1d(ctx, ggml_get_rows(ctx, semantic_lut_, semantic), N);
            ggml_tensor * out = ggml_reshape_2d(ctx, semantic, 1, N);
            auto * th = hidden;
            for (int64_t l = 0; l < static_cast<int64_t>(tower_.size()); ++l) {
                ggml_tensor * head_logits = nullptr;
                th = tower_layer(ctx, *g, l, th, code, &head_logits);
                if (head_logits) {
                    const int sampler = 2 + static_cast<int>(l - c.shared_layers);
                    const auto r = rank(ctx, head_logits);
                    code = pick(ctx, r, scores(ctx, head_logits, r, opt_.temperature, opt_.top_p, drop_main_,
                                               noise_view(ctx, *g, sampler)));
                    out = ggml_concat(ctx, out, ggml_reshape_2d(ctx, code, 1, N), 0);
                }
            }
            g->out = out;  // [10, N]
            ggml_set_output(g->out);
            ggml_build_forward_expand(g->graph, g->out);
        }
        stats_.build_ms += ms_since(start);
        if (opt_.verbose) {
            std::cerr << "[drama3_batch] built " << (decode ? "decode" : "prefill") << " graph T=" << T << " S=" << S
                      << " keys=" << keys << " nodes=" << ggml_graph_n_nodes(g->graph) << " in " << ms_since(start)
                      << " ms\n";
        }
        return g;
    }

    void compute(StepGraph & g) {
        if (core::compute_backend_graph(backend_, g.graph, nullptr, "drama3_batch") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Drama 3 batch graph compute failed");
        }
    }

    // ------------------------------------------------------------------ host inputs
    void frame_inputs(const Drama3Frame & frame, float * add, float & scale, float * extra) const {
        const auto & c = assets_->tower;
        const int64_t H = c.hidden_size;
        std::fill(add, add + H, 0.0F);
        std::fill(extra, extra + H, 0.0F);
        scale = 1.0F;
        if (!is_semantic(frame, c)) return;
        auto accumulate = [&](float * out, int64_t book) {
            const int32_t code = frame[book + 1];
            const float * e = code_embeddings_.data() + (book * c.codebook_size + code) * H;
            for (int64_t i = 0; i < H; ++i) out[i] += e[i];
        };
        for (int64_t book = 1; book < 1 + c.backbone_codebooks; ++book) accumulate(add, book);
        for (int64_t book = 1 + c.backbone_codebooks; book < 10; ++book) accumulate(extra, book);
        scale = 1.0F / std::sqrt(1.0F + static_cast<float>(c.backbone_codebooks));
    }

    std::vector<Drama3Frame> text_frames(const std::string & text) const {
        std::vector<Drama3Frame> out;
        for (const auto token : tokenizer_.encode(text)) {
            Drama3Frame f{};
            f[0] = token;
            out.push_back(f);
        }
        return out;
    }

    // Prompt split as build_drama3_prompt: [system + references + <|im_end|>\n] [user turn].
    std::vector<Drama3Frame> prefix_frames(const Drama3Voice & voice) const {
        Drama3Take take{voice.reference_text, voice.codes};
        auto full = build_drama3_prompt(*assets_, tokenizer_, "x", {take}, {});
        const auto suffix = suffix_frames("x");
        if (full.size() <= suffix.size()) throw std::runtime_error("Drama 3 prompt split failed");
        full.resize(full.size() - suffix.size());
        const auto head = suffix_head();
        full.insert(full.end(), head.begin(), head.end());
        return full;
    }
    // Leading suffix tokens shared by every line ("<|im_start|>user\n<|speaker:0|>"): they go into the cached
    // voice prefix instead of being teacher-forced for every line.
    std::vector<Drama3Frame> suffix_head() const {
        if (!suffix_head_.empty() || getenv("D3_NO_SUFFIX_HEAD")) return suffix_head_;
        const auto a = suffix_frames("Hello there.");
        const auto b = suffix_frames("[sighs] What now?");
        size_t n = 0;
        while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
        suffix_head_.assign(a.begin(), a.begin() + static_cast<long>(n));
        return suffix_head_;
    }
    std::vector<Drama3Frame> job_suffix(const std::string & text) const {
        auto s = suffix_frames(text);
        const auto head = suffix_head();
        if (s.size() <= head.size() || !std::equal(head.begin(), head.end(), s.begin())) {
            throw std::runtime_error("Drama 3 batch: line text does not start like the cached prefix: " + text);
        }
        s.erase(s.begin(), s.begin() + static_cast<long>(head.size()));
        return s;
    }
    std::vector<Drama3Frame> suffix_frames(const std::string & text) const {
        validate_drama3_text(text, "input");
        std::vector<Drama3Frame> out;
        for (const auto * part : {"<|im_start|>user\n"}) {
            auto f = text_frames(part);
            out.insert(out.end(), f.begin(), f.end());
        }
        // tagged(text, false): prefix a speaker tag unless the text starts with one.
        std::string body = text;
        size_t lead = 0;
        while (lead < body.size() && (body[lead] == ' ' || (body[lead] >= 9 && body[lead] <= 13))) ++lead;
        if (body.compare(lead, 10, "<|speaker:") != 0) body = "<|speaker:0|> " + body;
        auto f = text_frames(body);
        out.insert(out.end(), f.begin(), f.end());
        f = text_frames("<|im_end|>\n");
        out.insert(out.end(), f.begin(), f.end());
        f = text_frames("<|im_start|>assistant\n<|voice|>");
        out.insert(out.end(), f.begin(), f.end());
        return out;
    }

    // ------------------------------------------------------------------ voices
    int voice_slot(const Drama3Voice & voice) {
        for (int s = 0; s < opt_.voice_slots; ++s) {
            if (voice_slot_key_[s] == voice.key) {
                voice_lru_.remove(s);
                voice_lru_.push_back(s);
                return s;
            }
        }
        int slot = 0;
        bool found = false;
        for (int s = 0; s < opt_.voice_slots; ++s) {
            if (voice_slot_key_[s].empty()) { slot = s; found = true; break; }
        }
        if (!found) { slot = voice_lru_.front(); voice_lru_.pop_front(); }
        prefill_voice(voice, slot);
        voice_slot_key_[slot] = voice.key;
        voice_lru_.remove(slot);
        voice_lru_.push_back(slot);
        return slot;
    }

    void prefill_voice(const Drama3Voice & voice, int slot) {
        const auto start = Clock::now();
        auto prefix = prefix_frames(voice);
        const auto first_suffix = job_suffix("x").front();
        {   // The cached prefix plus the forced suffix must equal the reference prompt builder's output.
            const std::string probe = "[sighs] Hello there, friend.";
            Drama3Take take{voice.reference_text, voice.codes};
            const auto full = build_drama3_prompt(*assets_, tokenizer_, probe, {take}, {});
            auto joined = prefix;
            const auto tail = job_suffix(probe);
            joined.insert(joined.end(), tail.begin(), tail.end());
            if (joined != full) throw std::runtime_error("Drama 3 batch prompt split does not match the prompt builder");
        }
        const int64_t T = static_cast<int64_t>(prefix.size());
        if (T + 64 > opt_.capacity) throw std::runtime_error("Drama 3 voice prefix does not fit the row capacity");
        const int64_t base = opt_.batch + slot;
        prefill_ = build(T, 1, base, T, false);
        if (prefill_galloc_) ggml_gallocr_free(prefill_galloc_);
        prefill_galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (!ggml_gallocr_alloc_graph(prefill_galloc_, prefill_->graph)) throw std::runtime_error("Drama 3 prefill allocation failed");
        const int64_t H = assets_->tower.hidden_size;
        std::vector<int32_t> tok(T), pos(T), kv(T), codes(T * 10);
        std::vector<float> add(H * T), extra(H * T), scale(T), label(T);
        for (int64_t t = 0; t < T; ++t) {
            tok[t] = prefix[t][0];
            pos[t] = static_cast<int32_t>(t);
            kv[t] = static_cast<int32_t>(base * opt_.capacity + t);
            frame_inputs(prefix[t], add.data() + t * H, scale[t], extra.data() + t * H);
            const auto & next = t + 1 < T ? prefix[t + 1] : first_suffix;
            label[t] = is_semantic(next, assets_->tower) ? 1.0F : 0.0F;
            for (int book = 0; book < 10; ++book) codes[book * T + t] = next[1 + book];
        }
        std::vector<ggml_fp16_t> mask(T * T);
        const auto zero = ggml_fp32_to_fp16(0.0F), ninf = ggml_fp32_to_fp16(-INFINITY);
        for (int64_t q = 0; q < T; ++q) for (int64_t k = 0; k < T; ++k) mask[q * T + k] = k <= q ? zero : ninf;
        auto & g = *prefill_;
        ggml_backend_tensor_set(g.tok, tok.data(), 0, tok.size() * 4);
        ggml_backend_tensor_set(g.pos, pos.data(), 0, pos.size() * 4);
        ggml_backend_tensor_set(g.kv_index, kv.data(), 0, kv.size() * 4);
        ggml_backend_tensor_set(g.codes, codes.data(), 0, codes.size() * 4);
        ggml_backend_tensor_set(g.add, add.data(), 0, add.size() * 4);
        ggml_backend_tensor_set(g.extra, extra.data(), 0, extra.size() * 4);
        ggml_backend_tensor_set(g.scale, scale.data(), 0, scale.size() * 4);
        ggml_backend_tensor_set(g.label, label.data(), 0, label.size() * 4);
        ggml_backend_tensor_set(g.mask, mask.data(), 0, mask.size() * 2);
        // Fresh recurrent state for the voice slot.
        clear_slot(base);
        compute(g);
        prefill_.reset();
        ggml_gallocr_free(prefill_galloc_);
        prefill_galloc_ = nullptr;
        voice_prefix_len_[voice.key] = T;
        stats_.voice_prefills++;
        stats_.prefill_ms += ms_since(start);
        if (opt_.verbose) std::cerr << "[drama3_batch] voice " << voice.key << " prefix " << T << " tokens in "
                                    << ms_since(start) << " ms\n";
    }

    // ------------------------------------------------------------------ slot copies
    void ensure_copy_graph() {
        if (copy_graph_) return;
        copy_graph_ = std::make_unique<StepGraph>();
        ggml_init_params params{4ull * 1024 * 1024, nullptr, true};
        copy_graph_->ctx.reset(ggml_init(params));
        auto * ctx = copy_graph_->ctx.get();
        copy_graph_->graph = ggml_new_graph_custom(ctx, 4096, false);
        copy_src_ = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        copy_dst_ = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_input(copy_src_);
        ggml_set_input(copy_dst_);
        auto copy = [&](ggml_tensor * t) {
            const int64_t per_slot = ggml_nelements(t) / slots_;
            auto * flat = ggml_reshape_2d(ctx, t, per_slot, slots_);
            ggml_build_forward_expand(copy_graph_->graph,
                ggml_set_rows(ctx, flat, ggml_get_rows(ctx, flat, copy_src_), copy_dst_));
        };
        for (auto & L : layers_) {
            if (L.full) { copy(L.key_cache); copy(L.value_cache); }
            else { copy(L.conv_tail); copy(L.state); }
        }
        for (auto & w : tower_) { copy(w.key_cache); copy(w.value_cache); }
        copy_galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (!ggml_gallocr_alloc_graph(copy_galloc_, copy_graph_->graph)) throw std::runtime_error("Drama 3 copy graph allocation failed");
    }
    void copy_slot(int64_t from, int64_t to) {
        ensure_copy_graph();
        const int32_t s = static_cast<int32_t>(from), d = static_cast<int32_t>(to);
        ggml_backend_tensor_set(copy_src_, &s, 0, 4);
        ggml_backend_tensor_set(copy_dst_, &d, 0, 4);
        compute(*copy_graph_);
    }
    void clear_slot(int64_t slot) {
        for (auto & L : layers_) {
            if (L.full) continue;
            std::vector<float> zero(static_cast<size_t>(ggml_nelements(L.state) / slots_), 0.0F);
            ggml_backend_tensor_set(L.state, zero.data(), slot * zero.size() * 4, zero.size() * 4);
            std::vector<float> zt(static_cast<size_t>(ggml_nelements(L.conv_tail) / slots_), 0.0F);
            ggml_backend_tensor_set(L.conv_tail, zt.data(), slot * zt.size() * 4, zt.size() * 4);
        }
    }

    // ------------------------------------------------------------------ rows
    void admit(int r, const Drama3BatchJob & job) {
        const auto start = Clock::now();
        auto & row = rows_[r];
        const int slot = voice_slot(*job.voice);
        const auto start_copy = Clock::now();
        copy_slot(opt_.batch + slot, r);
        row = Row{};
        row.active = true;
        row.job = job;
        row.suffix = job_suffix(job.text);
        row.forced = 0;
        const int64_t prefix = voice_prefix_len_.at(job.voice->key);
        row.position = prefix;
        row.prompt_tokens = prefix + static_cast<int64_t>(row.suffix.size());
        const int64_t room = opt_.capacity - row.prompt_tokens;
        if (room < 16) throw std::runtime_error("Drama 3 job " + job.id + " does not fit the row capacity");
        row.limit = job.max_frames > 0 ? std::min<int64_t>(job.max_frames, room) : room;
        row.input = row.suffix[0];
        row.random.seed(job.seed);
        row_finished_[r] = false;
        row_limited_[r] = false;
        stats_.admit_ms += ms_since(start_copy);
        (void) start;
    }

    void step() {
        const auto host_start = Clock::now();
        const auto & c = assets_->tower;
        const int64_t B = opt_.batch, H = c.hidden_size;
        int64_t max_pos = 0;
        for (int r = 0; r < B; ++r) if (rows_[r].active) max_pos = std::max(max_pos, rows_[r].position);
        const int64_t keys = std::min<int64_t>(opt_.capacity, (max_pos + opt_.kv_bucket) / opt_.kv_bucket * opt_.kv_bucket);
        auto & g = decode_graph(keys);
        tok_.assign(B, 0); pos_.assign(B, 0); kv_.assign(B, 0);
        add_.assign(B * H, 0.0F); extra_.assign(B * H, 0.0F); scale_.assign(B, 1.0F); label_.assign(B, 0.0F);
        const int64_t mrows = g.mask_rows;
        mask_.assign(keys * mrows * B, ggml_fp32_to_fp16(-INFINITY));
        noise_.resize(static_cast<size_t>(kRanks * B * kSamplers));
        recent_.assign(vocab_ * B, 0.0F);
        const auto zero = ggml_fp32_to_fp16(0.0F);
        std::uniform_real_distribution<float> uniform(1.0e-7F, 1.0F - 1.0e-7F);
        for (int r = 0; r < B; ++r) {
            auto & row = rows_[r];
            if (!row.active) {
                kv_[r] = static_cast<int32_t>(r * opt_.capacity);
                for (int64_t m = 0; m < mrows; ++m) mask_[(r * mrows + m) * keys] = zero;
                continue;
            }
            tok_[r] = row.input[0];
            pos_[r] = static_cast<int32_t>(row.position);
            kv_[r] = static_cast<int32_t>(r * opt_.capacity + row.position);
            frame_inputs(row.input, add_.data() + r * H, scale_[r], extra_.data() + r * H);
            label_[r] = row.generating() ? 1.0F : 0.0F;
            for (int64_t m = 0; m < mrows; ++m) {
                std::fill(mask_.begin() + (r * mrows + m) * keys, mask_.begin() + (r * mrows + m) * keys + row.position + 1, zero);
            }
            for (int s = 0; s < kSamplers; ++s) {
                float * n = noise_.data() + (static_cast<size_t>(s) * B + r) * kRanks;
                for (int k = 0; k < kRanks; ++k) n[k] = -std::log(-std::log(uniform(row.random)));
            }
            for (const auto token : row.recent) recent_[r * vocab_ + (token - c.semantic_begin)] = 1.0F;
        }
        ggml_backend_tensor_set(g.tok, tok_.data(), 0, B * 4);
        ggml_backend_tensor_set(g.pos, pos_.data(), 0, B * 4);
        ggml_backend_tensor_set(g.kv_index, kv_.data(), 0, B * 4);
        ggml_backend_tensor_set(g.add, add_.data(), 0, add_.size() * 4);
        ggml_backend_tensor_set(g.extra, extra_.data(), 0, extra_.size() * 4);
        ggml_backend_tensor_set(g.scale, scale_.data(), 0, B * 4);
        ggml_backend_tensor_set(g.label, label_.data(), 0, B * 4);
        ggml_backend_tensor_set(g.mask, mask_.data(), 0, mask_.size() * 2);
        ggml_backend_tensor_set(g.noise, noise_.data(), 0, noise_.size() * 4);
        ggml_backend_tensor_set(g.recent, recent_.data(), 0, recent_.size() * 4);
        stats_.host_ms += ms_since(host_start);
        const auto step_start = Clock::now();
        compute(g);
        out_.resize(static_cast<size_t>(10 * B));
        ggml_backend_tensor_get(g.out, out_.data(), 0, out_.size() * 4);
        stats_.step_ms += ms_since(step_start);
        stats_.steps++;
        for (int r = 0; r < B; ++r) {
            auto & row = rows_[r];
            if (!row.active) continue;
            stats_.row_steps++;
            if (!row.generating()) {
                row.forced++;
                row.position++;
                row.input = row.suffix[row.forced];
                stats_.forced_tokens++;
                continue;
            }
            const int32_t * o = out_.data() + r * 10;
            if (o[0] >= c.codebook_size) { row_finished_[r] = true; continue; }
            Drama3Frame next{};
            next[0] = c.semantic_begin + o[0];
            for (int book = 0; book < 10; ++book) next[1 + book] = o[book];
            row.output.push_back(next);
            stats_.generated_frames++;
            row.recent.push_back(next[0]);
            if (row.recent.size() > kRasWindow) row.recent.pop_front();
            row.position++;
            row.input = next;
            if (static_cast<int64_t>(row.output.size()) >= row.limit || row.position >= opt_.capacity) {
                row_finished_[r] = true;
                row_limited_[r] = true;
            }
        }
    }

    StepGraph & decode_graph(int64_t keys) {
        auto it = graphs_.find(keys);
        if (it == graphs_.end()) it = graphs_.emplace(keys, build(1, opt_.batch, 0, keys, true)).first;
        if (current_ != it->second.get()) {
            if (!ggml_gallocr_alloc_graph(galloc_, it->second->graph)) throw std::runtime_error("Drama 3 decode allocation failed");
            current_ = it->second.get();
        }
        return *it->second;
    }

    std::shared_ptr<const Drama3Assets> assets_;
    core::ExecutionContext & execution_;
    ggml_backend_t backend_;
    Drama3BatchOptions opt_;
    fish_audio::FishAudioTextTokenizer tokenizer_;
    HostTensors constants_;
    int64_t slots_ = 0;
    int64_t vocab_ = 0;
    std::unique_ptr<core::BackendWeightStore> store_;
    ggml_tensor * token_embedding_ = nullptr;
    ggml_tensor * final_norm_ = nullptr;
    ggml_tensor * lm_head_ = nullptr;
    ggml_tensor * semantic_lut_ = nullptr;
    ggml_tensor * not_first_ = nullptr;
    ggml_tensor * drop_main_ = nullptr;
    ggml_tensor * drop_ras_ = nullptr;
    std::vector<BackboneLayer> layers_;
    std::vector<TowerLayer> tower_;
    std::vector<float> code_embeddings_;
    std::vector<OwnedTensor> owned_;
    int fused_count_ = 0;
    CtxPtr state_ctx_;
    ggml_backend_buffer_t state_buffer_ = nullptr;
    std::unique_ptr<core::ExecutionContext> codec_execution_;
    std::unique_ptr<codecs::FishDacCodecRuntime> codec_;
    std::unique_ptr<core::ExecutionContext> cpu_codec_execution_;
    std::vector<std::unique_ptr<codecs::FishDacCodecRuntime>> cpu_codecs_;
    ggml_gallocr_t galloc_ = nullptr;
    ggml_gallocr_t prefill_galloc_ = nullptr;
    ggml_gallocr_t copy_galloc_ = nullptr;
    std::map<int64_t, std::unique_ptr<StepGraph>> graphs_;
    StepGraph * current_ = nullptr;
    std::unique_ptr<StepGraph> prefill_;
    std::unique_ptr<StepGraph> copy_graph_;
    ggml_tensor * copy_src_ = nullptr;
    ggml_tensor * copy_dst_ = nullptr;
    std::vector<Row> rows_;
    std::unordered_map<int, bool> row_finished_, row_limited_;
    std::vector<std::string> voice_slot_key_;
    mutable std::vector<Drama3Frame> suffix_head_;
    ggml_cgraph * graph_ = nullptr;  // graph being built
    std::list<int> voice_lru_;
    std::unordered_map<std::string, int64_t> voice_prefix_len_;
    std::vector<int32_t> tok_, pos_, kv_, out_;
    std::vector<float> add_, extra_, scale_, label_, noise_, recent_;
    std::vector<ggml_fp16_t> mask_;
};

Drama3BatchEngine::Drama3BatchEngine(std::shared_ptr<const Drama3Assets> assets, core::ExecutionContext & execution,
                                     Drama3BatchOptions options)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, options)) {}
Drama3BatchEngine::~Drama3BatchEngine() = default;
codecs::FishDacCodes Drama3BatchEngine::encode_reference(const runtime::AudioBuffer & audio) {
    return impl_->encode_reference(audio);
}
runtime::AudioBuffer Drama3BatchEngine::decode(const codecs::FishDacCodes & codes) { return impl_->decode(codes); }
runtime::AudioBuffer Drama3BatchEngine::decode_cpu(int worker, const codecs::FishDacCodes & codes) {
    return impl_->decode_cpu(worker, codes);
}
void Drama3BatchEngine::run(const std::vector<Drama3BatchJob> & jobs, const std::function<void(Drama3BatchResult &&)> & done) {
    impl_->run(jobs, done);
}
const Drama3BatchStats & Drama3BatchEngine::stats() const { return impl_->stats_; }
void Drama3BatchEngine::reset_stats() { impl_->stats_ = {}; }

}  // namespace engine::models::fish_drama3
