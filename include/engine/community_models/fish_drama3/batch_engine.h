#pragma once

// Batched Drama 3 generation for throughput (many short lines, a few voices).
//
// One ggml graph per decode step runs B independent rows: token embedding,
// the 40-layer backbone, a sub-vocabulary LM head, GPU sampling with RAS, the
// 13 tower layers and their codebook sampling. Only the 10 sampled codes per
// row are read back each step. Rows are continuously refilled: a row that
// ends is replaced by the next job, whose voice prefix (system prompt plus
// reference) is copied from a per-voice cache slot and whose text suffix is
// teacher-forced through the same step graph.

#include "engine/community_models/fish_drama3/assets.h"
#include "engine/community_models/fish_drama3/prompt.h"
#include "engine/framework/codecs/fish_dac_codec_runtime.h"
#include "engine/framework/core/execution_context.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::fish_drama3 {

struct Drama3BatchOptions {
    int batch = 32;            // rows decoded per step
    int capacity = 1024;       // positions per row (prefix + text + generated frames)
    int kv_bucket = 128;       // attention length is rounded up to this
    int voice_slots = 2;       // cached voice prefixes
    float temperature = 1.0F;
    float top_p = 0.9F;
    int top_k = 30;
    int codec_cpu_workers = 0;  // >0: also load the codec on the CPU for this many decode workers
    int codec_cpu_threads = 4;  // threads per CPU decode worker
    bool verbose = false;
};

struct Drama3Voice {
    std::string key;
    std::string reference_text;
    codecs::FishDacCodes codes;
};

struct Drama3BatchJob {
    std::string id;
    std::shared_ptr<const Drama3Voice> voice;
    std::string text;
    int64_t max_frames = 0;    // 0: until the row's capacity is used up
    uint32_t seed = 0;
};

struct Drama3BatchResult {
    std::string id;
    codecs::FishDacCodes codes;
    bool reached_limit = false;
    int64_t prompt_tokens = 0;
};

struct Drama3BatchStats {
    int64_t steps = 0;
    int64_t row_steps = 0;        // active rows summed over steps
    int64_t generated_frames = 0;
    int64_t forced_tokens = 0;
    int64_t voice_prefills = 0;
    double step_ms = 0;           // graph compute and readback
    double host_ms = 0;           // input preparation
    double prefill_ms = 0;
    double admit_ms = 0;
    double build_ms = 0;
};

class Drama3BatchEngine {
public:
    Drama3BatchEngine(std::shared_ptr<const Drama3Assets> assets, core::ExecutionContext & execution,
                      Drama3BatchOptions options);
    ~Drama3BatchEngine();
    Drama3BatchEngine(const Drama3BatchEngine &) = delete;
    Drama3BatchEngine & operator=(const Drama3BatchEngine &) = delete;

    codecs::FishDacCodes encode_reference(const runtime::AudioBuffer & audio);
    runtime::AudioBuffer decode(const codecs::FishDacCodes & codes);
    // CPU codec worker `worker` (0 <= worker < codec_cpu_workers); each worker may run on its own thread.
    runtime::AudioBuffer decode_cpu(int worker, const codecs::FishDacCodes & codes);

    // Generates every job. Calls done() as each row finishes (in finishing order).
    void run(const std::vector<Drama3BatchJob> & jobs, const std::function<void(Drama3BatchResult &&)> & done);

    const Drama3BatchStats & stats() const;
    void reset_stats();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::fish_drama3
