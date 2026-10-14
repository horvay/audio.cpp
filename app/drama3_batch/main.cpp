// drama3_batch: batched Fish Audio Drama 3 voice cloning from a TSV of lines.
//
// Each TSV row is out_name<TAB>voice<TAB>text[<TAB>ref_text] (the fish_clone.py format):
//   voice     a reference wav path, or a name looked up as <--voices-dir>/<name>.wav
//   ref_text  what the reference says; defaults to the <voice>.txt sidecar
// Writes <--out>/<out_name>.wav (44.1 kHz mono, 16-bit). Existing outputs are skipped, so runs resume.
// Text may contain Drama 3 inline tags such as [whispering] or [sighs]; [crying] and [sobbing] are read as [tearful].
//
// usage:
//   drama3_batch <jobs.tsv> --model drama-3.gguf --voices-dir DIR --out DIR
//       [--batch 32] [--capacity 1024] [--temperature 1.0] [--top-p 0.9] [--top-k 30]
//       [--max-frames 0] [--codes-cache DIR] [--limit N] [--seed N] [--bench-tsv FILE --tag TAG] [--verbose]
//       [--codec gpu|cpu] [--decode-workers 6] [--codec-threads 4]
// --codec cpu decodes finished lines on CPU worker threads while the GPU keeps generating.

#include "engine/community_models/fish_drama3/assets.h"
#include "engine/community_models/fish_drama3/batch_engine.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/core/execution_context.h"
#include <ggml.h>

#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace engine::models::fish_drama3;
using Clock = std::chrono::steady_clock;

namespace {

struct Line {
    std::string name, voice, text, ref_text;
    fs::path out, wav;
};

uint64_t fnv1a(const std::string & s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

std::string read_text(const fs::path & p) {
    std::ifstream in(p);
    std::stringstream ss;
    ss << in.rdbuf();
    auto s = ss.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    size_t a = 0;
    while (a < s.size() && (s[a] == ' ' || s[a] == '\n')) ++a;
    return s.substr(a);
}

std::vector<std::string> split_tab(const std::string & line) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        const auto pos = line.find('\t', start);
        out.push_back(line.substr(start, pos == std::string::npos ? std::string::npos : pos - start));
        if (pos == std::string::npos) break;
        start = pos + 1;
    }
    return out;
}

void save_codes(const fs::path & p, const engine::codecs::FishDacCodes & c) {
    std::ofstream out(p, std::ios::binary);
    const int64_t head[2] = {c.codebooks, c.frames};
    out.write(reinterpret_cast<const char *>(head), sizeof(head));
    out.write(reinterpret_cast<const char *>(c.codes.data()), static_cast<std::streamsize>(c.codes.size() * 4));
}

bool load_codes(const fs::path & p, engine::codecs::FishDacCodes & c) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return false;
    int64_t head[2];
    if (!in.read(reinterpret_cast<char *>(head), sizeof(head))) return false;
    c.codebooks = head[0];
    c.frames = head[1];
    c.codes.resize(static_cast<size_t>(head[0] * head[1]));
    return static_cast<bool>(in.read(reinterpret_cast<char *>(c.codes.data()), static_cast<std::streamsize>(c.codes.size() * 4)));
}

// Drama 3 tends to produce long crying with few words for [crying]/[sobbing] (seen in the spike and the eval set),
// so those tags are read as [tearful]. Returns the number of tags replaced.
int swap_crying_tags(std::string & text) {
    int count = 0;
    size_t pos = 0;
    while ((pos = text.find('[', pos)) != std::string::npos) {
        const auto end = text.find(']', pos);
        if (end == std::string::npos) break;
        std::string tag = text.substr(pos + 1, end - pos - 1);
        for (auto & c : tag) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (tag.find("crying") != std::string::npos || tag.find("sobbing") != std::string::npos) {
            text.replace(pos, end - pos + 1, "[tearful]");
            ++count;
            pos += 9;
        } else {
            pos = end + 1;
        }
    }
    return count;
}

// Frame cap from the text: generous for normal speech (median is ~1.8 frames per spoken character), so a line that
// reaches it is almost always a runaway take and gets regenerated with another seed.
int64_t frame_cap(const std::string & text) {
    int64_t chars = 0, pauses = 0;
    bool in_tag = false;
    std::string tag;
    for (char ch : text) {
        if (ch == '[') { in_tag = true; tag.clear(); continue; }
        if (in_tag) {
            if (ch == ']') { in_tag = false; if (tag.find("pause") != std::string::npos || tag.find("silence") != std::string::npos) ++pauses; }
            else tag += ch;
            continue;
        }
        ++chars;
    }
    return 100 + chars * 9 / 2 + pauses * 65;
}

[[noreturn]] void usage() {
    std::cerr << "usage: drama3_batch <jobs.tsv> --model M.gguf --voices-dir DIR --out DIR [--batch 32] [--capacity 1024]\n"
                 "       [--temperature 1.0] [--top-p 0.9] [--top-k 30] [--max-frames 0] [--codes-cache DIR]\n"
                 "       [--limit N] [--seed N] [--bench-tsv FILE --tag TAG] [--verbose]\n";
    std::exit(2);
}

}  // namespace

int main(int argc, char ** argv) {
    std::string jobs_path, model, voices_dir = ".", out_dir, codes_cache, bench_tsv, tag = "run";
    Drama3BatchOptions opt;
    int64_t max_frames = 0, limit = 0;
    uint64_t seed_base = 0;
    std::string codec_mode = "gpu";
    int64_t codec_bench = 0;
    std::string save_codes_dir;
    int decode_workers = 6;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "--model") model = next();
        else if (a == "--voices-dir") voices_dir = next();
        else if (a == "--out") out_dir = next();
        else if (a == "--batch") opt.batch = std::stoi(next());
        else if (a == "--capacity") opt.capacity = std::stoi(next());
        else if (a == "--kv-bucket") opt.kv_bucket = std::stoi(next());
        else if (a == "--temperature") opt.temperature = std::stof(next());
        else if (a == "--top-p") opt.top_p = std::stof(next());
        else if (a == "--top-k") opt.top_k = std::stoi(next());
        else if (a == "--max-frames") max_frames = std::stoll(next());
        else if (a == "--codes-cache") codes_cache = next();
        else if (a == "--limit") limit = std::stoll(next());
        else if (a == "--seed") seed_base = std::stoull(next());
        else if (a == "--bench-tsv") bench_tsv = next();
        else if (a == "--tag") tag = next();
        else if (a == "--verbose") opt.verbose = true;
        else if (a == "--codec") codec_mode = next();
        else if (a == "--codec-bench") codec_bench = std::stoll(next());
        else if (a == "--save-codes") save_codes_dir = next();
        else if (a == "--decode-workers") decode_workers = std::stoi(next());
        else if (a == "--codec-threads") opt.codec_cpu_threads = std::stoi(next());
        else if (a == "-h" || a == "--help") usage();
        else if (!a.empty() && a[0] == '-') { std::cerr << "unknown option " << a << "\n"; usage(); }
        else jobs_path = a;
    }
    if (const char * names = getenv("D3_CHECK_RAW")) {
        // Compare raw GGUF bytes with what the weight store loads (TensorStorageType::Native) for comma-separated names.
        auto assets = load_drama3_assets(model);
        std::string all = names;
        size_t pos = 0;
        while (pos <= all.size()) {
            const auto end = all.find(',', pos);
            const auto name = all.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
            pos = end == std::string::npos ? all.size() + 1 : end + 1;
            const auto meta = assets->model_weights->require_metadata(name);
            const auto raw = assets->model_weights->require_tensor_data(name);
            const auto native = assets->model_weights->require_tensor(name, engine::assets::TensorStorageType::Native, meta.shape);
            size_t diff = 0;
            for (size_t i = 0; i < std::min(raw.bytes.size(), native.bytes.size()); ++i) diff += raw.bytes[i] != native.bytes[i];
            std::cerr << name << " dtype=" << meta.dtype << " shape=";
            for (auto d : meta.shape) std::cerr << d << ",";
            std::cerr << " raw=" << raw.bytes.size() << " native=" << native.bytes.size() << " type=" << ggml_type_name(native.type)
                      << " differing bytes=" << diff << "\n";
        }
        return 0;
    }
    if (codec_bench > 0 && !model.empty()) {
        // CPU codec throughput test: decode the first cached reference code file with N workers in parallel.
        auto assets = load_drama3_assets(model);
        engine::core::BackendConfig cpu;
        cpu.type = codec_mode == "gpu" ? engine::core::BackendType::Vulkan : engine::core::BackendType::Cpu;
        cpu.threads = opt.codec_cpu_threads;
        engine::core::ExecutionContext exec(cpu);
        engine::codecs::FishDacCodecRuntimeOptions copt;
        copt.weight_context_bytes = 16 * 1024 * 1024;
        auto comp = engine::codecs::FishDacCodecComponent::load_from_tensor_source(assets->codec_weights, assets->codec, {},
                                                                                    exec.backend(), cpu.type, copt);
        engine::codecs::FishDacCodes codes;
        for (const auto & e : fs::directory_iterator(codes_cache)) { if (load_codes(e.path(), codes)) break; }
        if (codes.frames == 0) { std::cerr << "no cached codes in " << codes_cache << "\n"; return 1; }
        const int64_t frames = std::min<int64_t>(codes.frames, codec_bench);
        engine::codecs::FishDacCodes cut; cut.codebooks = 10; cut.frames = frames;
        for (int64_t b = 0; b < 10; ++b) for (int64_t f = 0; f < frames; ++f) cut.codes.push_back(codes.codes[b * codes.frames + f]);
        std::vector<std::thread> th;
        const auto t0 = Clock::now();
        for (int w = 0; w < decode_workers; ++w) th.emplace_back([&]() {
            engine::codecs::FishDacCodecRuntime rt(comp, cpu, cpu.threads, copt.graph_arena_bytes);
            for (int i = 0; i < 3; ++i) (void) rt.decode_codes(cut);
        });
        for (auto & t : th) t.join();
        const double el = std::chrono::duration<double>(Clock::now() - t0).count();
        const double audio = 3.0 * decode_workers * frames / 21.533;
        std::fprintf(stderr, "codec bench: %d workers x %d threads, %lld frames, %.1f s for %.1f s audio = %.3f s/s\n",
                     decode_workers, cpu.threads, static_cast<long long>(frames), el, audio, el / audio);
        return 0;
    }
    if (jobs_path.empty() || model.empty() || out_dir.empty()) usage();
    if (codes_cache.empty()) codes_cache = (fs::path(out_dir) / ".ref_codes").string();

    // Read jobs, skipping finished outputs.
    std::vector<Line> lines;
    int swapped_tags = 0;
    {
        std::ifstream in(jobs_path);
        if (!in) { std::cerr << "cannot read " << jobs_path << "\n"; return 1; }
        std::string row;
        while (std::getline(in, row)) {
            if (!row.empty() && row.back() == '\r') row.pop_back();
            if (row.empty() || row[0] == '#') continue;
            auto f = split_tab(row);
            if (f.size() < 3) { std::cerr << "need out_name<TAB>voice<TAB>text: " << row << "\n"; return 1; }
            Line l;
            l.name = f[0];
            l.voice = f[1];
            l.text = f[2];
            swapped_tags += swap_crying_tags(l.text);
            std::string out_name = l.name;
            if (out_name.size() < 4 || out_name.substr(out_name.size() - 4) != ".wav") out_name += ".wav";
            l.out = fs::path(out_dir) / out_name;
            if (fs::exists(l.out)) continue;
            l.wav = (l.voice.size() > 4 && l.voice.substr(l.voice.size() - 4) == ".wav") ? fs::path(l.voice)
                                                                                         : fs::path(voices_dir) / (l.voice + ".wav");
            if (!fs::exists(l.wav)) { std::cerr << "voice not found: " << l.voice << " (" << l.wav << ")\n"; return 1; }
            if (f.size() > 3 && !f[3].empty()) l.ref_text = f[3];
            else {
                auto txt = l.wav;
                txt.replace_extension(".txt");
                if (!fs::exists(txt)) { std::cerr << l.name << ": no ref text (4th column or " << txt << ")\n"; return 1; }
                l.ref_text = read_text(txt);
            }
            lines.push_back(std::move(l));
        }
    }
    // Group by voice (the voice prefix is cached), longest lines first inside a voice.
    std::stable_sort(lines.begin(), lines.end(), [](const Line & a, const Line & b) {
        if (a.wav != b.wav) return a.wav < b.wav;
        return a.text.size() > b.text.size();
    });
    if (limit > 0 && static_cast<int64_t>(lines.size()) > limit) lines.resize(static_cast<size_t>(limit));
    std::cerr << "[drama3_batch] " << lines.size() << " lines to generate";
    if (swapped_tags) std::cerr << " (" << swapped_tags << " [crying]/[sobbing] tags read as [tearful])";
    std::cerr << "\n";
    if (lines.empty()) return 0;

    setenv("FISH_DAC_DECODE_BUCKET", "32", 0);  // decode graphs per 32-frame bucket (see fish_dac_codec_runtime.cpp)
    setenv("FISH_DAC_DECODE_GRAPHS", "3", 0);
    if (codec_mode == "cpu") opt.codec_cpu_workers = decode_workers;
    else if (codec_mode != "gpu") usage();
    const auto t_load = Clock::now();
    auto assets = load_drama3_assets(model);
    engine::core::BackendConfig backend;
    backend.type = engine::core::BackendType::Vulkan;
    backend.threads = 8;
    engine::core::ExecutionContext execution(backend);
    Drama3BatchEngine engine(assets, execution, opt);
    const double load_s = std::chrono::duration<double>(Clock::now() - t_load).count();
    std::cerr << "[drama3_batch] model loaded in " << load_s << " s\n";

    // Reference codes, cached on disk per (wav, ref text).
    const auto t_ref = Clock::now();
    fs::create_directories(codes_cache);
    std::map<std::string, std::shared_ptr<Drama3Voice>> voices;
    for (const auto & l : lines) {
        const std::string key = l.wav.string() + "\n" + l.ref_text;
        if (voices.count(key)) continue;
        auto v = std::make_shared<Drama3Voice>();
        v->key = key;
        v->reference_text = l.ref_text;
        const auto stamp = std::to_string(fs::last_write_time(l.wav).time_since_epoch().count());
        char hex[17];
        std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(fnv1a(key + stamp)));
        const auto cache = fs::path(codes_cache) / (l.wav.stem().string() + "." + hex + ".codes");
        if (!load_codes(cache, v->codes)) {
            auto wav = engine::audio::read_wav_f32(l.wav);
            v->codes = engine.encode_reference({wav.sample_rate, wav.channels, std::move(wav.samples)});
            save_codes(cache, v->codes);
        }
        voices[key] = v;
    }
    const double ref_s = std::chrono::duration<double>(Clock::now() - t_ref).count();

    std::vector<Drama3BatchJob> jobs;
    std::map<std::string, const Line *> by_id;
    for (const auto & l : lines) {
        Drama3BatchJob j;
        j.id = l.name;
        j.voice = voices.at(l.wav.string() + "\n" + l.ref_text);
        j.text = l.text;
        j.max_frames = max_frames > 0 ? max_frames : frame_cap(l.text);
        j.seed = static_cast<uint32_t>(fnv1a(l.name) ^ seed_base);
        jobs.push_back(std::move(j));
        by_id[l.name] = &l;
    }

    const auto t_gen = Clock::now();
    double audio_s = 0, decode_s = 0;
    int64_t done_count = 0, limited = 0;
    auto last_report = Clock::now();
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Drama3BatchResult> pending;
    bool finished = false;
    size_t max_backlog = 0;
    auto write_one = [&](const Drama3BatchResult & r, engine::runtime::AudioBuffer && audio, double dec) {
        const Line & l = *by_id.at(r.id);
        fs::create_directories(l.out.parent_path());
        auto tmp = l.out;
        tmp += ".part";
        engine::audio::write_pcm16_wav(tmp, audio.sample_rate, audio.channels, audio.samples);
        fs::rename(tmp, l.out);
        std::lock_guard<std::mutex> lock(mu);
        decode_s += dec;
        const double secs = static_cast<double>(audio.samples.size()) / audio.channels / audio.sample_rate;
        audio_s += secs;
        ++done_count;
        if (r.reached_limit) {
            ++limited;
            std::cerr << "[drama3_batch] " << r.id << ": hit the frame limit (" << r.codes.frames << " frames)\n";
        }
        if (opt.verbose || std::chrono::duration<double>(Clock::now() - last_report).count() > 30 ||
            done_count == static_cast<int64_t>(jobs.size())) {
            last_report = Clock::now();
            const double el = std::chrono::duration<double>(Clock::now() - t_gen).count();
            const double rate = el / std::max(audio_s, 1e-9);
            const double eta = (static_cast<double>(jobs.size()) - done_count) * (el / done_count);
            std::fprintf(stderr, "[drama3_batch] %lld/%zu lines, %.1f s audio, %.1f s elapsed, %.3f s/s, ETA %.0f min, decode backlog %zu\n",
                         static_cast<long long>(done_count), jobs.size(), audio_s, el, rate, eta / 60.0, pending.size());
        }
    };
    std::vector<std::thread> workers;
    // GPU codec: one worker thread with the codec's own Vulkan backend, overlapping generation. CPU codec: N workers.
    const bool threaded_gpu = codec_mode == "gpu" && getenv("D3_ASYNC_DECODE") != nullptr;
    if (codec_mode == "cpu" || threaded_gpu) {
        const int nworkers = codec_mode == "cpu" ? decode_workers : 1;
        for (int w = 0; w < nworkers; ++w) {
            workers.emplace_back([&, w]() {
                while (true) {
                    Drama3BatchResult r;
                    {
                        std::unique_lock<std::mutex> lock(mu);
                        cv.wait(lock, [&] { return finished || !pending.empty(); });
                        if (pending.empty()) return;
                        r = std::move(pending.front());
                        pending.pop_front();
                    }
                    const auto t_dec = Clock::now();
                    auto audio = codec_mode == "cpu" ? engine.decode_cpu(w, r.codes) : engine.decode(r.codes);
                    write_one(r, std::move(audio), std::chrono::duration<double>(Clock::now() - t_dec).count());
                }
            });
        }
    }
    if (!save_codes_dir.empty()) fs::create_directories(save_codes_dir);
    std::map<std::string, int> attempts;
    std::ofstream limited_log(fs::path(out_dir) / "limited.tsv", std::ios::app);
    auto on_done = [&](Drama3BatchResult && r) {
        if (r.reached_limit && attempts[r.id]++ == 0) {
            // Runaway take: regenerate once with another seed, in the same running batch.
            for (const auto & j : jobs) if (j.id == r.id) { auto k = j; k.seed ^= 0x9e3779b9U; engine.requeue(k); }
            limited_log << r.id << "\tretry\t" << r.codes.frames << "\n";
            limited_log.flush();
            return;
        }
        if (r.reached_limit) { limited_log << r.id << "\tkept\t" << r.codes.frames << "\n"; limited_log.flush(); }
        if (!save_codes_dir.empty()) {
            std::string n = r.id;
            std::replace(n.begin(), n.end(), '/', '_');
            save_codes(fs::path(save_codes_dir) / (n + ".codes"), r.codes);
        }
        if (r.codes.frames == 0) {
            std::cerr << "[drama3_batch] " << r.id << ": no frames generated, skipped\n";
            return;
        }
        if (codec_mode == "cpu" || threaded_gpu) {
            std::lock_guard<std::mutex> lock(mu);
            pending.push_back(std::move(r));
            max_backlog = std::max(max_backlog, pending.size());
            cv.notify_one();
            return;
        }
        const auto t_dec = Clock::now();
        auto audio = engine.decode(r.codes);
        write_one(r, std::move(audio), std::chrono::duration<double>(Clock::now() - t_dec).count());
    };
    engine.run(jobs, on_done);
    const double gpu_s = std::chrono::duration<double>(Clock::now() - t_gen).count();
    {
        std::lock_guard<std::mutex> lock(mu);
        finished = true;
    }
    cv.notify_all();
    for (auto & t : workers) t.join();
    std::fprintf(stderr, "[drama3_batch] generation finished after %.1f s; decode tail %.1f s; max decode backlog %zu\n",
                 gpu_s, std::chrono::duration<double>(Clock::now() - t_gen).count() - gpu_s, max_backlog);
    const double gen_s = std::chrono::duration<double>(Clock::now() - t_gen).count();
    const auto & st = engine.stats();
    std::fprintf(stderr,
        "[drama3_batch] DONE %lld lines, %.1f s audio. generate+decode %.1f s (%.3f s/s), decode %.1f s, refs %.1f s,"
        " load %.1f s. end-to-end incl. refs %.3f s/s. limited %lld\n",
        static_cast<long long>(done_count), audio_s, gen_s, gen_s / std::max(audio_s, 1e-9), decode_s, ref_s, load_s,
        (gen_s + ref_s) / std::max(audio_s, 1e-9), static_cast<long long>(limited));
    std::fprintf(stderr,
        "[drama3_batch] steps %lld, row-steps %lld (%.1f%% of slots), frames %lld, forced %lld, step %.1f ms avg"
        " (host %.2f ms), prefill %.1f s for %lld voices, admit %.1f s, graph builds %.1f s\n",
        static_cast<long long>(st.steps), static_cast<long long>(st.row_steps),
        100.0 * st.row_steps / std::max<int64_t>(1, st.steps * opt.batch), static_cast<long long>(st.generated_frames),
        static_cast<long long>(st.forced_tokens), st.step_ms / std::max<int64_t>(1, st.steps),
        st.host_ms / std::max<int64_t>(1, st.steps), st.prefill_ms / 1000.0, static_cast<long long>(st.voice_prefills),
        st.admit_ms / 1000.0, st.build_ms / 1000.0);
    if (!bench_tsv.empty()) {
        const bool fresh = !fs::exists(bench_tsv);
        std::ofstream b(bench_tsv, std::ios::app);
        if (fresh) b << "tag\tbatch\tlines\taudio_s\tgen_decode_s\tdecode_s\tref_s\ts_per_s_gen\ts_per_s_e2e\tstep_ms\tsteps\n";
        b << tag << "\t" << opt.batch << "\t" << done_count << "\t" << audio_s << "\t" << gen_s << "\t" << decode_s << "\t"
          << ref_s << "\t" << (gen_s - decode_s) / std::max(audio_s, 1e-9) << "\t" << (gen_s + ref_s) / std::max(audio_s, 1e-9)
          << "\t" << st.step_ms / std::max<int64_t>(1, st.steps) << "\t" << st.steps << "\n";
    }
    return 0;
}
