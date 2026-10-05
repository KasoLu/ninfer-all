#include "runtime/engine/qwen4_exp_core.h"

#include "artifact/formats.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/startup.h"
#include "models/qwen4_exp/ngram_component.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/target_logprobs.h"
#include "runtime/engine/diagnostics.h"
#include "runtime/engine/effective_thinking_budget.h"
#include "runtime/engine/generation_budget.h"
#include "runtime/engine/model_instance.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <thread>
#include <variant>

namespace ninfer::runtime {
namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(Clock::time_point from, Clock::time_point to) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count());
}

double seconds(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double>(to - from).count();
}

// Requests fail alone on their own errors; anything else may have left the device in an unknown
// state, so it fails the Engine.
bool request_error(const std::exception_ptr& error) {
    try {
        std::rethrow_exception(error);
    } catch (const RequestError&) { return true; } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {}
    return false;
}

} // namespace

bool is_qwen4_exp_artifact(const std::filesystem::path& path) {
    if (path.extension() != ".ninfer") { return false; }
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) { return false; }
    try {
        return models::qwen4_exp::is_qwen4_exp(artifact::Reader(path));
    } catch (const std::exception&) { return false; }
}

ConstructedQwen4Exp construct_qwen4_exp(const EngineOptions& options, DeviceContext& device) {
    const auto start = Clock::now();
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    if (options.enable_vision) {
        throw std::invalid_argument("Qwen3.8-Flash-Next artifacts carry no Vision component yet");
    }
    if (options.speculative.backend != SpeculativeBackend::None ||
        options.speculative.ngram_draft_tokens != 0) {
        throw std::invalid_argument("speculative decoding is not available for Qwen3.8-Flash-Next");
    }
    if (options.context_cache.enabled && !options.context_cache.disk_kv_path.empty()) {
        throw std::invalid_argument("the context cache is not available for Qwen3.8-Flash-Next");
    }
    install_device_route_profile_for(options, device);
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    const artifact::Reader reader(options.artifact_path);
    // The n-gram table is checked before the weights load, so a model without one fails at once.
    auto ngram = models::qwen4_exp::ngram_table_source(
        reader, options.artifact_path,
        models::qwen4_exp::parse_text_config(reader.directory().component("text").config));
    inspect.complete();
    models::qwen4_exp::LoadOptions load;
    load.artifact     = options.artifact_path;
    load.ranks        = device.size();
    load.stage_layers = options.stage_layers;
    load.experts      = options.expert_residency;
    StartupPhaseScope materialize(options.startup_observer, StartupPhase::TargetPlan);
    auto model = models::qwen4_exp::load_model(reader, load, device, &options.startup_observer);
    device.synchronize();
    materialize.complete();

    StartupPhaseScope frontend_phase(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<Qwen4ExpInstance>(Qwen4ExpInstance{
        .model    = nullptr,
        .frontend = models::qwen3_5::make_frontend(
            model->resources(), {.chat_template_path      = options.chat_template_path,
                                 .architecture            = models::Architecture::Qwen4Exp,
                                 .vision_enabled          = false,
                                 .max_context             = options.max_context,
                                 .media_cache_bytes       = options.media_cache_bytes,
                                 .media_live_bytes        = options.media_live_bytes,
                                 .thinking_budget_message = options.thinking_budget_message}),
        .executor = nullptr,
        .capacity = options.max_context});
    frontend_phase.complete();
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    models::qwen4_exp::ExecutorOptions executor;
    executor.max_context     = options.max_context;
    executor.prefill_chunk   = std::clamp<std::uint32_t>(options.prefill_chunk, 64, 4096);
    executor.sequences       = 1;
    executor.ngram           = std::move(ngram);
    executor.ngram_residency = options.ngram_ram ? models::qwen4_exp::NgramResidency::Ram
                                                 : models::qwen4_exp::NgramResidency::Disk;
    executor.expert_cache_bytes =
        options.expert_cache_bytes.value_or(models::qwen4_exp::ExecutorOptions::kAutoExpertCache);
    executor.cuda_graphs = options.use_cuda_graph;
    instance->executor = std::make_unique<models::qwen4_exp::Executor>(*model, device, executor);
    device.synchronize();
    program.complete();
    publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                       "Qwen3.8-Flash-Next: %zu stage(s), experts in %s memory, n-gram table "
                       "%s, state %.0f MiB, workspace %.0f MiB, expert cache %.0f MiB",
                       model->stages().stages(),
                       options.expert_residency == ExpertResidency::Host   ? "pinned host"
                       : options.expert_residency == ExpertResidency::Disk ? "the artifact's files"
                                                                           : "device",
                       options.ngram_ram ? "in RAM" : "read from the artifact",
                       double(instance->executor->memory().state_bytes) / 1048576.0,
                       double(instance->executor->memory().workspace_bytes) / 1048576.0,
                       double(instance->executor->memory().expert_cache_bytes) / 1048576.0);
    if (options.kv_cache != KvCacheStorage::BFloat16) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning,
                           "Qwen3.8-Flash-Next keeps its KV in BF16; the requested KV storage does "
                           "not apply");
    }

    ConstructedQwen4Exp out;
    const auto& stats     = model->storage_stats();
    out.load.architecture = std::string(models::architecture_name(models::Architecture::Qwen4Exp));
    out.load.model_name   = model->info().name;
    out.load.prefill_signature = "qwen4_exp";
    std::set<std::string> formats;
    for (const auto& weight : model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    out.load.weight_formats.assign(formats.begin(), formats.end());
    out.load.load_seconds             = seconds(start, Clock::now());
    out.load.upload_seconds           = stats.upload_seconds;
    out.load.artifact_bytes_read      = stats.read_bytes;
    out.load.host_to_device_bytes     = stats.h2d_bytes;
    out.load.peak_staging_bytes       = stats.peak_staging_bytes;
    out.load.pinned_weight_bytes      = stats.pinned_bytes;
    out.load.device_object_count      = stats.device_object_count;
    out.load.host_object_count        = stats.host_object_count;
    const auto& config                = model->config();
    out.model_metadata.model_id       = out.load.model_name;
    out.model_metadata.vocab_size     = config.vocab_size;
    out.model_metadata.embedding_size = config.hidden_size;
    out.model_metadata.native_context = config.max_position_embeddings;
    std::set<std::string> tensor_formats;
    for (const auto& object : reader.directory().objects) {
        const auto* tensor = std::get_if<artifact::TensorObject>(&object);
        if (tensor == nullptr) { continue; }
        std::uint64_t elements = 1;
        for (const auto dimension : tensor->shape) { elements *= dimension; }
        out.model_metadata.parameters += elements;
        out.model_metadata.weight_bytes += tensor->bytes;
        tensor_formats.emplace(tensor->format);
    }
    for (const auto& format : tensor_formats) {
        if (!out.model_metadata.weights_id.empty()) { out.model_metadata.weights_id += "+"; }
        out.model_metadata.weights_id += format;
    }
    instance->model = std::move(model);
    out.instance    = std::move(instance);
    return out;
}

struct Qwen4ExpCore::Request {
    std::uint64_t id = 0;
    models::qwen3_5::PreparedPrompt prompt;
    std::vector<TokenId> prompt_tokens;
    models::qwen3_5::OutputSession output;
    PromptSummary prompt_summary;
    double prepare_seconds = 0.0;
    ResolvedRequestOptions options;
    OutputConsumerMode consumer_mode = OutputConsumerMode::Aggregate;
    GenerationObservationOptions observation;
    Clock::time_point deadline;
    Clock::time_point submitted;
    std::atomic<bool> cancelled{false};

    std::mutex mutex;
    std::condition_variable cv;
    std::optional<GenerationStart> stream_start;
    std::optional<PromptProgress> stream_progress;
    std::vector<std::variant<OutputDelta, GenerationTimingObservation>> events;
    bool response_done = false;
    std::exception_ptr error;
    GenerationResult result;
    std::string content, reasoning;
};

struct Qwen4ExpCore::Impl {
    Qwen4ExpInstance& instance;
    DeviceContext& device;
    const std::uint32_t max_context;
    const std::size_t max_outstanding;
    const std::chrono::milliseconds pending_timeout;
    const std::uint32_t domain;

    mutable std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::deque<std::shared_ptr<Request>> pending;
    std::size_t outstanding = 0;
    bool stopping           = false;
    bool failed             = false;
    std::uint64_t next_id   = 1;

    mutable std::mutex stats_mutex;
    RuntimeStats stats;

    // Head-device sampling planes.
    DeviceBuffer sample_config, sample_position, sample_out, token_counts, score_targets, score_out;
    std::unique_ptr<WorkspaceArena> sample_workspace;
    std::vector<__nv_bfloat16> host_logits;

    std::thread worker;

    Impl(Qwen4ExpInstance& i, DeviceContext& d, const EngineOptions& options)
        : instance(i), device(d), max_context(options.max_context),
          max_outstanding(std::size_t(options.max_concurrency) + options.max_pending_requests),
          pending_timeout(options.pending_timeout_ms),
          domain(i.model->resources().public_token_count) {
        RankBinding bind(device, i.executor->head_rank());
        sample_config    = DeviceBuffer(sizeof(ops::SamplingConfig));
        sample_position  = DeviceBuffer(sizeof(std::int32_t));
        sample_out       = DeviceBuffer(sizeof(std::int32_t));
        token_counts     = DeviceBuffer(std::size_t(domain) * sizeof(std::int32_t));
        score_targets    = DeviceBuffer(4096 * sizeof(std::int32_t));
        score_out        = DeviceBuffer(4096 * sizeof(float));
        sample_workspace = std::make_unique<WorkspaceArena>(std::max<std::size_t>(
            ops::sampling_workspace_capacity_bytes(static_cast<std::int32_t>(domain), 1, 1), 256));
        host_logits.resize(i.model->config().vocab_size);
        worker = std::thread([this] {
            device.bind_to_current_thread();
            loop();
        });
    }

    ~Impl() {
        {
            std::lock_guard lock(queue_mutex);
            stopping = true;
        }
        queue_cv.notify_all();
        if (worker.joinable()) { worker.join(); }
    }

    void complete(const std::shared_ptr<Request>& request, std::exception_ptr error) {
        {
            std::lock_guard lock(request->mutex);
            if (!request->response_done) {
                request->error         = std::move(error);
                request->response_done = true;
            }
        }
        request->cv.notify_all();
        std::lock_guard lock(queue_mutex);
        --outstanding;
    }

    void complete(const std::shared_ptr<Request>& request, GenerationResult result) {
        {
            std::lock_guard lock(request->mutex);
            if (!request->response_done) {
                request->result        = std::move(result);
                request->response_done = true;
            }
        }
        request->cv.notify_all();
        std::lock_guard lock(queue_mutex);
        --outstanding;
    }

    void loop() {
        for (;;) {
            std::shared_ptr<Request> request;
            {
                std::unique_lock lock(queue_mutex);
                queue_cv.wait(lock, [&] { return stopping || !pending.empty(); });
                if (stopping) {
                    auto waiting = std::move(pending);
                    pending.clear();
                    lock.unlock();
                    for (auto& item : waiting) {
                        complete(item, std::make_exception_ptr(
                                           RequestError(RequestErrorKind::Unavailable,
                                                        "inference engine is stopping")));
                    }
                    return;
                }
                request = std::move(pending.front());
                pending.pop_front();
            }
            publish_queue();
            if (Clock::now() > request->deadline) {
                complete(request, std::make_exception_ptr(
                                      RequestError(RequestErrorKind::QueueTimeout,
                                                   "inference request expired in the queue")));
                continue;
            }
            try {
                complete(request, run(*request));
            } catch (...) {
                const auto error = std::current_exception();
                if (!request_error(error)) {
                    std::lock_guard lock(queue_mutex);
                    failed   = true;
                    stopping = true;
                }
                complete(request, error);
                if (!request_error(error)) { queue_cv.notify_all(); }
            }
        }
    }

    void publish_queue() {
        std::lock_guard lock(queue_mutex);
        std::lock_guard stats_lock(stats_mutex);
        stats.waiting_requests = static_cast<std::uint32_t>(pending.size());
    }

    void push_events(Request& r, models::qwen3_5::PublishedOutput published,
                     std::optional<GenerationTimingObservation> timing) {
        const bool streaming = r.consumer_mode == OutputConsumerMode::Streaming;
        if (published.empty() && !timing) { return; }
        {
            std::lock_guard lock(r.mutex);
            if (streaming && timing) { r.events.emplace_back(*timing); }
            for (OutputDelta& delta : published) {
                (delta.channel == OutputChannel::Reasoning ? r.reasoning : r.content) += delta.text;
                if (streaming) { r.events.emplace_back(std::move(delta)); }
            }
        }
        if (streaming) { r.cv.notify_all(); }
    }

    // Samples one token from the head logits with `params`, at logical position `position`.
    TokenId sample(const ResolvedSamplingParameters& params, std::uint32_t position,
                   std::int32_t purpose, bool counts) {
        RankBinding bind(device, instance.executor->head_rank());
        const cudaStream_t stream = instance.executor->head_stream();
        ops::SamplingConfig config;
        config.temperature       = params.temperature;
        config.top_k             = params.top_k;
        config.top_p             = params.top_p;
        config.min_p             = params.min_p;
        config.presence_penalty  = params.presence_penalty;
        config.frequency_penalty = params.frequency_penalty;
        config.seed              = params.seed;
        config.token_counts      = counts ? static_cast<std::int32_t*>(token_counts.p) : nullptr;
        const auto at            = static_cast<std::int32_t>(position);
        CUDA_CHECK(cudaMemcpyAsync(sample_config.p, &config, sizeof(config), cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(
            cudaMemcpyAsync(sample_position.p, &at, sizeof(at), cudaMemcpyHostToDevice, stream));
        Tensor out(sample_out.p, DType::I32, {1});
        const Tensor positions(sample_position.p, DType::I32, {1});
        auto scope = sample_workspace->scope();
        ops::sample(instance.executor->logits(1), out, static_cast<std::int32_t>(domain),
                    static_cast<const ops::SamplingConfig*>(sample_config.p), positions, purpose,
                    *sample_workspace, stream);
        std::int32_t token = 0;
        CUDA_CHECK(
            cudaMemcpyAsync(&token, sample_out.p, sizeof(token), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (token == ops::kSamplerNonFiniteToken || token < 0 || std::uint32_t(token) >= domain) {
            throw std::runtime_error("Qwen3.8-Flash-Next produced non-finite logits");
        }
        return token;
    }

    FirstTokenLogprobs first_token_logprobs(TokenId selected, std::uint32_t count) {
        RankBinding bind(device, instance.executor->head_rank());
        const cudaStream_t stream = instance.executor->head_stream();
        CUDA_CHECK(cudaMemcpyAsync(host_logits.data(), instance.executor->logits(1).data,
                                   host_logits.size() * 2, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double maximum = -INFINITY;
        for (std::uint32_t v = 0; v < domain; ++v) {
            maximum = std::max(maximum, double(__bfloat162float(host_logits[v])));
        }
        double sum = 0.0;
        for (std::uint32_t v = 0; v < domain; ++v) {
            sum += std::exp(double(__bfloat162float(host_logits[v])) - maximum);
        }
        const double log_z = maximum + std::log(sum);
        const auto logprob = [&](TokenId v) {
            return static_cast<float>(double(__bfloat162float(host_logits[v])) - log_z);
        };
        std::vector<TokenId> order(domain);
        std::iota(order.begin(), order.end(), 0);
        const std::uint32_t top = std::min(count, domain);
        std::partial_sort(order.begin(), order.begin() + top, order.end(),
                          [&](TokenId a, TokenId b) {
                              const float la = __bfloat162float(host_logits[a]),
                                          lb = __bfloat162float(host_logits[b]);
                              return la != lb ? la > lb : a < b;
                          });
        FirstTokenLogprobs out;
        out.selected = {selected, logprob(selected)};
        for (std::uint32_t i = 0; i < top; ++i) {
            out.top.push_back({order[i], logprob(order[i])});
        }
        return out;
    }

    GenerationResult run(Request& r) {
        const auto admitted = Clock::now();
        auto& executor      = *instance.executor;
        const auto& tokens  = r.prompt_tokens;
        const auto prompt_n = static_cast<std::uint32_t>(tokens.size());
        if (prompt_n == 0) { throw std::invalid_argument("prepared prompt is empty"); }
        if (prompt_n > max_context) {
            throw RequestError(RequestErrorKind::ContextLengthExceeded,
                               "prepared prompt exceeds Engine max_context");
        }
        {
            std::lock_guard lock(stats_mutex);
            stats.running_requests    = 1;
            stats.prefilling_requests = 1;
        }
        if (r.consumer_mode == OutputConsumerMode::Streaming) {
            {
                std::lock_guard lock(r.mutex);
                r.stream_start =
                    GenerationStart{.prompt = r.prompt_summary, .reused_prompt_tokens = 0};
            }
            r.cv.notify_all();
        }
        executor.reset(0);
        const std::uint32_t chunk = executor.options().prefill_chunk;
        const auto prefill_start  = Clock::now();
        for (std::uint32_t at = 0; at < prompt_n; at += chunk) {
            if (r.cancelled.load(std::memory_order_acquire)) { break; }
            const std::uint32_t n = std::min(chunk, prompt_n - at);
            executor.forward(0, std::span<const TokenId>(tokens).subspan(at, n), 1);
            if (r.observation.prompt_progress) {
                CUDA_CHECK(cudaStreamSynchronize(executor.head_stream()));
                {
                    std::lock_guard lock(r.mutex);
                    r.stream_progress =
                        PromptProgress{.total_prompt_tokens     = prompt_n,
                                       .reused_prompt_tokens    = 0,
                                       .processed_prompt_tokens = at + n,
                                       .elapsed_ns = elapsed_ns(admitted, Clock::now())};
                }
                r.cv.notify_all();
            }
        }
        GenerationResult result;
        result.prompt = r.prompt_summary;
        std::vector<TokenId> generated;
        const auto finish = [&](FinishReason reason, Clock::time_point first_token,
                                Clock::time_point last_token, Clock::time_point prefill_end) {
            result.generated_token_ids = std::move(generated);
            {
                std::lock_guard lock(r.mutex);
                result.content   = std::move(r.content);
                result.reasoning = std::move(r.reasoning);
            }
            result.tool_calls              = r.output.take_tool_calls();
            result.tool_call_parse         = r.output.tool_call_parse_diagnostics();
            result.reasoning_tokens        = r.output.reasoning_tokens();
            result.finish_reason           = reason;
            result.matched_stop_string     = r.output.matched_stop_string();
            result.thinking                = r.output.thinking_stats();
            result.timings.prepare_seconds = r.prepare_seconds;
            result.timings.prefill_seconds = seconds(prefill_start, prefill_end);
            result.timings.decode_seconds  = seconds(prefill_end, last_token);
            result.timings.first_token_seconds =
                r.prepare_seconds + seconds(r.submitted, first_token);
            if (r.observation.phase_timings) {
                result.timings.prompt_wall_seconds     = seconds(admitted, first_token);
                result.timings.generation_wall_seconds = seconds(first_token, last_token);
            }
            result.timings.total_seconds = r.prepare_seconds + seconds(r.submitted, Clock::now());
            std::lock_guard lock(stats_mutex);
            stats.running_requests      = 0;
            stats.prefilling_requests   = 0;
            stats.decode_ready_requests = 0;
            return std::move(result);
        };
        if (r.cancelled.load(std::memory_order_acquire)) {
            (void)r.output.preview_terminal(FinishReason::Cancelled);
            push_events(r, r.output.commit_preview(), std::nullopt);
            const auto now = Clock::now();
            return finish(FinishReason::Cancelled, now, now, now);
        }
        CUDA_CHECK(cudaStreamSynchronize(executor.head_stream()));
        const auto prefill_end = Clock::now();
        {
            std::lock_guard lock(stats_mutex);
            stats.computed_prefill_tokens += prompt_n;
            stats.prefill_seconds_total += seconds(prefill_start, prefill_end);
            stats.prefilling_requests   = 0;
            stats.decode_ready_requests = 1;
        }
        const auto& exec = r.options.execution;
        const std::uint32_t capacity =
            effective_output_capacity(exec.requested_output_tokens, max_context, prompt_n);
        GenerationBudget budget(capacity, exec.requested_output_tokens <= capacity
                                              ? FinishReason::OutputLimit
                                              : FinishReason::ContextCapacity);
        const bool penalties = exec.sampling.presence_penalty != 0.0F ||
                               exec.sampling.frequency_penalty != 0.0F ||
                               (exec.post_thinking_sampling &&
                                (exec.post_thinking_sampling->presence_penalty != 0.0F ||
                                 exec.post_thinking_sampling->frequency_penalty != 0.0F));
        if (penalties) {
            RankBinding bind(device, executor.head_rank());
            CUDA_CHECK(
                cudaMemsetAsync(token_counts.p, 0, token_counts.bytes, executor.head_stream()));
        }
        bool post_thinking = false;
        Clock::time_point first_token{}, last_token = prefill_end;
        std::uint32_t position = prompt_n;
        std::int32_t purpose   = ops::kSamplePurposePrefill;
        for (;;) {
            if (!post_thinking && exec.post_thinking_sampling && r.output.reasoning_closed()) {
                post_thinking                          = true;
                result.thinking.post_thinking_sampling = true;
            }
            const ResolvedSamplingParameters& params =
                post_thinking ? *exec.post_thinking_sampling : exec.sampling;
            const TokenId token = sample(params, position, purpose, penalties);
            if (purpose == ops::kSamplePurposePrefill && exec.first_token_top_logprobs != 0) {
                result.first_token_logprobs =
                    first_token_logprobs(token, exec.first_token_top_logprobs);
            }
            purpose                       = ops::kSamplePurposeDecode;
            const OutputDecision decision = r.output.preview_model(
                std::span<const TokenId>(&token, 1), budget.remaining(), budget.limit_reason());
            const auto now = Clock::now();
            if (first_token == Clock::time_point{}) { first_token = now; }
            last_token = now;
            if (decision.accepted_tokens > 1) {
                throw std::logic_error("output policy accepted more than the sampled token");
            }
            if (decision.accepted_tokens == 1) {
                generated.push_back(token);
                budget.commit(1);
            }
            std::optional<GenerationTimingObservation> timing;
            if (r.observation.live_timings) {
                timing = GenerationTimingObservation{
                    .generated_tokens      = static_cast<std::uint32_t>(generated.size()),
                    .prompt_elapsed_ns     = elapsed_ns(admitted, first_token),
                    .generation_elapsed_ns = elapsed_ns(first_token, now)};
            }
            push_events(r, r.output.commit_preview(), timing);
            {
                std::lock_guard lock(stats_mutex);
                stats.committed_decode_tokens += decision.accepted_tokens;
            }
            if (decision.finished()) {
                return finish(decision.finish_reason, first_token, last_token, prefill_end);
            }
            std::vector<TokenId> feed{token};
            if (decision.continuation == ContinuationAction::ApplyTargetControl) {
                const auto pending_control = r.output.pending_control_tokens();
                const std::vector<TokenId> control(pending_control.begin(), pending_control.end());
                const OutputDecision forced = r.output.preview_control(control, budget.remaining());
                if (forced.accepted_tokens != control.size() || forced.finished()) {
                    throw std::logic_error("thinking control preview returned an invalid decision");
                }
                generated.insert(generated.end(), control.begin(), control.end());
                budget.commit(static_cast<std::uint32_t>(control.size()));
                push_events(r, r.output.commit_preview(), std::nullopt);
                feed.insert(feed.end(), control.begin(), control.end());
            }
            if (r.cancelled.load(std::memory_order_acquire)) {
                (void)r.output.preview_terminal(FinishReason::Cancelled);
                push_events(r, r.output.commit_preview(), std::nullopt);
                return finish(FinishReason::Cancelled, first_token, last_token, prefill_end);
            }
            if (position + feed.size() > max_context) {
                (void)r.output.preview_terminal(FinishReason::ContextCapacity);
                push_events(r, r.output.commit_preview(), std::nullopt);
                return finish(FinishReason::ContextCapacity, first_token, last_token, prefill_end);
            }
            const auto decode_start = Clock::now();
            executor.forward(0, feed, 1);
            position += static_cast<std::uint32_t>(feed.size());
            std::lock_guard lock(stats_mutex);
            stats.decode_rounds += 1;
            stats.decode_row_rounds += 1;
            stats.decode_seconds_total += seconds(decode_start, Clock::now());
        }
    }

    std::vector<float> score(const std::vector<TokenId>& tokens, std::uint32_t first_target) {
        auto& executor        = *instance.executor;
        const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
        if (n < 2 || first_target == 0 || first_target >= n || n > max_context) {
            throw std::invalid_argument("score: invalid token window");
        }
        // Logits for position p predict token p + 1; the executor keeps at most 512 logit rows.
        const std::uint32_t chunk = std::min<std::uint32_t>(executor.options().prefill_chunk, 512);
        executor.reset(0);
        std::vector<float> out;
        out.reserve(n - first_target);
        RankBinding bind(device, executor.head_rank());
        for (std::uint32_t at = 0; at + 1 < n; at += chunk) {
            const std::uint32_t count = std::min(chunk, n - 1 - at);
            executor.forward(0, std::span<const TokenId>(tokens).subspan(at, count), count);
            // Rows predicting targets [at + 1, at + count].
            const std::uint32_t first_row = first_target > at + 1 ? first_target - at - 1 : 0;
            if (first_row >= count) { continue; }
            const std::uint32_t rows = count - first_row;
            std::vector<std::int32_t> targets(rows);
            for (std::uint32_t i = 0; i < rows; ++i) {
                targets[i] = tokens[at + 1 + first_row + i];
            }
            const cudaStream_t stream = executor.head_stream();
            CUDA_CHECK(cudaMemcpyAsync(score_targets.p, targets.data(), rows * 4,
                                       cudaMemcpyHostToDevice, stream));
            const Tensor all = executor.logits(count);
            const Tensor logits(static_cast<__nv_bfloat16*>(all.data) +
                                    std::size_t(first_row) * all.ne[0],
                                DType::BF16, {all.ne[0], static_cast<std::int32_t>(rows)});
            const Tensor target_ids(score_targets.p, DType::I32, {static_cast<std::int32_t>(rows)});
            Tensor result(score_out.p, DType::FP32, {static_cast<std::int32_t>(rows)});
            ops::target_logprobs(logits, target_ids, static_cast<std::int32_t>(domain), result,
                                 stream);
            std::vector<float> host(rows);
            CUDA_CHECK(cudaMemcpyAsync(host.data(), score_out.p, rows * 4, cudaMemcpyDeviceToHost,
                                       stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            out.insert(out.end(), host.begin(), host.end());
        }
        return out;
    }
};

Qwen4ExpCore::Submission::Submission(Qwen4ExpCore& owner, std::shared_ptr<Request> request,
                                     std::optional<std::uint32_t> budget) noexcept
    : owner_(&owner), request_(std::move(request)), effective_thinking_budget_(budget) {}

Qwen4ExpCore::Submission::~Submission() {
    if (request_ != nullptr) { request_->cancelled.store(true, std::memory_order_release); }
}

Qwen4ExpCore::Submission::Submission(Submission&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)),
      effective_thinking_budget_(other.effective_thinking_budget_) {}

Qwen4ExpCore::Submission& Qwen4ExpCore::Submission::operator=(Submission&& other) noexcept {
    if (this != &other) {
        if (request_ != nullptr) { request_->cancelled.store(true, std::memory_order_release); }
        owner_                     = std::exchange(other.owner_, nullptr);
        request_                   = std::move(other.request_);
        effective_thinking_budget_ = other.effective_thinking_budget_;
    }
    return *this;
}

GenerationResult Qwen4ExpCore::Submission::wait(OutputSink* sink,
                                                const CancellationView& cancellation) {
    if (request_ == nullptr) { throw std::logic_error("submission is empty"); }
    const std::shared_ptr<Request> request = std::move(request_);
    const bool streaming = request->consumer_mode == OutputConsumerMode::Streaming;
    if (streaming != (sink != nullptr)) {
        request->cancelled.store(true, std::memory_order_release);
        throw std::invalid_argument(
            "GenerationHandle wait sink does not match its submitted consumer mode");
    }
    std::exception_ptr caller_error;
    for (;;) {
        std::optional<GenerationStart> start;
        std::optional<PromptProgress> progress;
        std::vector<std::variant<OutputDelta, GenerationTimingObservation>> events;
        bool done = false;
        {
            std::unique_lock lock(request->mutex);
            request->cv.wait_for(lock, std::chrono::milliseconds(10), [&] {
                return request->response_done || request->stream_start.has_value() ||
                       request->stream_progress.has_value() || !request->events.empty();
            });
            start    = std::exchange(request->stream_start, std::nullopt);
            progress = std::exchange(request->stream_progress, std::nullopt);
            events.swap(request->events);
            done = request->response_done;
        }
        if (caller_error == nullptr && sink != nullptr) {
            try {
                if (start) { sink->start(std::move(*start)); }
                if (progress) { sink->progress(std::move(*progress)); }
                for (auto& event : events) {
                    if (auto* timing = std::get_if<GenerationTimingObservation>(&event)) {
                        sink->timing(*timing);
                    } else {
                        sink->publish(std::move(std::get<OutputDelta>(event)));
                    }
                }
            } catch (...) {
                caller_error = std::current_exception();
                request->cancelled.store(true, std::memory_order_release);
            }
        }
        if (caller_error == nullptr) {
            try {
                if (cancellation.requested()) {
                    request->cancelled.store(true, std::memory_order_release);
                }
            } catch (...) {
                caller_error = std::current_exception();
                request->cancelled.store(true, std::memory_order_release);
            }
        }
        if (!done) { continue; }
        if (caller_error != nullptr) { std::rethrow_exception(caller_error); }
        std::lock_guard lock(request->mutex);
        if (request->error != nullptr) { std::rethrow_exception(request->error); }
        return std::move(request->result);
    }
}

Qwen4ExpCore::Qwen4ExpCore(Qwen4ExpInstance& instance, DeviceContext& device,
                           const EngineOptions& options)
    : impl_(std::make_unique<Impl>(instance, device, options)) {
    if (options.max_concurrency == 0 || options.max_pending_requests == 0 ||
        options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine core bounds are invalid");
    }
}

Qwen4ExpCore::~Qwen4ExpCore() = default;

void Qwen4ExpCore::stop() noexcept {
    {
        std::lock_guard lock(impl_->queue_mutex);
        impl_->stopping = true;
    }
    impl_->queue_cv.notify_all();
}

Qwen4ExpCore::Submission Qwen4ExpCore::submit(models::qwen3_5::PreparedPrompt prompt,
                                              PromptSummary prompt_summary, double prepare_seconds,
                                              ResolvedRequestOptions options,
                                              OutputConsumerMode consumer_mode,
                                              GenerationObservationOptions observation,
                                              Clock::time_point pending_deadline) {
    const auto submitted = Clock::now();
    if (options.execution.structured_output.kind != StructuredOutputKind::None) {
        throw std::invalid_argument(
            "structured output is not available for Qwen3.8-Flash-Next yet");
    }
    if (pending_deadline == Clock::time_point{}) {
        pending_deadline = submitted + impl_->pending_timeout;
    }
    auto request = std::make_shared<Request>();
    request->prompt_tokens =
        std::vector<TokenId>(prompt.token_ids().begin(), prompt.token_ids().end());
    request->prompt_summary  = prompt_summary;
    request->prepare_seconds = prepare_seconds;
    request->consumer_mode   = consumer_mode;
    request->observation     = observation;
    request->deadline        = pending_deadline;
    request->submitted       = submitted;
    apply_effective_thinking_budget(
        options.execution.thinking,
        effective_output_capacity(options.execution.requested_output_tokens, impl_->max_context,
                                  prompt_summary.prompt_tokens),
        impl_->instance.frontend.thinking_control_token_count());
    request->output = impl_->instance.frontend.make_output_session(
        prompt, options.stop, options.output, options.execution.thinking,
        options.execution.structured_output);
    const std::optional<std::uint32_t> budget = request->output.thinking_stats().effective_budget;
    request->prompt                           = std::move(prompt);
    request->options                          = std::move(options);
    {
        std::lock_guard lock(impl_->queue_mutex);
        if (impl_->stopping || impl_->failed) {
            throw RequestError(RequestErrorKind::Unavailable, "inference engine is unavailable");
        }
        if (impl_->outstanding >= impl_->max_outstanding) {
            throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
        }
        ++impl_->outstanding;
        request->id = impl_->next_id++;
        impl_->pending.push_back(request);
    }
    impl_->queue_cv.notify_one();
    impl_->publish_queue();
    return Submission(*this, std::move(request), budget);
}

std::vector<float> Qwen4ExpCore::score(models::qwen3_5::PreparedPrompt prompt,
                                       std::uint32_t first_target) {
    const std::vector<TokenId> tokens(prompt.token_ids().begin(), prompt.token_ids().end());
    {
        std::lock_guard lock(impl_->queue_mutex);
        if (impl_->stopping || impl_->failed) {
            throw RequestError(RequestErrorKind::Unavailable, "inference engine is unavailable");
        }
        if (impl_->outstanding != 0) {
            throw std::logic_error("score requires an idle Qwen3.8-Flash-Next Engine");
        }
        ++impl_->outstanding;
    }

    struct Release {
        Impl& impl;

        ~Release() {
            std::lock_guard lock(impl.queue_mutex);
            --impl.outstanding;
        }
    } release{*impl_};

    impl_->device.bind_to_current_thread();
    return impl_->score(tokens, first_target);
}

MemorySummary Qwen4ExpCore::memory_summary() const {
    MemorySummary out;
    out.device                    = impl_->device.rank(0).device;
    out.max_context               = impl_->max_context;
    out.kv_capacity               = impl_->max_context;
    const auto& stats             = impl_->instance.model->storage_stats();
    out.weights.capacity_bytes    = stats.device_capacity_bytes;
    out.weights.used_bytes        = stats.device_capacity_bytes;
    out.sequence.capacity_bytes   = impl_->instance.executor->memory().state_bytes;
    out.sequence.used_bytes       = out.sequence.capacity_bytes;
    out.workspace.capacity_bytes  = impl_->instance.executor->memory().workspace_bytes;
    out.workspace.used_bytes      = out.workspace.capacity_bytes;
    out.runtime_reservation_bytes = out.sequence.capacity_bytes + out.workspace.capacity_bytes;
    return out;
}

RuntimeStats Qwen4ExpCore::runtime_stats() const {
    std::lock_guard lock(impl_->stats_mutex);
    return impl_->stats;
}

bool Qwen4ExpCore::is_available() const {
    std::lock_guard lock(impl_->queue_mutex);
    return !impl_->stopping && !impl_->failed;
}

bool Qwen4ExpCore::has_failed() const {
    std::lock_guard lock(impl_->queue_mutex);
    return impl_->failed;
}

} // namespace ninfer::runtime
