#pragma once

// The forward pass of Qwen3.8-Flash-Next over a loaded Model: the four-stream residual stack in
// FP32, Gated DeltaNet and sparse-attention mixers, the PLE n-gram injection and the 512-expert
// MoE, layer by layer on each layer's stage device, the stack crossing to the next device at a
// stage boundary. The executor owns every sequence's mutable state (recurrent and convolution
// states, paged KV, the indexer's pooled keys, the PLE history and n-gram context) and the
// workspace; the model stays immutable.

#include "core/device.h"
#include "core/tensor.h"
#include "models/qwen4_exp/expert_cache.h"
#include "models/qwen4_exp/model.h"
#include "models/qwen4_exp/ngram_companion.h"

#include <cstdint>
#include <memory>
#include <span>

namespace ninfer::models::qwen4_exp {

struct ExecutorOptions {
    std::uint32_t max_context = 32768;
    // Tokens per forward call: a prompt is fed in calls of at most this many.
    std::uint32_t prefill_chunk = 2048;
    std::uint32_t sequences     = 1;
    NgramCompanion ngram;
    NgramResidency ngram_residency = NgramResidency::Disk;
    // Host-resident experts only: device memory lent to the expert cache, split evenly over the
    // ranks; kAutoExpertCache takes what each device has free less a margin, 0 none.
    static constexpr std::uint64_t kAutoExpertCache = ~std::uint64_t{0};
    std::uint64_t expert_cache_bytes                = kAutoExpertCache;
    // Decode steps (one token) of device- and host-resident experts replay a CUDA graph per
    // segment of consecutive layers on one device.
    bool cuda_graphs = true;
};

struct ExecutorMemory {
    std::uint64_t state_bytes        = 0; // per-sequence state on every device
    std::uint64_t workspace_bytes    = 0;
    std::uint64_t expert_cache_bytes = 0;
};

class Executor {
public:
    Executor(const Model& model, DeviceContext& device, ExecutorOptions options);
    ~Executor();
    Executor(const Executor&)            = delete;
    Executor& operator=(const Executor&) = delete;

    [[nodiscard]] const ExecutorOptions& options() const noexcept;
    [[nodiscard]] ExecutorMemory memory() const noexcept;

    // Empties the sequence: position zero, zero states, a fresh n-gram context.
    void reset(std::uint32_t sequence);
    [[nodiscard]] std::uint32_t position(std::uint32_t sequence) const;

    // Runs `tokens` (at most prefill_chunk) at the sequence's next positions and leaves the logits
    // of the last `logit_rows` of them in logits(), BF16 [vocab, logit_rows] on the head device.
    // Work is queued on the device streams; logits() is ready on head_stream().
    void forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                 std::uint32_t logit_rows);
    [[nodiscard]] Tensor logits(std::uint32_t rows) const;
    [[nodiscard]] std::size_t head_rank() const noexcept;
    [[nodiscard]] ExpertCacheStats expert_cache_stats() const noexcept;
    [[nodiscard]] cudaStream_t head_stream() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen4_exp
