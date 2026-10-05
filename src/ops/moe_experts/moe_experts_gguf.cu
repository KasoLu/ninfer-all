// ninfer::ops - Qwen3.8-Flash-Next MoE experts over GGUF block banks (contract in
// include/ninfer/ops/moe_experts.h). The router's pairs are grouped by expert on the device; the
// gate/up products then run once per active expert over all of its tokens (the vector kernel's MoE
// form, ops/linear/gguf), the down products accumulate weighted into a fixed-point plane, and the
// shared expert takes the same path as a one-expert bank that every token selects.
#include "ninfer/ops/moe_experts.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kHidden  = 2560;
constexpr int kWidth   = 640;
constexpr int kExperts = 512;
constexpr int kTopK    = 10;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("moe_experts_gguf: ") + message); }
}

bool shaped(const Tensor& tensor, DType dtype, std::int32_t n0, std::int32_t n1) {
    return tensor.dtype == dtype && tensor.is_contiguous() && tensor.data != nullptr &&
           tensor.ne[0] == n0 && tensor.ne[1] == n1 && tensor.ne[2] == 1 && tensor.ne[3] == 1;
}

// How many of one expert's columns share a pass over its rows, from the average it gets.
int chunk_for(std::int64_t pairs, std::int64_t experts) {
    const std::int64_t average = (pairs + experts - 1) / experts;
    return average >= 6 ? 8 : average >= 3 ? 4 : average >= 2 ? 2 : 1;
}

gguf::MoeTable table(const GgufExpertTable& bank, int rows, int k) {
    require(bank.experts != nullptr && bank.row_bytes > 0, "an expert table is empty");
    return gguf::MoeTable{bank.experts, bank.row_bytes, rows, k};
}

bool fusable(const GgufExpertTable& gate, const GgufExpertTable& up) {
    return gate.format == up.format && gate.row_bytes == up.row_bytes;
}

__global__ void store_fixed_kernel(const unsigned long long* __restrict__ fixed,
                                   float* __restrict__ y, std::int64_t count) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    y[i] = static_cast<float>(static_cast<double>(static_cast<long long>(fixed[i])) *
                              (1.0 / 4294967296.0));
}

struct Plan {
    std::int32_t tokens = 0;
    std::int32_t pairs  = 0;
};

// One bank's pass: the middle of every pair, then its weighted down product into `fixed`.
void run_bank(const Tensor& routing_ids, std::int32_t pairs, std::int32_t experts, int per_token,
              const float* pair_weights, const void* activation, std::int32_t tokens,
              const GgufExpertTable& gate, const GgufExpertTable& up, const GgufExpertTable& down,
              WorkspaceArena& workspace, unsigned long long* fixed, cudaStream_t stream) {
    auto scope           = workspace.scope();
    Tensor routing_bytes = workspace.alloc(
        DType::U8, {static_cast<std::int32_t>(gguf::moe_routing_bytes(experts, pairs))});
    const gguf::MoeRouting routing =
        gguf::moe_sort_routes(static_cast<const std::int32_t*>(routing_ids.data), pairs, experts,
                              routing_bytes.data, stream);
    const int active                = std::min(pairs, experts);
    const int chunk                 = chunk_for(pairs, active);
    Tensor middle                   = workspace.alloc(DType::BF16, {kWidth, pairs});
    auto* middle_p                  = static_cast<__nv_bfloat16*>(middle.data);
    const gguf::MoeTable gate_table = table(gate, kWidth, kHidden);
    const gguf::MoeTable up_table   = table(up, kWidth, kHidden);
    if (fusable(gate, up)) {
        gguf::moe_vector_swiglu(detail::gguf_type(gate.format), gate_table, up_table, routing,
                                active, per_token, activation, tokens, middle_p, chunk, stream);
    } else {
        Tensor gate_plane = workspace.alloc(DType::FP32, {kWidth, pairs});
        auto* gate_p      = static_cast<float*>(gate_plane.data);
        gguf::moe_vector_product(detail::gguf_type(gate.format), gate_table, routing, active,
                                 per_token, per_token, activation, tokens,
                                 gguf::MoeOutput{.f32 = gate_p}, chunk, stream);
        gguf::moe_vector_product(detail::gguf_type(up.format), up_table, routing, active, per_token,
                                 per_token, activation, tokens,
                                 gguf::MoeOutput{.bf16 = middle_p, .gate = gate_p}, chunk, stream);
    }
    Tensor middle_q = workspace.alloc(
        DType::U8, {static_cast<std::int32_t>(gguf::vector_activation_bytes(kWidth, pairs))});
    gguf::quantize_vector_activation(middle_p, kWidth, pairs, nullptr, middle_q.data, stream);
    gguf::moe_vector_product(detail::gguf_type(down.format), table(down, kHidden, kWidth), routing,
                             active, 1, per_token, middle_q.data, pairs,
                             gguf::MoeOutput{.weighted = fixed, .weights = pair_weights}, chunk,
                             stream);
}

} // namespace

std::size_t moe_experts_gguf_workspace_bytes(std::int32_t tokens) {
    require(tokens > 0, "tokens must be positive");
    const std::int32_t pairs = tokens * kTopK;
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::U8,
                       {static_cast<std::int32_t>(gguf::vector_activation_bytes(kHidden, tokens))});
    (void)layout.alloc(DType::I64, {kHidden, tokens});
    (void)layout.alloc(DType::I32, {tokens});
    // The larger of the two banks' passes, which run one after the other in the same scope.
    (void)layout.alloc(DType::U8,
                       {static_cast<std::int32_t>(gguf::moe_routing_bytes(kExperts, pairs))});
    (void)layout.alloc(DType::BF16, {kWidth, pairs});
    (void)layout.alloc(DType::FP32, {kWidth, pairs});
    (void)layout.alloc(DType::U8,
                       {static_cast<std::int32_t>(gguf::vector_activation_bytes(kWidth, pairs))});
    return layout.peak_bytes(1);
}

void moe_experts_gguf(const Tensor& m, const Tensor& ids, const Tensor& weights,
                      const Tensor& shared, const GgufMoeWeights& banks, WorkspaceArena& workspace,
                      Tensor& y, cudaStream_t stream) {
    const std::int32_t tokens = m.ne[1];
    require(tokens > 0 && shaped(m, DType::BF16, kHidden, tokens), "m must be BF16 [2560, tokens]");
    require(tokens * kTopK <= 65535, "at most 6553 tokens per call");
    require(shaped(ids, DType::I32, kTopK, tokens), "ids must be I32 [10, tokens]");
    require(shaped(weights, DType::FP32, kTopK, tokens), "weights must be FP32 [10, tokens]");
    require(shared.dtype == DType::FP32 && shared.is_contiguous() && shared.numel() == tokens,
            "shared must be FP32 [tokens]");
    require(shaped(y, DType::FP32, kHidden, tokens), "y must be FP32 [2560, tokens]");
    auto scope        = workspace.scope();
    Tensor activation = workspace.alloc(
        DType::U8, {static_cast<std::int32_t>(gguf::vector_activation_bytes(kHidden, tokens))});
    gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(m.data), kHidden, tokens,
                                     nullptr, activation.data, stream);
    Tensor fixed = workspace.alloc(DType::I64, {kHidden, tokens});
    CUDA_CHECK(cudaMemsetAsync(fixed.data, 0, fixed.bytes(), stream));
    auto* fixed_p = static_cast<unsigned long long*>(fixed.data);
    // Every token selects the shared expert (index 0 of its one-entry bank).
    Tensor shared_ids = workspace.alloc(DType::I32, {tokens});
    CUDA_CHECK(cudaMemsetAsync(shared_ids.data, 0, shared_ids.bytes(), stream));
    run_bank(ids, tokens * kTopK, kExperts, kTopK, static_cast<const float*>(weights.data),
             activation.data, tokens, banks.gate, banks.up, banks.down, workspace, fixed_p, stream);
    run_bank(shared_ids, tokens, 1, 1, static_cast<const float*>(shared.data), activation.data,
             tokens, banks.shared_gate, banks.shared_up, banks.shared_down, workspace, fixed_p,
             stream);
    const std::int64_t count = std::int64_t(kHidden) * tokens;
    store_fixed_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        fixed_p, static_cast<float*>(y.data), count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
