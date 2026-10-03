#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * The routed and shared experts of Qwen3.8-Flash-Next's MoE over BF16 weights (the debug and
 * oracle representation), after moe_route:
 *
 *   e_j(x) = down_j . (silu(gate_j . x) * (up_j . x))          expert width 640
 *   y      = sum_{k<10} weights[k] e_{ids[k]}(m) + shared * e_shared(m)
 *
 * `m` BF16 [2560, tokens]; `ids` I32 [10, tokens], `weights` FP32 [10, tokens], `shared` FP32
 * [tokens] from moe_route; `gate_up` BF16 [2560, 1280, 512] (each expert's 640 gate rows then 640
 * up rows), `down` BF16 [640, 2560, 512], `shared_gate_up` BF16 [2560, 1280], `shared_down` BF16
 * [640, 2560]; `y` FP32 [2560, tokens]. The oracle evaluates in FP64 from the represented inputs;
 * y is compared as FP32 by relative L2 and gross error.
 */
[[nodiscard]] std::size_t moe_experts_bf16_workspace_bytes(std::int32_t tokens);

void moe_experts_bf16(const Tensor& m, const Tensor& ids, const Tensor& weights,
                      const Tensor& shared, const Tensor& gate_up, const Tensor& down,
                      const Tensor& shared_gate_up, const Tensor& shared_down,
                      WorkspaceArena& workspace, Tensor& y, cudaStream_t stream);

} // namespace ninfer::ops
