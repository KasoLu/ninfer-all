#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/weight_input.h"

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

/**
 * The same experts over GGUF block banks (an imported GSQ-RCO release), reached through tables of
 * expert base pointers, so an expert may sit in a device bank, a cache slot or mapped host memory.
 * `gate` and `up` hold each expert's 640 rows of 2560 values, `down` its 2560 rows of 640; the
 * routed tables have `experts` entries (up to 512), the shared expert's tables one. Every entry
 * points at device-readable memory in the table's block type, rows `row_bytes` apart.
 *
 * The products quantize their activation to ggml's q8_1 as llama.cpp does (the format's
 * arithmetic): m once per token, the middle silu(gate . m) * (up . m) after its BF16 store. Each
 * expert's contribution weights[k] * e (shared * e_shared for the shared expert) is accumulated in
 * 2^-32 fixed point, so y does not depend on the order the experts run in, and stored as FP32.
 * The oracle independently decodes stored blocks and evaluates the complete expert mathematics
 * in FP64 from the public BF16 input, without the implementation's internal q8_1 quantization or
 * BF16 middle cast; y is compared as FP32 by relative L2 and gross error. Q2_0 has an independent
 * CPU decoder in its qualification; other formats' existing fixtures still use the GPU decoder.
 *
 * Up to eight tokens run ggml-style vector products. Wider calls over `device_resident` banks run
 * all three projections through ggml's integer tensor-core matrix kernel (with the same q8_1
 * arithmetic): the tables then point at device memory only, and at least 256 zero bytes follow the
 * last expert of every down bank (and of a slot pool), since the kernel reads a 640-value row in
 * three 256-value steps. Banks in mapped host memory stay on the vector products, which read them
 * across the bus better.
 */
struct GgufExpertTable {
    QType format               = QType::GGUF_Q8_0;
    const void* const* experts = nullptr; // device array of expert base pointers
    std::int64_t row_bytes     = 0;
};

struct GgufMoeWeights {
    GgufExpertTable gate, up, down;
    GgufExpertTable shared_gate, shared_up, shared_down;
    // Entries of the routed tables: 512, or the 256 an expert-pruned release keeps.
    std::int32_t experts = 512;
    // Every entry in device memory, zeros past each down bank (see above).
    bool device_resident = false;
};

[[nodiscard]] std::size_t moe_experts_gguf_workspace_bytes(std::int32_t tokens);
void moe_experts_gguf(const Tensor& m, const Tensor& ids, const Tensor& weights,
                      const Tensor& shared, const GgufMoeWeights& banks, WorkspaceArena& workspace,
                      Tensor& y, cudaStream_t stream);

/**
 * One native expert projection bank. `experts` is a caller-owned device array of prepared
 * Weight operands, all in `format`: BF16 or Q2/Q4/Q5/Q6/Q8 row-split weights. The operands
 * retain their actual code/high/scale planes and layout; no selected-weight repack is needed.
 * `integer_a8` selects private group-32 activation quantization and signed-byte dot products;
 * BF16 banks take A16 only. Preparation validates the represented shapes and storage.
 */
struct NativeExpertTable {
    QType format = QType::BF16;
    const Weight* experts = nullptr;
    bool integer_a8 = false;
};

struct NativeMoeWeights {
    NativeExpertTable gate, up, down;
    NativeExpertTable shared_gate, shared_up, shared_down;
    std::int32_t experts = 512;
};

[[nodiscard]] Weight prepare_native_expert(const WeightInput& input, std::int32_t rows,
                                           std::int32_t columns, bool integer_a8);
[[nodiscard]] std::size_t moe_experts_native_workspace_bytes(std::int32_t tokens);

/**
 * Residual-free Flash-Next experts with the same represented inputs, top-10 routing and FP32
 * output as moe_experts_bf16. Routed banks have `experts` entries (10..512), shared banks one;
 * gate/up have 640 rows of 2560 values, down 2560 rows of 640. Each projection keeps its own
 * format and compute profile. Each call supports 1..65535 tokens and enqueues without host synchronization.
 *
 * The independent oracle decodes the original stored words and evaluates complete SwiGLU and
 * weighted merge in FP64 from public BF16 m. Middle products, activation quantization and
 * reduction order are private arithmetic, not oracle boundaries. Repeats with fixed operands
 * and profile are bit-exact. Optional `parts` is U8 [10,tokens]: zero excludes that routed
 * contribution, nonzero evaluates it; null includes every route. The shared expert always
 * contributes. Excluded experts' weight operands are never read. A caller may compute those
 * experts on the CPU and add their weighted sum after both parts finish.
 * Caller-owned workspace and operand tables are graph-stable;
 * live inputs, weights, output and workspace do not overlap.
 */
void moe_experts_native(const Tensor& m, const Tensor& ids, const Tensor& weights,
                        const Tensor& shared, const NativeMoeWeights& banks,
                        const Tensor* parts, WorkspaceArena& workspace, Tensor& y, cudaStream_t stream);

} // namespace ninfer::ops
