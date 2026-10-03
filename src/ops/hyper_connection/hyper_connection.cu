// ninfer::ops - hyper-connection read/write of Qwen3.8-Flash-Next (contract in
// include/ninfer/ops/hyper_connection.h). Four launches: per-stream RMSNorm into FP32 workspace,
// the down and inject GEMVs with their activations, the up GEMV with the gated stream mix, and the
// weighted write. Every product accumulates in FP32; the weights stream once per column chunk.
#include "ninfer/ops/hyper_connection.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/common/math.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kStreams = 4;
constexpr std::int32_t kHidden  = 2560;
constexpr std::int32_t kLowrank = 320;
constexpr std::int32_t kWidth   = kStreams * kHidden;
// Columns one warp accumulates at once; wider calls loop over column chunks.
constexpr int kColumnChunk = 8;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + __expf(-x)); }

// One block per (stream, token): xn = x * rsqrt(mean x^2 + eps) * (1 + g).
__global__ void __launch_bounds__(256)
    hc_norm_kernel(const float* __restrict__ stack, const __nv_bfloat16* __restrict__ norm,
                   float eps, float* __restrict__ normalized) {
    const int c                 = blockIdx.x;
    const int t                 = blockIdx.y;
    const std::int64_t base     = (static_cast<std::int64_t>(t) * kStreams + c) * kHidden;
    __shared__ float partial[8];
    float sum = 0.0f;
    for (int d = threadIdx.x; d < kHidden; d += blockDim.x) {
        const float x = stack[base + d];
        sum += x * x;
    }
    sum = warp_sum(sum);
    if ((threadIdx.x & 31) == 0) { partial[threadIdx.x >> 5] = sum; }
    __syncthreads();
    if (threadIdx.x < 32) {
        float total = threadIdx.x < blockDim.x / 32 ? partial[threadIdx.x] : 0.0f;
        total       = warp_sum(total);
        if (threadIdx.x == 0) { partial[0] = total; }
    }
    __syncthreads();
    const float scale = rsqrtf(partial[0] / kHidden + eps);
    for (int d = threadIdx.x; d < kHidden; d += blockDim.x) {
        normalized[base + d] =
            stack[base + d] * scale * (1.0f + __bfloat162float(norm[c * kHidden + d]));
    }
}

// One warp per output row of [down; inject] (lowrank + streams rows over the 10240-wide input).
// Rows below lowrank write silu(v / n) to `low`; inject rows write 2 sigmoid(v / n).
__global__ void __launch_bounds__(256)
    hc_down_kernel(const float* __restrict__ normalized, const __nv_bfloat16* __restrict__ down,
                   const __nv_bfloat16* __restrict__ inject, int tokens, int rows,
                   float* __restrict__ low, float* __restrict__ inject_weights) {
    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rows) { return; }
    const __nv_bfloat16* weights = row < kLowrank
                                       ? down + static_cast<std::int64_t>(row) * kWidth
                                       : inject + static_cast<std::int64_t>(row - kLowrank) * kWidth;
    for (int t0 = 0; t0 < tokens; t0 += kColumnChunk) {
        const int count = min(kColumnChunk, tokens - t0);
        float acc[kColumnChunk] = {};
        for (int k = lane * 2; k < kWidth; k += 64) {
            const float2 w = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(weights + k));
#pragma unroll
            for (int j = 0; j < kColumnChunk; ++j) {
                if (j < count) {
                    const float2 x = *reinterpret_cast<const float2*>(
                        normalized + static_cast<std::int64_t>(t0 + j) * kWidth + k);
                    acc[j] = fmaf(w.x, x.x, fmaf(w.y, x.y, acc[j]));
                }
            }
        }
#pragma unroll
        for (int j = 0; j < kColumnChunk; ++j) {
            const float v = warp_sum(acc[j]) / kStreams;
            if (lane == 0 && j < count) {
                const int t = t0 + j;
                if (row < kLowrank) {
                    low[static_cast<std::int64_t>(t) * kLowrank + row] = v * sigmoid_f(v);
                } else {
                    inject_weights[static_cast<std::int64_t>(t) * kStreams + row - kLowrank] =
                        2.0f * sigmoid_f(v);
                }
            }
        }
    }
}

// One warp per hidden index d: the four gate rows c * hidden + d of `up` (each lowrank wide)
// against lo, then mixed[d] = (1/n) sum_c sigmoid(gate_c) xn[c, d].
__global__ void __launch_bounds__(256)
    hc_up_mix_kernel(const float* __restrict__ normalized, const float* __restrict__ low,
                     const __nv_bfloat16* __restrict__ up, int tokens,
                     __nv_bfloat16* __restrict__ mixed) {
    const int d    = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (d >= kHidden) { return; }
    for (int t0 = 0; t0 < tokens; t0 += kColumnChunk) {
        const int count = min(kColumnChunk, tokens - t0);
        float out[kColumnChunk] = {};
#pragma unroll
        for (int c = 0; c < kStreams; ++c) {
            const __nv_bfloat16* weights =
                up + static_cast<std::int64_t>(c * kHidden + d) * kLowrank;
            float acc[kColumnChunk] = {};
            for (int k = lane * 2; k < kLowrank; k += 64) {
                const float2 w =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(weights + k));
#pragma unroll
                for (int j = 0; j < kColumnChunk; ++j) {
                    if (j < count) {
                        const float2 x = *reinterpret_cast<const float2*>(
                            low + static_cast<std::int64_t>(t0 + j) * kLowrank + k);
                        acc[j] = fmaf(w.x, x.x, fmaf(w.y, x.y, acc[j]));
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < kColumnChunk; ++j) {
                const float gate = sigmoid_f(warp_sum(acc[j]));
                if (j < count) {
                    out[j] += gate * normalized[static_cast<std::int64_t>(t0 + j) * kWidth +
                                                c * kHidden + d];
                }
            }
        }
        if (lane == 0) {
            for (int j = 0; j < count; ++j) {
                mixed[static_cast<std::int64_t>(t0 + j) * kHidden + d] =
                    __float2bfloat16_rn(out[j] / kStreams);
            }
        }
    }
}

__device__ __forceinline__ float to_float(float value) { return value; }
__device__ __forceinline__ float to_float(__nv_bfloat16 value) { return __bfloat162float(value); }

template <typename Output>
__global__ void __launch_bounds__(256)
    hc_write_kernel(float* __restrict__ stack, const Output* __restrict__ y,
                    const float* __restrict__ inject_weights, std::int64_t elements) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) { return; }
    const std::int64_t d      = i % kHidden;
    const std::int64_t column = i / kHidden; // t * streams + c
    const std::int64_t t      = column / kStreams;
    stack[i] = fmaf(to_float(y[t * kHidden + d]), inject_weights[column], stack[i]);
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("hyper_connection: ") + message); }
}

void require_bf16(const Tensor* tensor, std::int32_t n0, std::int32_t n1, const char* name) {
    require(tensor != nullptr && tensor->data != nullptr, name);
    require(tensor->dtype == DType::BF16 && tensor->is_contiguous() && tensor->ne[0] == n0 &&
                tensor->ne[1] == n1 && tensor->ne[2] == 1 && tensor->ne[3] == 1,
            name);
}

void require_geometry(std::int32_t streams, std::int32_t hidden, std::int32_t lowrank) {
    require(streams == kStreams && hidden == kHidden && lowrank == kLowrank,
            "unsupported geometry (streams 4, hidden 2560, lowrank 320 are implemented)");
}

} // namespace

std::size_t hyper_connection_read_workspace_bytes(std::int32_t streams, std::int32_t hidden,
                                                  std::int32_t lowrank, std::int32_t tokens) {
    require_geometry(streams, hidden, lowrank);
    require(tokens > 0, "tokens must be positive");
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP32, {kWidth, tokens});
    (void)layout.alloc(DType::FP32, {kLowrank, tokens});
    (void)layout.alloc(DType::FP32, {kStreams, tokens});
    return layout.peak_bytes(1);
}

void hyper_connection_read(const Tensor& stack, const HyperConnectionWeights& weights, float eps,
                           WorkspaceArena& workspace, Tensor& mixed, Tensor* inject_weights,
                           cudaStream_t stream) {
    require(stack.dtype == DType::FP32 && stack.is_contiguous() && stack.data != nullptr &&
                stack.ne[0] == kHidden && stack.ne[1] == kStreams && stack.ne[3] == 1,
            "stack must be contiguous FP32 [2560, 4, tokens]");
    const std::int32_t tokens = stack.ne[2];
    require(tokens > 0, "tokens must be positive");
    require(eps > 0.0f, "eps must be positive");
    require_bf16(weights.norm, kWidth, 1, "norm must be BF16 [10240]");
    require_bf16(weights.down, kWidth, kLowrank, "down must be BF16 [10240, 320]");
    require_bf16(weights.up, kLowrank, kWidth, "up must be BF16 [320, 10240]");
    require((weights.inject == nullptr) == (inject_weights == nullptr),
            "inject weights and their output come together");
    if (weights.inject != nullptr) {
        require_bf16(weights.inject, kWidth, kStreams, "inject must be BF16 [10240, 4]");
        require(inject_weights->dtype == DType::FP32 && inject_weights->is_contiguous() &&
                    inject_weights->data != nullptr && inject_weights->ne[0] == kStreams &&
                    inject_weights->ne[1] == tokens,
                "inject output must be contiguous FP32 [4, tokens]");
    }
    require(mixed.dtype == DType::BF16 && mixed.is_contiguous() && mixed.data != nullptr &&
                mixed.ne[0] == kHidden && mixed.ne[1] == tokens,
            "mixed must be contiguous BF16 [2560, tokens]");

    auto scope         = workspace.scope();
    Tensor normalized  = workspace.alloc(DType::FP32, {kWidth, tokens});
    Tensor low         = workspace.alloc(DType::FP32, {kLowrank, tokens});
    Tensor scratch     = workspace.alloc(DType::FP32, {kStreams, tokens});
    float* inject_out  = inject_weights != nullptr ? static_cast<float*>(inject_weights->data)
                                                   : static_cast<float*>(scratch.data);

    hc_norm_kernel<<<dim3(kStreams, tokens), 256, 0, stream>>>(
        static_cast<const float*>(stack.data),
        static_cast<const __nv_bfloat16*>(weights.norm->data), eps,
        static_cast<float*>(normalized.data));
    CUDA_CHECK(cudaGetLastError());
    const int rows = kLowrank + (weights.inject != nullptr ? kStreams : 0);
    hc_down_kernel<<<div_up(rows, 8), 256, 0, stream>>>(
        static_cast<const float*>(normalized.data),
        static_cast<const __nv_bfloat16*>(weights.down->data),
        weights.inject != nullptr ? static_cast<const __nv_bfloat16*>(weights.inject->data)
                                  : nullptr,
        tokens, rows, static_cast<float*>(low.data), inject_out);
    CUDA_CHECK(cudaGetLastError());
    hc_up_mix_kernel<<<div_up(kHidden, 8), 256, 0, stream>>>(
        static_cast<const float*>(normalized.data), static_cast<const float*>(low.data),
        static_cast<const __nv_bfloat16*>(weights.up->data), tokens,
        static_cast<__nv_bfloat16*>(mixed.data));
    CUDA_CHECK(cudaGetLastError());
}

void hyper_connection_write(Tensor& stack, const Tensor& y, const Tensor& inject_weights,
                            cudaStream_t stream) {
    require(stack.dtype == DType::FP32 && stack.is_contiguous() && stack.data != nullptr &&
                stack.ne[0] == kHidden && stack.ne[1] == kStreams && stack.ne[3] == 1,
            "stack must be contiguous FP32 [2560, 4, tokens]");
    const std::int32_t tokens = stack.ne[2];
    require(tokens > 0, "tokens must be positive");
    require((y.dtype == DType::BF16 || y.dtype == DType::FP32) && y.is_contiguous() &&
                y.data != nullptr && y.ne[0] == kHidden && y.ne[1] == tokens,
            "y must be contiguous BF16 or FP32 [2560, tokens]");
    require(inject_weights.dtype == DType::FP32 && inject_weights.is_contiguous() &&
                inject_weights.data != nullptr && inject_weights.ne[0] == kStreams &&
                inject_weights.ne[1] == tokens,
            "inject weights must be contiguous FP32 [4, tokens]");
    const std::int64_t elements = static_cast<std::int64_t>(kWidth) * tokens;
    const auto blocks = static_cast<unsigned>(div_up(elements, std::int64_t{256}));
    if (y.dtype == DType::FP32) {
        hc_write_kernel<<<blocks, 256, 0, stream>>>(static_cast<float*>(stack.data),
                                                    static_cast<const float*>(y.data),
                                                    static_cast<const float*>(inject_weights.data),
                                                    elements);
    } else {
        hc_write_kernel<<<blocks, 256, 0, stream>>>(
            static_cast<float*>(stack.data), static_cast<const __nv_bfloat16*>(y.data),
            static_cast<const float*>(inject_weights.data), elements);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
