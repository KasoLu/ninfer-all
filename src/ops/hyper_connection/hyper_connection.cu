// ninfer::ops - hyper-connection read/write of Qwen3.8-Flash-Next (contract in
// include/ninfer/ops/hyper_connection.h). Four launches: per-stream RMSNorm into FP32 workspace,
// the down and inject GEMVs with their activations, the up GEMV with the gated stream mix, and the
// weighted write. Every product accumulates in FP32. The two GEMVs stream their 6.5 MB of weights
// in 16-byte vectors with every CTA busy at one token: the down rows split K over a CTA's warps,
// the up rows of eight hidden indices are one contiguous run per stream.
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
// Tokens one CTA accumulates at once; wider calls cover column chunks with grid.y.
constexpr int kColumnChunk = 8;
// The down GEMV: one CTA per row, its warps splitting the 10240-wide input.
constexpr int kDownWarps = 8;
constexpr int kDownSlice = kWidth / kDownWarps; // 1280 inputs, five 8-wide vectors per lane
static_assert(kDownSlice % 256 == 0);
// The up GEMV: per CTA eight hidden indices, one warp per stream, four lanes per 320-wide row.
constexpr int kUpRows    = 8;
constexpr int kUpLanes   = 4;
constexpr int kUpVectors = kLowrank / 8; // 40 per row
static_assert(kUpRows * kUpLanes == 32 && kUpVectors % kUpLanes == 0);
static_assert(kUpRows * kColumnChunk <= kStreams * 32);
// The norm: one thread per four values of a stream.
constexpr int kNormThreads = kHidden / 4;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + __expf(-x)); }

__device__ __forceinline__ void unpack_bf16x8(const uint4& v, float (&out)[8]) {
    const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(pairs[i]);
        out[2 * i]     = f.x;
        out[2 * i + 1] = f.y;
    }
}

// Eight consecutive FP32 values, 32-byte aligned.
__device__ __forceinline__ void load_f32x8(const float* p, float (&out)[8]) {
    const float4 a = *reinterpret_cast<const float4*>(p);
    const float4 b = *reinterpret_cast<const float4*>(p + 4);
    out[0] = a.x, out[1] = a.y, out[2] = a.z, out[3] = a.w;
    out[4] = b.x, out[5] = b.y, out[6] = b.z, out[7] = b.w;
}

// One block per (stream, token), four values per thread: xn = x * rsqrt(mean x^2 + eps) * (1 + g).
__global__ void __launch_bounds__(kNormThreads)
    hc_norm_kernel(const float* __restrict__ stack, const __nv_bfloat16* __restrict__ norm,
                   float eps, float* __restrict__ normalized) {
    const int c             = blockIdx.x;
    const int t             = blockIdx.y;
    const std::int64_t base = (static_cast<std::int64_t>(t) * kStreams + c) * kHidden;
    __shared__ float partial[kNormThreads / 32];
    const float4 x = reinterpret_cast<const float4*>(stack + base)[threadIdx.x];
    float sum      = warp_sum(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
    if ((threadIdx.x & 31) == 0) { partial[threadIdx.x >> 5] = sum; }
    __syncthreads();
    sum = 0.0f;
#pragma unroll
    for (int w = 0; w < kNormThreads / 32; ++w) { sum += partial[w]; }
    const float scale = rsqrtf(sum / kHidden + eps);
    const auto* g   = reinterpret_cast<const __nv_bfloat162*>(norm + c * kHidden) + 2 * threadIdx.x;
    const float2 g0 = __bfloat1622float2(g[0]), g1 = __bfloat1622float2(g[1]);
    reinterpret_cast<float4*>(normalized + base)[threadIdx.x] =
        make_float4(x.x * scale * (1.0f + g0.x), x.y * scale * (1.0f + g0.y),
                    x.z * scale * (1.0f + g1.x), x.w * scale * (1.0f + g1.y));
}

// One block per (row of [down; inject], column chunk): lowrank + streams rows over the 10240-wide
// input, each warp a 1280-wide slice. Rows below lowrank write silu(v / n) to `low`; inject rows
// write 2 sigmoid(v / n).
__global__ void __launch_bounds__(kDownWarps * 32)
    hc_down_kernel(const float* __restrict__ normalized, const __nv_bfloat16* __restrict__ down,
                   const __nv_bfloat16* __restrict__ inject, int tokens, float* __restrict__ low,
                   float* __restrict__ inject_weights) {
    __shared__ float partial[kDownWarps][kColumnChunk];
    const int row   = blockIdx.x;
    const int t0    = blockIdx.y * kColumnChunk;
    const int count = min(kColumnChunk, tokens - t0);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const __nv_bfloat16* weights = row < kLowrank
                                       ? down + static_cast<std::int64_t>(row) * kWidth
                                       : inject + static_cast<std::int64_t>(row - kLowrank) * kWidth;
    const int first              = warp * kDownSlice;
    float acc[kColumnChunk]      = {};
#pragma unroll
    for (int i = 0; i < kDownSlice / 256; ++i) {
        const int k = first + 8 * (lane + 32 * i);
        float w[8];
        unpack_bf16x8(__ldg(reinterpret_cast<const uint4*>(weights + k)), w);
#pragma unroll
        for (int j = 0; j < kColumnChunk; ++j) {
            if (j < count) {
                float x[8];
                load_f32x8(normalized + static_cast<std::int64_t>(t0 + j) * kWidth + k, x);
#pragma unroll
                for (int e = 0; e < 8; ++e) { acc[j] = fmaf(w[e], x[e], acc[j]); }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < kColumnChunk; ++j) {
        const float v = warp_sum(acc[j]);
        if (lane == 0) { partial[warp][j] = v; }
    }
    __syncthreads();
    if (warp != 0 || lane >= count) { return; }
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < kDownWarps; ++w) { v += partial[w][lane]; }
    v /= kStreams;
    const int t = t0 + lane;
    if (row < kLowrank) {
        low[static_cast<std::int64_t>(t) * kLowrank + row] = v * sigmoid_f(v);
    } else {
        inject_weights[static_cast<std::int64_t>(t) * kStreams + row - kLowrank] =
            2.0f * sigmoid_f(v);
    }
}

// One block per (eight consecutive hidden indices, column chunk), one warp per stream c: the
// eight gate rows c * hidden + d of `up` are contiguous (each lowrank wide), four lanes on each.
// Then mixed[d] = (1/n) sum_c sigmoid(gate_c) xn[c, d].
__global__ void __launch_bounds__(kStreams * 32)
    hc_up_mix_kernel(const float* __restrict__ normalized, const float* __restrict__ low,
                     const __nv_bfloat16* __restrict__ up, int tokens,
                     __nv_bfloat16* __restrict__ mixed) {
    __shared__ float gated[kStreams][kUpRows][kColumnChunk];
    const int c    = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int r = lane / kUpLanes, part = lane % kUpLanes;
    const int d0                 = blockIdx.x * kUpRows;
    const int d                  = d0 + r;
    const int t0                 = blockIdx.y * kColumnChunk;
    const int count              = min(kColumnChunk, tokens - t0);
    const __nv_bfloat16* weights = up + static_cast<std::int64_t>(c * kHidden + d) * kLowrank;
    float acc[kColumnChunk]      = {};
#pragma unroll
    for (int i = 0; i < kUpVectors / kUpLanes; ++i) {
        const int v = part + kUpLanes * i;
        float w[8];
        unpack_bf16x8(__ldg(reinterpret_cast<const uint4*>(weights + 8 * v)), w);
#pragma unroll
        for (int j = 0; j < kColumnChunk; ++j) {
            if (j < count) {
                float x[8];
                load_f32x8(low + static_cast<std::int64_t>(t0 + j) * kLowrank + 8 * v, x);
#pragma unroll
                for (int e = 0; e < 8; ++e) { acc[j] = fmaf(w[e], x[e], acc[j]); }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < kColumnChunk; ++j) {
        acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], 1);
        acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], 2);
        if (part == 0 && j < count) {
            gated[c][r][j] =
                sigmoid_f(acc[j]) *
                normalized[static_cast<std::int64_t>(t0 + j) * kWidth + c * kHidden + d];
        }
    }
    __syncthreads();
    if (threadIdx.x >= kUpRows * kColumnChunk) { return; }
    const int row = threadIdx.x / kColumnChunk, j = threadIdx.x % kColumnChunk;
    if (j < count) {
        float sum = 0.0f;
#pragma unroll
        for (int s = 0; s < kStreams; ++s) { sum += gated[s][row][j]; }
        mixed[static_cast<std::int64_t>(t0 + j) * kHidden + d0 + row] =
            __float2bfloat16_rn(sum / kStreams);
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

__global__ void __launch_bounds__(256)
    hc_expand_kernel(const __nv_bfloat16* __restrict__ x, float* __restrict__ stack,
                     std::int64_t elements) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    // stack element i is stream (i / hidden) % streams of token i / width.
    const std::int64_t token = i / kWidth;
    const std::int64_t d     = i % kHidden;
    stack[i]                 = __bfloat162float(x[token * kHidden + d]);
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("hyper_connection: ") + message); }
}

bool aligned16(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0; }

// The kernels read the weights in 16-byte vectors.
void require_bf16(const Tensor* tensor, std::int32_t n0, std::int32_t n1, const char* name) {
    require(tensor != nullptr && tensor->data != nullptr && aligned16(tensor->data), name);
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
                aligned16(stack.data) && stack.ne[0] == kHidden && stack.ne[1] == kStreams &&
                stack.ne[3] == 1,
            "stack must be contiguous 16-byte aligned FP32 [2560, 4, tokens]");
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

    hc_norm_kernel<<<dim3(kStreams, tokens), kNormThreads, 0, stream>>>(
        static_cast<const float*>(stack.data),
        static_cast<const __nv_bfloat16*>(weights.norm->data), eps,
        static_cast<float*>(normalized.data));
    CUDA_CHECK(cudaGetLastError());
    const int rows   = kLowrank + (weights.inject != nullptr ? kStreams : 0);
    const int chunks = div_up(tokens, kColumnChunk);
    hc_down_kernel<<<dim3(rows, chunks), kDownWarps * 32, 0, stream>>>(
        static_cast<const float*>(normalized.data),
        static_cast<const __nv_bfloat16*>(weights.down->data),
        weights.inject != nullptr ? static_cast<const __nv_bfloat16*>(weights.inject->data)
                                  : nullptr,
        tokens, static_cast<float*>(low.data), inject_out);
    CUDA_CHECK(cudaGetLastError());
    hc_up_mix_kernel<<<dim3(kHidden / kUpRows, chunks), kStreams * 32, 0, stream>>>(
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

void hyper_connection_expand(const Tensor& x, Tensor& stack, cudaStream_t stream) {
    require(stack.dtype == DType::FP32 && stack.is_contiguous() && stack.data != nullptr &&
                stack.ne[0] == kHidden && stack.ne[1] == kStreams && stack.ne[3] == 1,
            "stack must be contiguous FP32 [2560, 4, tokens]");
    const std::int32_t tokens = stack.ne[2];
    require(tokens > 0, "tokens must be positive");
    require(x.dtype == DType::BF16 && x.is_contiguous() && x.data != nullptr &&
                x.ne[0] == kHidden && x.ne[1] == tokens,
            "x must be contiguous BF16 [2560, tokens]");
    const std::int64_t elements = static_cast<std::int64_t>(kWidth) * tokens;
    hc_expand_kernel<<<static_cast<unsigned>(div_up(elements, std::int64_t{256})), 256, 0,
                       stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                 static_cast<float*>(stack.data), elements);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
