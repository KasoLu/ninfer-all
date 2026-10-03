// ninfer::ops - Qwen3.8-Flash-Next MoE router (contract in include/ninfer/ops/moe_route.h).
// One CTA per token: 513 dot products over the block input (one warp per row), a softmax over
// the 512 logits, then ten rounds of an arg-max that takes the lower index on ties.
#include "ninfer/ops/moe_route.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops {
namespace {

constexpr int kHidden  = 2560;
constexpr int kExperts = 512;
constexpr int kTopK    = 10;
constexpr int kThreads = 512;
static_assert(kThreads == kExperts, "one thread per expert in the top-10 rounds");

template <typename Input>
__device__ __forceinline__ float load(const Input* m, std::int64_t i) {
    if constexpr (std::is_same_v<Input, float>) {
        return m[i];
    } else {
        return __bfloat162float(m[i]);
    }
}

template <typename Input>
__global__ void __launch_bounds__(kThreads)
    moe_route_kernel(const Input* __restrict__ m, const __nv_bfloat16* __restrict__ router,
                     const __nv_bfloat16* __restrict__ shared_gate, int* __restrict__ ids,
                     float* __restrict__ weights, float* __restrict__ shared) {
    __shared__ float input[kHidden];
    __shared__ float logits[kExperts + 1];
    __shared__ float best_value[kThreads / 32];
    __shared__ int best_index[kThreads / 32];
    __shared__ float chosen[kTopK];
    __shared__ int chosen_index[kTopK];
    const int t    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    for (int d = tid; d < kHidden; d += kThreads) input[d] = load(m, static_cast<std::int64_t>(t) * kHidden + d);
    __syncthreads();
    for (int row = warp; row <= kExperts; row += kThreads / 32) {
        const __nv_bfloat16* w = row < kExperts ? router + static_cast<std::int64_t>(row) * kHidden : shared_gate;
        float dot = 0.0f;
        for (int d = lane; d < kHidden; d += 32) dot = fmaf(__bfloat162float(w[d]), input[d], dot);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, offset);
        if (lane == 0) logits[row] = dot;
    }
    __syncthreads();
    if (tid == 0) shared[t] = 1.0f / (1.0f + __expf(-logits[kExperts]));
    // Top-10 of the logits (softmax is monotonic), lower index on ties.
    for (int k = 0; k < kTopK; ++k) {
        float value = logits[tid]; // kThreads == kExperts
        int index   = tid;
        for (int j = 0; j < k; ++j) {
            if (chosen_index[j] == tid) value = -INFINITY, index = kExperts;
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_value = __shfl_xor_sync(0xffffffffu, value, offset);
            const int other_index   = __shfl_xor_sync(0xffffffffu, index, offset);
            if (other_value > value || (other_value == value && other_index < index)) {
                value = other_value;
                index = other_index;
            }
        }
        if (lane == 0) {
            best_value[warp] = value;
            best_index[warp] = index;
        }
        __syncthreads();
        if (tid == 0) {
            float v = best_value[0];
            int i   = best_index[0];
            for (int w = 1; w < kThreads / 32; ++w) {
                if (best_value[w] > v || (best_value[w] == v && best_index[w] < i)) {
                    v = best_value[w];
                    i = best_index[w];
                }
            }
            chosen[k]       = v;
            chosen_index[k] = i;
        }
        __syncthreads();
    }
    // Renormalised softmax over the chosen ten: the full softmax's denominator cancels.
    if (tid == 0) {
        const float top = chosen[0];
        float sum       = 0.0f;
        float e[kTopK];
        for (int k = 0; k < kTopK; ++k) sum += (e[k] = __expf(chosen[k] - top));
        for (int k = 0; k < kTopK; ++k) {
            ids[t * kTopK + k]     = chosen_index[k];
            weights[t * kTopK + k] = e[k] / sum;
        }
    }
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("moe_route: ") + message); }
}

} // namespace

void moe_route(const Tensor& m, const Tensor& router, const Tensor& shared_gate, Tensor& ids,
               Tensor& weights, Tensor& shared, cudaStream_t stream) {
    require((m.dtype == DType::BF16 || m.dtype == DType::FP32) && m.is_contiguous() &&
                m.data != nullptr && m.ne[0] == kHidden && m.ne[1] > 0 && m.ne[2] == 1,
            "m must be contiguous BF16 or FP32 [2560, tokens]");
    const std::int32_t tokens = m.ne[1];
    require(router.dtype == DType::BF16 && router.is_contiguous() && router.data != nullptr &&
                router.ne[0] == kHidden && router.ne[1] == kExperts,
            "router must be BF16 [2560, 512]");
    require(shared_gate.dtype == DType::BF16 && shared_gate.is_contiguous() &&
                shared_gate.data != nullptr && shared_gate.ne[0] == kHidden && shared_gate.ne[1] == 1,
            "shared_gate must be BF16 [2560]");
    require(ids.dtype == DType::I32 && ids.is_contiguous() && ids.data != nullptr &&
                ids.ne[0] == kTopK && ids.ne[1] == tokens,
            "ids must be I32 [10, tokens]");
    require(weights.dtype == DType::FP32 && weights.is_contiguous() && weights.data != nullptr &&
                weights.ne[0] == kTopK && weights.ne[1] == tokens,
            "weights must be FP32 [10, tokens]");
    require(shared.dtype == DType::FP32 && shared.is_contiguous() && shared.data != nullptr &&
                shared.ne[0] == tokens,
            "shared must be FP32 [tokens]");
    const auto* r = static_cast<const __nv_bfloat16*>(router.data);
    const auto* g = static_cast<const __nv_bfloat16*>(shared_gate.data);
    if (m.dtype == DType::FP32) {
        moe_route_kernel<float><<<tokens, kThreads, 0, stream>>>(
            static_cast<const float*>(m.data), r, g, static_cast<int*>(ids.data),
            static_cast<float*>(weights.data), static_cast<float*>(shared.data));
    } else {
        moe_route_kernel<__nv_bfloat16><<<tokens, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(m.data), r, g, static_cast<int*>(ids.data),
            static_cast<float*>(weights.data), static_cast<float*>(shared.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
