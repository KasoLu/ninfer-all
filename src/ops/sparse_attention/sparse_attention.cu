// ninfer::ops - Qwen3.8-Flash-Next sparse attention over selected blocks and the query's own
// incomplete block (contract in include/ninfer/ops/sparse_attention.h). A CTA takes one (query,
// KV head): its twelve query heads, one warp each, stream positions in 32-key tiles staged in
// shared memory, each lane scoring one key and accumulating eight output dims online. Up to eight
// queries split their positions over CTAs of 64 positions each (a grid fixed for any position, so
// a graph replays it), and a second launch merges the partial softmaxes; wider calls run the
// whole list in one CTA per (query, KV head).
#include "ninfer/ops/sparse_attention.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/common/math.h"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kHeadDim     = 256;
constexpr int kQueryHeads  = 24;
constexpr int kKvHeads     = 2;
constexpr int kGroup       = kQueryHeads / kKvHeads; // 12
constexpr int kTopBlocks   = 512;
constexpr int kTile        = 32;
constexpr int kThreads     = kGroup * 32;
constexpr int kRowStride   = kHeadDim + 2; // BF16 elements; 129 words per row spreads the banks
constexpr int kSplitTokens    = 8;
constexpr int kSplitPositions = 2 * kTile;
constexpr int kMaxPositions   = 4 * kTopBlocks + 3; // the selected blocks and the query's own
constexpr int kSplits         = (kMaxPositions + kSplitPositions - 1) / kSplitPositions;
constexpr int kPartialFloats  = kHeadDim + 2; // unnormalised output, running max, running sum

__device__ __forceinline__ int position_at(const int* blocks, int count, int tail_begin, int i) {
    return i < 4 * count ? blocks[i >> 2] * 4 + (i & 3) : tail_begin + (i - 4 * count);
}

// Split: this CTA takes positions [64 z, 64 z + 64) of the list and leaves its partial softmax in
// `partial`; otherwise the whole list, normalised into `out`.
template <bool Split>
__global__ void __launch_bounds__(kThreads)
    sparse_attention_kernel(const __nv_bfloat16* __restrict__ q, const int* __restrict__ first,
                            const int* __restrict__ selected, const int* __restrict__ counts,
                            const __nv_bfloat16* __restrict__ k_pages,
                            const __nv_bfloat16* __restrict__ v_pages,
                            const std::int32_t* __restrict__ block_table, float scale,
                            __nv_bfloat16* __restrict__ out, float* __restrict__ partial) {
    __shared__ float query[kGroup][kHeadDim];
    __shared__ __nv_bfloat16 keys[kTile * kRowStride];
    __shared__ __nv_bfloat16 values[kTile * kRowStride];

    const int t        = blockIdx.x;
    const int kv_head  = blockIdx.y;
    const int warp     = threadIdx.x >> 5;
    const int lane     = threadIdx.x & 31;
    const int q_head   = kv_head * kGroup + warp;
    const int position   = *first + t;
    const int count    = counts[t];
    const int tail_begin = (position + 1) / 4 * 4;
    const int total      = 4 * count + (position + 1 - tail_begin);
    const int* blocks    = selected + static_cast<std::int64_t>(t) * kTopBlocks;
    const int begin      = Split ? static_cast<int>(blockIdx.z) * kSplitPositions : 0;
    const int end        = Split ? min(total, begin + kSplitPositions) : total;

    for (int d = lane; d < kHeadDim; d += 32) {
        query[warp][d] = __bfloat162float(
            q[(static_cast<std::int64_t>(t) * kQueryHeads + q_head) * kHeadDim + d]);
    }

    float running_max = -INFINITY, running_sum = 0.0f;
    float acc[8]      = {};
    for (int base = begin; base < end; base += kTile) {
        __syncthreads();
        // Stage the tile: 32 rows of K and V, 4-byte words (the padded rows are not 16B aligned).
        for (int i = threadIdx.x; i < kTile * (kHeadDim / 2); i += kThreads) {
            const int row = i / (kHeadDim / 2), pair = i % (kHeadDim / 2);
            const int index = base + row;
            unsigned kw = 0, vw = 0;
            if (index < end) {
                const int p = position_at(blocks, count, tail_begin, index);
                const std::int64_t offset =
                    paged_kv_element_offset<kHeadDim, kKvHeads>(block_table, kv_head, p, 2 * pair);
                kw = *reinterpret_cast<const unsigned*>(k_pages + offset);
                vw = *reinterpret_cast<const unsigned*>(v_pages + offset);
            }
            *reinterpret_cast<unsigned*>(&keys[row * kRowStride + 2 * pair])   = kw;
            *reinterpret_cast<unsigned*>(&values[row * kRowStride + 2 * pair]) = vw;
        }
        __syncthreads();
        // Lane j scores key base + j for this warp's head.
        float score = -INFINITY;
        if (base + lane < end) {
            float dot = 0.0f;
            const __nv_bfloat162* row =
                reinterpret_cast<const __nv_bfloat162*>(&keys[lane * kRowStride]);
#pragma unroll 8
            for (int pair = 0; pair < kHeadDim / 2; ++pair) {
                const float2 k = __bfloat1622float2(row[pair]);
                dot = fmaf(query[warp][2 * pair], k.x, fmaf(query[warp][2 * pair + 1], k.y, dot));
            }
            score = dot * scale;
        }
        float tile_max = score;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, offset));
        }
        const float new_max    = fmaxf(running_max, tile_max);
        const float correction = __expf(running_max - new_max);
        const float weight     = base + lane < end ? __expf(score - new_max) : 0.0f;
        float weight_sum       = weight;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            weight_sum += __shfl_xor_sync(0xffffffffu, weight_sum, offset);
        }
        running_sum = running_sum * correction + weight_sum;
        running_max = new_max;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] *= correction;
        const int keys_in_tile = min(kTile, end - base);
        for (int j = 0; j < keys_in_tile; ++j) {
            const float p = __shfl_sync(0xffffffffu, weight, j);
            const __nv_bfloat162* row =
                reinterpret_cast<const __nv_bfloat162*>(&values[j * kRowStride + lane * 8]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 v = __bfloat1622float2(row[i]);
                acc[2 * i]     = fmaf(p, v.x, acc[2 * i]);
                acc[2 * i + 1] = fmaf(p, v.y, acc[2 * i + 1]);
            }
        }
    }
    if constexpr (Split) {
        float* slot =
            partial +
            ((static_cast<std::int64_t>(t) * kSplits + blockIdx.z) * kQueryHeads + q_head) *
                kPartialFloats;
#pragma unroll
        for (int i = 0; i < 8; ++i) slot[lane * 8 + i] = acc[i];
        if (lane == 0) {
            slot[kHeadDim]     = running_max;
            slot[kHeadDim + 1] = running_sum;
        }
    } else {
        const float inverse = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
        __nv_bfloat16* destination =
            out + (static_cast<std::int64_t>(t) * kQueryHeads + q_head) * kHeadDim + lane * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) destination[i] = __float2bfloat16_rn(acc[i] * inverse);
    }
}

// One CTA per (query, query head), one thread per output dim: the splits' partial softmaxes
// rescaled to the common maximum. A split with no positions has a zero sum and adds nothing.
__global__ void __launch_bounds__(kHeadDim)
    sparse_attention_merge_kernel(const float* __restrict__ partial,
                                  __nv_bfloat16* __restrict__ out) {
    const int t = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const float* first_split =
        partial + (static_cast<std::int64_t>(t) * kSplits * kQueryHeads + head) * kPartialFloats;
    constexpr std::int64_t kStride = std::int64_t(kQueryHeads) * kPartialFloats;
    float maximum                  = -INFINITY;
    for (int s = 0; s < kSplits; ++s) {
        const float* slot = first_split + s * kStride;
        if (slot[kHeadDim + 1] > 0.0f) maximum = fmaxf(maximum, slot[kHeadDim]);
    }
    float sum = 0.0f, value = 0.0f;
    for (int s = 0; s < kSplits; ++s) {
        const float* slot = first_split + s * kStride;
        if (slot[kHeadDim + 1] > 0.0f) {
            const float weight = __expf(slot[kHeadDim] - maximum);
            sum += slot[kHeadDim + 1] * weight;
            value += slot[d] * weight;
        }
    }
    out[(static_cast<std::int64_t>(t) * kQueryHeads + head) * kHeadDim + d] =
        __float2bfloat16_rn(sum > 0.0f ? value / sum : 0.0f);
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(std::string("sparse_softmax_attention: ") + message);
    }
}

} // namespace

std::size_t sparse_softmax_attention_workspace_bytes(std::int32_t tokens) {
    require(tokens > 0, "tokens must be positive");
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP32,
                       {kPartialFloats, kQueryHeads, kSplits, std::min(tokens, kSplitTokens)});
    return layout.peak_bytes(1);
}

void sparse_softmax_attention(const Tensor& q, const Tensor& first_position, const Tensor& selected,
                              const Tensor& counts, const PagedKVLayerView& cache, float scale,
                              WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    require(q.dtype == DType::BF16 && q.is_contiguous() && q.data != nullptr &&
                q.ne[0] == kHeadDim && q.ne[1] == kQueryHeads && q.ne[2] > 0 && q.ne[3] == 1,
            "q must be contiguous BF16 [256, 24, tokens]");
    const std::int32_t tokens = q.ne[2];
    require(first_position.dtype == DType::I32 && first_position.data != nullptr &&
                first_position.numel() >= 1,
            "first_position must be a device I32 word");
    require(selected.dtype == DType::I32 && selected.is_contiguous() && selected.data != nullptr &&
                selected.ne[0] == kTopBlocks && selected.ne[1] == tokens,
            "selected must be contiguous I32 [512, tokens]");
    require(counts.dtype == DType::I32 && counts.is_contiguous() && counts.data != nullptr &&
                counts.ne[0] == tokens,
            "counts must be contiguous I32 [tokens]");
    require(out.dtype == DType::BF16 && out.is_contiguous() && out.data != nullptr &&
                out.ne[0] == kHeadDim && out.ne[1] == kQueryHeads && out.ne[2] == tokens,
            "out must be contiguous BF16 [256, 24, tokens]");
    require(cache.storage == KvCacheStorage::BFloat16 && cache.head_dim == kHeadDim &&
                cache.num_kv_heads == kKvHeads && cache.k_pages.dtype == DType::BF16 &&
                cache.v_pages.dtype == DType::BF16 && cache.block_table.dtype == DType::I32 &&
                cache.block_table.data != nullptr,
            "cache must be a BF16 paged layer of 2 heads of 256");
    require(std::isfinite(scale) && scale > 0.0f, "scale must be positive and finite");
    const auto* q_p      = static_cast<const __nv_bfloat16*>(q.data);
    const auto* first_p  = static_cast<const int*>(first_position.data);
    const auto* select_p = static_cast<const int*>(selected.data);
    const auto* counts_p = static_cast<const int*>(counts.data);
    const auto* k_p      = static_cast<const __nv_bfloat16*>(cache.k_pages.data);
    const auto* v_p      = static_cast<const __nv_bfloat16*>(cache.v_pages.data);
    const auto* table_p  = static_cast<const std::int32_t*>(cache.block_table.data);
    auto* out_p          = static_cast<__nv_bfloat16*>(out.data);
    if (tokens > kSplitTokens) {
        sparse_attention_kernel<false><<<dim3(tokens, kKvHeads), kThreads, 0, stream>>>(
            q_p, first_p, select_p, counts_p, k_p, v_p, table_p, scale, out_p, nullptr);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    auto scope      = workspace.scope();
    Tensor partial  = workspace.alloc(DType::FP32, {kPartialFloats, kQueryHeads, kSplits, tokens});
    auto* partial_p = static_cast<float*>(partial.data);
    sparse_attention_kernel<true><<<dim3(tokens, kKvHeads, kSplits), kThreads, 0, stream>>>(
        q_p, first_p, select_p, counts_p, k_p, v_p, table_p, scale, out_p, partial_p);
    CUDA_CHECK(cudaGetLastError());
    sparse_attention_merge_kernel<<<dim3(tokens, kQueryHeads), kHeadDim, 0, stream>>>(partial_p,
                                                                                      out_p);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
