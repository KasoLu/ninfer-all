#pragma once

// Signed int8, per-token G64 KV-cache codec shared by append and causal-attention kernels.
// This header owns index math, vectorized decode, and scalar encode; there is deliberately no
// standalone transcode kernel in the production path.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/e8_root_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops {

inline constexpr int kKVCacheInt8HeadDim = 256;
inline constexpr int kKVCacheInt8Group   = 64;
inline constexpr int kKVCacheInt8Groups  = kKVCacheInt8HeadDim / kKVCacheInt8Group;

template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_int8_quant_code_index(int physical_page, int kv_head, int d, int page_offset) {
    return paged_kv_element_offset<kKVCacheInt8HeadDim, Geometry::KVHeads>(physical_page, kv_head,
                                                                           page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_int8_quant_scale_index(int physical_page, int kv_head, int group, int page_offset) {
    return paged_kv_element_offset<kKVCacheInt8Groups, Geometry::KVHeads>(physical_page, kv_head,
                                                                          page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_int8_quant_src_index(int kv_head, int d,
                                                                      int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kKVCacheInt8HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

struct KVCacheInt8QuantParams {
    __half scale;
    float inverse_scale;
};

// Exact persistent group-scale boundary shared by standalone and fused append. The stored scale is
// FP16-RNE(absmax/127); codes always use the reciprocal of that represented FP16 value.
__device__ __forceinline__ KVCacheInt8QuantParams kv_cache_int8_quant_params(float absmax) {
    const __half scale            = __float2half_rn(absmax > 0.0f ? absmax / 127.0f : 0.0f);
    const float represented_scale = __half2float(scale);
    return {
        .scale         = scale,
        .inverse_scale = represented_scale > 0.0f ? 1.0f / represented_scale : 0.0f,
    };
}

__device__ __forceinline__ std::int8_t kv_cache_int8_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-127, min(127, q));
    return static_cast<std::int8_t>(q);
}

// --- packed signed int4 value codec (rk8v4) ---------------------------------------------------
// The rk8v4 profile keeps K as rotated INT8 and stores V as two signed 4-bit codes per byte, so a
// V plane is half as wide as its INT8 counterpart while the G64 scale plane is unchanged. The
// group encoding mirrors the INT8 one with 7 in place of 127, which is the largest magnitude a
// signed 4-bit code can represent.
inline constexpr int kKVCacheInt4Max = 7;

// Values use a finer group than keys. Four-bit codes resolve a group to 15 levels, so the group
// absmax sets the step for every member and one outlier costs the whole group precision. Halving
// the group to 32 halves the number of values a single outlier can degrade, at the price of four
// extra FP16 scales per row.
inline constexpr int kKVCacheInt4ValueGroup  = 32;
inline constexpr int kKVCacheInt4ValueGroups = kKVCacheInt8HeadDim / kKVCacheInt4ValueGroup;

template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_int4_value_scale_index(int physical_page, int kv_head, int group, int page_offset) {
    return paged_kv_element_offset<kKVCacheInt4ValueGroups, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_int4_value_code_index(int physical_page, int kv_head, int packed_d, int page_offset) {
    return paged_kv_element_offset<kKVCacheInt8HeadDim / 2, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, packed_d);
}

// Persistent group-scale boundary for packed int4 values: FP16-RNE(absmax/7). Codes always use
// the reciprocal of that represented FP16 value, exactly as the INT8 codec does.
__device__ __forceinline__ KVCacheInt8QuantParams kv_cache_int4_quant_params(float absmax) {
    const __half scale = __float2half_rn(
        absmax > 0.0f ? absmax / static_cast<float>(kKVCacheInt4Max) : 0.0f);
    const float represented_scale = __half2float(scale);
    return {
        .scale         = scale,
        .inverse_scale = represented_scale > 0.0f ? 1.0f / represented_scale : 0.0f,
    };
}

__device__ __forceinline__ std::int8_t kv_cache_int4_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-kKVCacheInt4Max, min(kKVCacheInt4Max, q));
    return static_cast<std::int8_t>(q);
}

// Byte layout: dimension d occupies the low nibble when d is even and the high nibble when odd,
// so a packed byte holds the adjacent pair (2p, 2p+1).
__device__ __forceinline__ std::uint8_t kv_cache_int4_pack(std::int8_t low, std::int8_t high) {
    return static_cast<std::uint8_t>((static_cast<unsigned>(low) & 0x0fu) |
                                     ((static_cast<unsigned>(high) & 0x0fu) << 4));
}

__device__ __forceinline__ std::int8_t kv_cache_int4_unpack(std::uint8_t packed, int high) {
    const unsigned nibble = high != 0 ? (packed >> 4) : (packed & 0x0fu);
    return static_cast<std::int8_t>(static_cast<int>(nibble ^ 8u) - 8);
}

// Dequantize the eight consecutive dimensions [d, d+8) that one INT8 call would have produced,
// reading the four bytes that hold them. Returns the same packed bf16 int4 as the INT8 path so
// the consuming kernels differ only in which loader they call.
__device__ __forceinline__ int4 kv_cache_int4_dequant_i4x8_from(const std::uint8_t* packed4,
                                                                float s) {
    const unsigned raw          = load_vec<unsigned>(packed4);
    const std::uint8_t* bytes   = reinterpret_cast<const std::uint8_t*>(&raw);
    unsigned out[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(kv_cache_int4_unpack(bytes[i], 0)) * s;
        const float x1 = static_cast<float>(kv_cache_int4_unpack(bytes[i], 1)) * s;
        out[i]         = pack_bf16x2(x0, x1);
    }
    return make_int4(static_cast<int>(out[0]), static_cast<int>(out[1]), static_cast<int>(out[2]),
                     static_cast<int>(out[3]));
}

// --- rotated Lloyd-Max 4-bit key codec (rk4v4) ------------------------------------------------
// The rk4v4 profile stores each rotated key dimension as a 4-bit index into the symmetric
// 16-level Lloyd-Max quantizer for N(0,1), packed two per byte exactly like the rk8v4 value
// plane (low nibble = even dimension). Index = sign << 3 | magnitude, magnitude 0..7.
//
// Each level is represented by a fixed INT8 code, round(L_m / L_7 * 127), and the G64 scale plane
// keeps one FP16 scale per 64 dimensions, exactly as the INT8 key coding does. Expanding a nibble
// to its code therefore reproduces an ordinary rotated-INT8 key, and every INT8-family attention
// kernel consumes it through its unchanged m16n8k32.s8 QK path. The encoder picks each index
// against the group's RMS and then takes the least-squares scale for the chosen codes.
// --- packed signed int4 E8 key codec (rk4v4-e8) ------------------------------------------------
// The rk4v4-e8 key plane stores the rotated key as two signed 4-bit codes per byte under the INT8
// family's G64 FP16 scale, FP16-RNE(absmax/7). Before rounding, each block of eight consecutive
// scaled dimensions is snapped to the nearest point of the E8 lattice (D8 or its half-integer
// coset D8+1/2); the stored code is then that point rounded to an integer and clamped to
// [-8, 7]. No coset bit is kept, so a coset point loses its half on the way into the code. The
// read side is therefore exactly the plain packed int4 decode.
//
// Every helper below takes one coordinate per lane with the block held by an aligned lane octet
// (lanes 8j..8j+7), and must be called by the whole warp.
inline constexpr int kKVCacheInt4KeyMin = -8;

// Nearest D8 point to the octet's vector y: round every coordinate, and when the rounded sum is
// odd move the coordinate with the largest rounding error (lowest lane on a tie) to its other
// neighbour, upwards when it rounded exactly.
__device__ __forceinline__ float kv_cache_e8_nearest_d8(float y, int lane) {
    constexpr unsigned FullMask = 0xffffffffu;
    float r = rintf(y);
    int odd = static_cast<int>(r) & 1;
#pragma unroll
    for (int step = 1; step < 8; step <<= 1) { odd ^= __shfl_xor_sync(FullMask, odd, step); }
    // The parity differs between octets, so the reduction runs unconditionally: a full-mask
    // shuffle under a branch only some octets take would be undefined.
    float worst    = fabsf(y - r);
    int worst_lane = lane;
#pragma unroll
    for (int step = 1; step < 8; step <<= 1) {
        const float other    = __shfl_xor_sync(FullMask, worst, step);
        const int other_lane = __shfl_xor_sync(FullMask, worst_lane, step);
        if (other > worst || (other == worst && other_lane < worst_lane)) {
            worst      = other;
            worst_lane = other_lane;
        }
    }
    if (odd != 0 && worst_lane == lane) { r += y >= r ? 1.0f : -1.0f; }
    return r;
}

// Octet sum in butterfly order. Every lane ends with the same bits because each step adds the
// same two operands, only commuted; the product is rounded explicitly so the compiler cannot
// contract it into a lane-asymmetric FMA.
__device__ __forceinline__ float kv_cache_e8_octet_sq_distance(float x, float p) {
    constexpr unsigned FullMask = 0xffffffffu;
    const float d = x - p;
    float sum     = __fmul_rn(d, d);
#pragma unroll
    for (int step = 1; step < 8; step <<= 1) {
        sum = __fadd_rn(sum, __shfl_xor_sync(FullMask, sum, step));
    }
    return sum;
}

// Nearest E8 point: the closer of the D8 and D8+1/2 candidates, D8 on a tie.
__device__ __forceinline__ float kv_cache_e8_nearest(float x, int lane) {
    const float integer_point = kv_cache_e8_nearest_d8(x, lane);
    const float coset_point   = kv_cache_e8_nearest_d8(x - 0.5f, lane) + 0.5f;
    const float integer_dist  = kv_cache_e8_octet_sq_distance(x, integer_point);
    const float coset_dist    = kv_cache_e8_octet_sq_distance(x, coset_point);
    return coset_dist < integer_dist ? coset_point : integer_point;
}

__device__ __forceinline__ std::int8_t kv_cache_int4_e8_key_code(float x, float inv_scale,
                                                                 int lane) {
    const float scaled = inv_scale == 0.0f ? 0.0f : __fmul_rn(x, inv_scale);
    const float point  = kv_cache_e8_nearest(scaled, lane);
    int q              = __float2int_rn(point);
    q                  = max(kKVCacheInt4KeyMin, min(kKVCacheInt4Max, q));
    return static_cast<std::int8_t>(q);
}

// Expand the eight packed bytes holding dimensions [d, d+16) into sixteen signed int8 codes in
// dimension order, so an int4 key row can be staged into the same shared layout, and consumed by
// the same s8 MMA, as an INT8 key row.
__device__ __forceinline__ int4 kv_cache_int4_unpack_i8x16(uint2 packed) {
    const auto expand = [](unsigned word, unsigned& first, unsigned& second) {
        const unsigned low  = __vsub4((word & 0x0f0f0f0fu) ^ 0x08080808u, 0x08080808u);
        const unsigned high = __vsub4(((word >> 4) & 0x0f0f0f0fu) ^ 0x08080808u, 0x08080808u);
        first               = __byte_perm(low, high, 0x5140);
        second              = __byte_perm(low, high, 0x7362);
    };
    unsigned out[4];
    expand(packed.x, out[0], out[1]);
    expand(packed.y, out[2], out[3]);
    return make_int4(static_cast<int>(out[0]), static_cast<int>(out[1]), static_cast<int>(out[2]),
                     static_cast<int>(out[3]));
}
// --- packed signed 6-bit E8 key codec (rk6v4-e8) ---------------------------------------------
// The rk6v4-e8 key plane stores each rotated key dimension as a signed 6-bit code under the INT8
// family's G64 FP16 scale, FP16-RNE(absmax/31). Before rounding, each block of eight consecutive
// scaled dimensions is snapped to the nearest point of the E8 lattice (D8 or its half-integer
// coset D8+1/2); the stored code is that point rounded to an integer and clamped to [-32, 31].
// No coset bit is kept, so a coset point loses its half on the way into the code -- the same
// deliberate approximation as the rk4v4-e8 path.
//
// Four codes pack into one 24-bit word (three bytes): a 16-dimension block is 12 bytes (three
// aligned u32), a 64-d group 48 bytes, and one (token, kv_head) row 192 bytes. The stored u6 code
// is value & 0x3F (two's complement mod 64); unpacking mirrors the int4 xor-shift convention,
// (code ^ 32) - 32.
inline constexpr int kKVCacheI6HeadExtent = kKVCacheInt8HeadDim * 3 / 4;

__device__ __forceinline__ std::uint8_t kv_cache_i6_code_from_int(int value) {
    return static_cast<std::uint8_t>(max(-32, min(31, value))) & 0x3Fu;
}

__device__ __forceinline__ std::int8_t kv_cache_unpack_i6(std::uint8_t code) {
    return static_cast<std::int8_t>((static_cast<int>(code) ^ 32) - 32);
}

// Pack the four consecutive 6-bit codes (dimension order) into the three bytes holding them.
__device__ __forceinline__ void kv_cache_pack_i6_quad(const std::uint8_t* codes4, std::uint8_t* out3) {
    const std::uint32_t word = static_cast<std::uint32_t>(codes4[0]) |
                               (static_cast<std::uint32_t>(codes4[1]) << 6) |
                               (static_cast<std::uint32_t>(codes4[2]) << 12) |
                               (static_cast<std::uint32_t>(codes4[3]) << 18);
    out3[0] = static_cast<std::uint8_t>(word & 0xFFu);
    out3[1] = static_cast<std::uint8_t>((word >> 8) & 0xFFu);
    out3[2] = static_cast<std::uint8_t>((word >> 16) & 0xFFu);
}

// Warp-wide 24-bit quad gather and write. Four consecutive lanes hold one 24-bit quad (three
// bytes); every lane converges on each full-mask shuffle read from an explicit source lane, and
// the four-lane leader writes the packed d0 quad at `row` and the d1 quad at `row + 24` (both
// halves of this group's 48-byte row). All lanes of the warp must call this together. The
// explicit-source full-mask gather -- not a __shfl_down_sync over a per-quad sub-mask -- is the
// only mask form this codebase relies on: the sub-mask variant made a quad's non-leader lanes
// read source lanes outside their mask, undefined behaviour that deadlocked sm_89 (4090 hang
// 2026-09-04).
template <typename CacheK>
__device__ __forceinline__ void kv_cache_lanes_write_i6_quad(std::uint8_t code0, std::uint8_t code1,
                                                             int lane, unsigned full_mask,
                                                             std::int64_t row, CacheK cache_k) {
    const int c0i   = static_cast<int>(code0);
    const int c1i   = static_cast<int>(code1);
    const int qbase = lane & ~3;
    const std::uint8_t quad0[4] = {
        static_cast<std::uint8_t>(__shfl_sync(full_mask, c0i, qbase)),
        static_cast<std::uint8_t>(__shfl_sync(full_mask, c0i, qbase + 1)),
        static_cast<std::uint8_t>(__shfl_sync(full_mask, c0i, qbase + 2)),
        static_cast<std::uint8_t>(__shfl_sync(full_mask, c0i, qbase + 3))};
    const std::uint8_t quad1[4] = {
        static_cast<std::uint8_t>(__shfl_sync(full_mask, c1i, qbase)),
        static_cast<std::uint8_t>(__shfl_sync(full_mask, c1i, qbase + 1)),
        static_cast<std::uint8_t>(__shfl_sync(full_mask, c1i, qbase + 2)),
        static_cast<std::uint8_t>(__shfl_sync(full_mask, c1i, qbase + 3))};
    if ((lane & 3) == 0) {
        const std::int64_t quad_off = static_cast<std::int64_t>(lane >> 2) * 3;
        auto* k_bytes               = reinterpret_cast<std::uint8_t*>(cache_k);
        kv_cache_pack_i6_quad(quad0, &k_bytes[row + quad_off]);
        kv_cache_pack_i6_quad(quad1, &k_bytes[row + 24 + quad_off]);
    }
}

// Expand the 12 bytes holding dimensions [d, d+16) into sixteen signed int8 codes in dimension
// order.
__device__ __forceinline__ void kv_cache_unpack_i6x16(const std::uint8_t* src12, std::int8_t* dst16) {
    const std::uint32_t* w3 = reinterpret_cast<const std::uint32_t*>(src12);
    const std::uint32_t b0  = w3[0];
    const std::uint32_t b1  = w3[1];
    const std::uint32_t b2  = w3[2];
    const std::uint32_t q0  = b0 & 0xffffffu;
    const std::uint32_t q1  = ((b0 >> 24) & 0xffu) | ((b1 & 0xffffu) << 8);
    const std::uint32_t q2  = ((b1 >> 16) & 0xffffu) | ((b2 & 0xffu) << 16);
    const std::uint32_t q3  = (b2 >> 8) & 0xffffffu;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const std::uint32_t word = j == 0 ? q0 : (j == 1 ? q1 : (j == 2 ? q2 : q3));
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            dst16[j * 4 + m] = kv_cache_unpack_i6(static_cast<std::uint8_t>((word >> (6 * m)) & 0x3Fu));
        }
    }
}

// Sixteen consecutive dimensions of an rk6v4-e8 key row as INT8 codes, the same four-int layout
// (byte m of word j is code 4j+m, dimension order) the int4 and E8-root expanders produce, so an
// rk6v4-e8 key row stages into the same shared layout, and is consumed by the same s8 MMA, as
// any other INT8-family key row.
__device__ __forceinline__ int4 kv_cache_i6_unpack_i8x16(const std::uint8_t* src12) {
    std::int8_t codes[16];
    kv_cache_unpack_i6x16(src12, codes);
    int out[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        unsigned word = 0;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            word |= static_cast<unsigned>(static_cast<std::uint8_t>(codes[j * 4 + m])) << (8 * m);
        }
        out[j] = static_cast<int>(word);
    }
    return make_int4(out[0], out[1], out[2], out[3]);
}

inline constexpr int kKVCacheLloyd4Levels = 8;

// Level boundaries in units of the group RMS: midpoints of the Lloyd-Max reconstruction points
// {0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6181, 2.0690, 2.7326}.
__device__ __forceinline__ int kv_cache_lloyd4_magnitude(float z) {
    const float a = fabsf(z);
    int m         = 0;
    m += a >= 0.25825f;
    m += a >= 0.52245f;
    m += a >= 0.79960f;
    m += a >= 1.09930f;
    m += a >= 1.43715f;
    m += a >= 1.84355f;
    m += a >= 2.40080f;
    return m;
}

// The INT8 code of each magnitude, packed four per word for the byte-permute expander.
inline constexpr unsigned kKVCacheLloyd4PosLo = 0x2C1F1206u; // 6, 18, 31, 44
inline constexpr unsigned kKVCacheLloyd4PosHi = 0x7F604B3Au; // 58, 75, 96, 127
inline constexpr unsigned kKVCacheLloyd4NegLo = 0xD4E1EEFAu; // -6, -18, -31, -44
inline constexpr unsigned kKVCacheLloyd4NegHi = 0x81A0B5C6u; // -58, -75, -96, -127

__host__ __device__ constexpr int kv_cache_lloyd4_code(int index) {
    constexpr int codes[kKVCacheLloyd4Levels] = {6, 18, 31, 44, 58, 75, 96, 127};
    return (index & 8) != 0 ? -codes[index & 7] : codes[index & 7];
}

// Expand four nibbles (the low 16 bits of `nibbles`, dimension order) into four INT8 codes.
// prmt reads its selector's bit 3 as sign-replicate, so the magnitude selects from a positive and
// a negative table and the sign bit, spread to a byte mask by a third prmt, picks between them.
__device__ __forceinline__ unsigned kv_cache_lloyd4_expand4(unsigned nibbles) {
    const unsigned magnitude = nibbles & 0x7777u;
    const unsigned pos       = __byte_perm(kKVCacheLloyd4PosLo, kKVCacheLloyd4PosHi, magnitude);
    const unsigned neg       = __byte_perm(kKVCacheLloyd4NegLo, kKVCacheLloyd4NegHi, magnitude);
    const unsigned mask      = __byte_perm(0u, 0xFF00u, ((nibbles >> 3) & 0x1111u) | 0x4444u);
    return (neg & mask) | (pos & ~mask);
}

// Sixteen consecutive dimensions: eight packed bytes in, sixteen INT8 codes out.
__device__ __forceinline__ int4 kv_cache_lloyd4_expand16(uint2 packed) {
    return make_int4(static_cast<int>(kv_cache_lloyd4_expand4(packed.x)),
                     static_cast<int>(kv_cache_lloyd4_expand4(packed.x >> 16)),
                     static_cast<int>(kv_cache_lloyd4_expand4(packed.y)),
                     static_cast<int>(kv_cache_lloyd4_expand4(packed.y >> 16)));
}

struct KVCacheLloyd4Encoded {
    int index0;
    int index1;
    __half scale;
};

// Encode one G64 group held by a full warp, two dimensions per lane (the append kernels' d0/d1
// lane assignment). Every lane receives its two indices and the group's represented scale.
//
// Every operation is spelled with an explicit rounding intrinsic and the reductions are xor
// butterflies, so the result is a fixed function of the 64 inputs that a host oracle reproduces
// exactly: nvcc would otherwise be free to contract the products into FMAs.
__device__ __forceinline__ KVCacheLloyd4Encoded kv_cache_lloyd4_encode_group(float x0, float x1) {
    constexpr unsigned FullMask = 0xffffffffu;
    float sum_sq                = __fadd_rn(__fmul_rn(x0, x0), __fmul_rn(x1, x1));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(FullMask, sum_sq, offset));
    }
    const float rms     = __fsqrt_rn(__fmul_rn(sum_sq, 1.0f / 64.0f));
    const float inv_rms = rms > 0.0f ? __fdiv_rn(1.0f, rms) : 0.0f;
    const int index0    = (x0 < 0.0f ? 8 : 0) | kv_cache_lloyd4_magnitude(__fmul_rn(x0, inv_rms));
    const int index1    = (x1 < 0.0f ? 8 : 0) | kv_cache_lloyd4_magnitude(__fmul_rn(x1, inv_rms));
    const float c0      = static_cast<float>(kv_cache_lloyd4_code(index0));
    const float c1      = static_cast<float>(kv_cache_lloyd4_code(index1));
    float xc            = __fadd_rn(__fmul_rn(x0, c0), __fmul_rn(x1, c1));
    float cc            = __fadd_rn(__fmul_rn(c0, c0), __fmul_rn(c1, c1));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        xc = __fadd_rn(xc, __shfl_xor_sync(FullMask, xc, offset));
        cc = __fadd_rn(cc, __shfl_xor_sync(FullMask, cc, offset));
    }
    return {index0, index1, __float2half_rn(rms > 0.0f ? __fdiv_rn(xc, cc) : 0.0f)};
}

__device__ __forceinline__ int4 kv_cache_int8_dequant_i8x8_from(const std::int8_t* codes8,
                                                                float s) {
    const int2 raw       = load_vec<int2>(codes8);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(c[2 * i]) * s;
        const float x1 = static_cast<float>(c[2 * i + 1]) * s;
        packed[i]      = pack_bf16x2(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// FP16 counterparts of the two loaders above.
//
// A kernel that stages V for an f16 PV MMA needs f16 codes, and the bf16 loaders cannot serve it:
// both plane types are sixteen bits wide, so storing a bf16 result through a __half* compiles,
// runs, and silently reinterprets every value. That is not hypothetical -- it is what upstream's
// int8 small-T kernel does today, and it is why the whole INT8 family (int8-g64 and rk8v4 alike,
// since both go through these two helpers) comes out 3-11x the reference with a ratio that varies
// per value. Nothing else in the pipeline can catch it: the sizes match, so no allocation, no
// dtype validation and no compiler diagnostic fires.
//
// int8 codes carry at most eight significant bits and a scale, so f16's narrower exponent is not a
// constraint here while its wider mantissa is a small gain on the P*V product.
__device__ __forceinline__ int4 kv_cache_int4_dequant_f16x8_from(const std::uint8_t* packed4,
                                                                 float s) {
    const unsigned raw        = load_vec<unsigned>(packed4);
    const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(&raw);
    unsigned out[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(kv_cache_int4_unpack(bytes[i], 0)) * s;
        const float x1 = static_cast<float>(kv_cache_int4_unpack(bytes[i], 1)) * s;
        out[i]         = pack_f16x2(x0, x1);
    }
    return make_int4(static_cast<int>(out[0]), static_cast<int>(out[1]), static_cast<int>(out[2]),
                     static_cast<int>(out[3]));
}

__device__ __forceinline__ int4 kv_cache_int8_dequant_f16x8_from(const std::int8_t* codes8,
                                                                 float s) {
    const int2 raw       = load_vec<int2>(codes8);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(c[2 * i]) * s;
        const float x1 = static_cast<float>(c[2 * i + 1]) * s;
        packed[i]      = pack_f16x2(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// The key coding of an INT8-family cache: INT8 G64 codes (int8, rk8v4), 4-bit Lloyd-Max indices
// (rk4v4), E8-snapped signed int4 codes (rk4v4-e8), E8 root codes (rk2v4-e8) or packed signed
// 6-bit E8 codes (rk6v4-e8). The packed codings share U8 key planes, so kernels are told which
// one they read by this and not by the plane's dtype.
enum class KvKeyCoding : int { Int8, Lloyd4, Int4E8, RootE8, K6E8 };

// The packed bytes of sixteen consecutive key dimensions: eight for the 4-bit codings, four for
// rk2v4-e8's two E8 blocks, twelve for rk6v4-e8's four 24-bit quads (three aligned u32).
struct KvCacheI6KeyChunk {
    std::uint32_t word0;
    std::uint32_t word1;
    std::uint32_t word2;
};

template <KvKeyCoding Keys>
using KvPackedKeyChunk = std::conditional_t<Keys == KvKeyCoding::RootE8, std::uint32_t,
                              std::conditional_t<Keys == KvKeyCoding::K6E8, KvCacheI6KeyChunk,
                                                uint2>>;

// Byte offset of the chunk holding key dimensions [d, d+16) of one token and KV head.
template <typename Geometry, KvKeyCoding Keys>
__device__ __forceinline__ std::int64_t
kv_cache_packed_key_chunk_index(int physical_page, int kv_head, int d, int page_offset) {
    if constexpr (Keys == KvKeyCoding::RootE8) {
        return kv_cache_e8_root_code_index<Geometry>(physical_page, kv_head, d >> 2, page_offset);
    } else if constexpr (Keys == KvKeyCoding::K6E8) {
        // d must be a multiple of 16 so the 3/4 byte offset stays 4-byte aligned.
        return paged_kv_page_head_offset<kKVCacheI6HeadExtent, Geometry::KVHeads>(physical_page,
                                                                                  kv_head) +
               static_cast<std::int64_t>(kKVCacheI6HeadExtent) * page_offset +
               (static_cast<std::int64_t>(d) * 3) / 4;
    } else {
        return kv_cache_int4_value_code_index<Geometry>(physical_page, kv_head, d >> 1,
                                                        page_offset);
    }
}

// Sixteen consecutive dimensions of a packed key row as INT8 codes.
template <KvKeyCoding Keys>
__device__ __forceinline__ int4 kv_cache_packed_key_expand16(KvPackedKeyChunk<Keys> packed) {
    static_assert(Keys != KvKeyCoding::Int8, "INT8 keys are stored expanded");
    if constexpr (Keys == KvKeyCoding::Lloyd4) {
        return kv_cache_lloyd4_expand16(packed);
    } else if constexpr (Keys == KvKeyCoding::Int4E8) {
        return kv_cache_int4_unpack_i8x16(packed);
    } else if constexpr (Keys == KvKeyCoding::K6E8) {
        return kv_cache_i6_unpack_i8x16(reinterpret_cast<const std::uint8_t*>(&packed));
    } else {
        return kv_cache_e8_root_unpack_i8x16(packed);
    }
}

// Write one rotated G64 key group held by a full warp (lane owns d0 = 64g+lane and d1 = d0+32),
// shared by the standalone append kernels and the small-T kernel's fused append. The packed
// codings store two 4-bit codes per byte, low nibble = even dimension, so a byte spans lanes l and
// l^1 exactly as the packed value plane does: rk4v4 its Lloyd-Max indices, rk4v4-e8 its E8-snapped
// int4 codes over the absmax/7 scale. rk2v4-e8 stores two bytes per eight dimensions over the same
// scale (e8_root_codec.cuh). Otherwise the key is INT8 G64. All keep one FP16 scale per group in
// the same scale plane.
template <typename Geometry, KvKeyCoding Keys>
__device__ __forceinline__ void
kv_cache_i8_family_store_key_group(std::int8_t* cache_k, __half* scale_k, int page, int kv_head,
                                   int group, int page_off, int lane, float k0, float k1) {
    constexpr unsigned FullMask = 0xffffffffu;
    __half scale;
    if constexpr (Keys == KvKeyCoding::RootE8) {
        const float k_abs  = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
        const auto k_quant = kv_cache_int4_quant_params(k_abs);
        kv_cache_e8_root_store_key_group(
            reinterpret_cast<std::uint8_t*>(cache_k),
            kv_cache_e8_root_code_index<Geometry>(page, kv_head, 0, page_off), group, k0, k1,
            __half2float(k_quant.scale), lane);
        scale = k_quant.scale;
    } else if constexpr (Keys == KvKeyCoding::Int4E8) {
        const float k_abs  = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
        const auto k_quant = kv_cache_int4_quant_params(k_abs);
        const std::int8_t c0 = kv_cache_int4_e8_key_code(k0, k_quant.inverse_scale, lane);
        const std::int8_t c1 = kv_cache_int4_e8_key_code(k1, k_quant.inverse_scale, lane);
        const int partner0   = __shfl_xor_sync(FullMask, static_cast<int>(c0), 1);
        const int partner1   = __shfl_xor_sync(FullMask, static_cast<int>(c1), 1);
        if ((lane & 1) == 0) {
            auto* packed_k                 = reinterpret_cast<std::uint8_t*>(cache_k);
            const std::int64_t packed_base = kv_cache_int4_value_code_index<Geometry>(
                page, kv_head, group * (kKVCacheInt8Group / 2), page_off);
            packed_k[packed_base + (lane >> 1)] =
                kv_cache_int4_pack(c0, static_cast<std::int8_t>(partner0));
            packed_k[packed_base + (lane >> 1) + 16] =
                kv_cache_int4_pack(c1, static_cast<std::int8_t>(partner1));
        }
        scale = k_quant.scale;
    } else if constexpr (Keys == KvKeyCoding::K6E8) {
        const float k_abs = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
        const __half ksh  = __float2half_rn(k_abs > 0.0f ? k_abs / 31.0f : 0.0f);
        const float ks    = __half2float(ksh);
        const float kinv  = ks > 0.0f ? 1.0f / ks : 0.0f;
        // The E8 lattice snap and rounding to 6-bit codes; the same deliberate half-coset
        // approximation as the rk4v4-e8 path, since no coset bit exists in the packed codes.
        const float p0 = kv_cache_e8_nearest(__fmul_rn(k0, kinv), lane);
        const float p1 = kv_cache_e8_nearest(__fmul_rn(k1, kinv), lane);
        const std::uint8_t c0 = kv_cache_i6_code_from_int(static_cast<int>(rintf(p0)));
        const std::uint8_t c1 = kv_cache_i6_code_from_int(static_cast<int>(rintf(p1)));
        auto* k_bytes = reinterpret_cast<std::uint8_t*>(cache_k);
        const std::int64_t k_row =
            paged_kv_page_head_offset<kKVCacheI6HeadExtent, Geometry::KVHeads>(page, kv_head) +
            static_cast<std::int64_t>(page_off) * kKVCacheI6HeadExtent + group * 48;
        kv_cache_lanes_write_i6_quad(c0, c1, lane, FullMask, k_row, k_bytes);
        scale = ksh;
    } else if constexpr (Keys == KvKeyCoding::Lloyd4) {
        const auto encoded = kv_cache_lloyd4_encode_group(k0, k1);
        const int partner0 = __shfl_xor_sync(FullMask, encoded.index0, 1);
        const int partner1 = __shfl_xor_sync(FullMask, encoded.index1, 1);
        if ((lane & 1) == 0) {
            auto* packed_k                 = reinterpret_cast<std::uint8_t*>(cache_k);
            const std::int64_t packed_base = kv_cache_int4_value_code_index<Geometry>(
                page, kv_head, group * (kKVCacheInt8Group / 2), page_off);
            packed_k[packed_base + (lane >> 1)] =
                static_cast<std::uint8_t>(encoded.index0 | (partner0 << 4));
            packed_k[packed_base + (lane >> 1) + 16] =
                static_cast<std::uint8_t>(encoded.index1 | (partner1 << 4));
        }
        scale = encoded.scale;
    } else {
        const float k_abs            = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
        const auto k_quant           = kv_cache_int8_quant_params(k_abs);
        const std::int64_t code_base = kv_cache_int8_quant_code_index<Geometry>(
            page, kv_head, group * kKVCacheInt8Group, page_off);
        cache_k[code_base + lane]      = kv_cache_int8_quant_code(k0, k_quant.inverse_scale);
        cache_k[code_base + lane + 32] = kv_cache_int8_quant_code(k1, k_quant.inverse_scale);
        scale                          = k_quant.scale;
    }
    if (lane == 0) {
        scale_k[kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, group, page_off)] = scale;
    }
}

} // namespace ninfer::ops
