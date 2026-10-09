// Hard gate on the production rk6v4-e8 key packing: kv_cache_pack_i6_quad /
// kv_cache_unpack_i6x16 must invert each other over every code value at every position in the
// 16-dim block, plus a randomized block sweep. This is the check the 2026-09 high-byte bug in the
// 4090 repo escaped (its cosine bench decoded with its own reference arithmetic and could not see
// a byte-layout slip in the production unpacker), so the gate runs the production pair itself.
// Any bit mismatch is a failure.

#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

using namespace ninfer::test;

constexpr int kSweepBlocks = 16 * 64; // every code value at every position
constexpr int kRandBlocks  = 4096;

__global__ void i6_roundtrip_check_kernel(const std::uint8_t* d_codes, std::uint8_t* d_bad,
                                          int blocks) {
    const int b = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (b >= blocks) { return; }
    const std::uint8_t* src = d_codes + static_cast<std::size_t>(b) * 16;
    std::uint8_t packed[12];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        ninfer::ops::kv_cache_pack_i6_quad(&src[4 * j], &packed[3 * j]);
    }
    std::int8_t dec[16];
    ninfer::ops::kv_cache_unpack_i6x16(packed, dec);
    int bad = 0;
    #pragma unroll
    for (int m = 0; m < 16; ++m) {
        const int expected = (static_cast<int>(src[m]) ^ 32) - 32;
        if (static_cast<int>(dec[m]) != expected) { ++bad; }
    }
    d_bad[b] = static_cast<std::uint8_t>(bad);
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) { return 77; }
        const int blocks = kSweepBlocks + kRandBlocks;
        std::vector<std::uint8_t> codes(static_cast<std::size_t>(blocks) * 16);
        for (int p = 0; p < 16; ++p) {
            for (int v = 0; v < 64; ++v) {
                std::uint8_t* row = &codes[static_cast<std::size_t>(p * 64 + v) * 16];
                for (int m = 0; m < 16; ++m) {
                    row[m] =
                        static_cast<std::uint8_t>(v ^ ((p + m) & 1)); // every value, all positions
                }
            }
        }
        std::mt19937 rng(0x66e8u);
        for (int b = kSweepBlocks; b < blocks; ++b) {
            std::uint8_t* row = &codes[static_cast<std::size_t>(b) * 16];
            for (int m = 0; m < 16; ++m) { row[m] = static_cast<std::uint8_t>(rng() & 0x3Fu); }
        }

        std::uint8_t* d_codes = nullptr;
        std::uint8_t* d_bad   = nullptr;
        cuda_check(cudaMalloc(&d_codes, codes.size()), "cudaMalloc codes");
        cuda_check(cudaMalloc(&d_bad, blocks), "cudaMalloc bad");
        cuda_check(cudaMemcpy(d_codes, codes.data(), codes.size(), cudaMemcpyHostToDevice),
                   "cudaMemcpy codes");
        const int threads = 256;
        i6_roundtrip_check_kernel<<<(blocks + threads - 1) / threads, threads>>>(d_codes, d_bad,
                                                                                blocks);
        cuda_check_last_launch("i6_roundtrip_check_kernel");
        std::vector<std::uint8_t> bad(blocks);
        cuda_check(cudaMemcpy(bad.data(), d_bad, blocks, cudaMemcpyDeviceToHost),
                   "cudaMemcpy bad");
        cuda_check(cudaFree(d_codes), "cudaFree codes");
        cuda_check(cudaFree(d_bad), "cudaFree bad");
        std::size_t failures = 0;
        for (int b = 0; b < blocks; ++b) {
            if (bad[b] != 0) {
                if (++failures <= 8) {
                    std::cerr << "i6 round-trip mismatch in block " << b << ": " << static_cast<int>(bad[b])
                              << " of 16 dims decode wrong\n";
                }
            }
        }
        if (failures != 0) {
            std::cerr << failures << " i6 pack/unpack round-trip blocks mismatch\n";
            return 1;
        }
        std::cout << "i6 round-trip: PASS (" << blocks << " blocks, production pack/unpack)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
