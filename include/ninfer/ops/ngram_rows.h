#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>

namespace ninfer::ops {

// Stored row encodings of the Qwen3.8-Flash-Next n-gram table (one row is one hash head's
// 160-wide slice of the embedding).
enum class NgramRowFormat : std::uint8_t {
    // 160 BF16 values, 320 bytes: the checkpoint's own encoding (reference and oracle).
    Bf16,
    // 160 FP8 E4M3 codes followed by one FP16 scale, 162 bytes: value = e4m3(code) * scale.
    Fp8E4M3RowScale,
};

[[nodiscard]] std::uint32_t ngram_row_bytes(NgramRowFormat format);

/**
 * Decodes staged n-gram rows into the PLE embedding. `rows` is U8 [row_bytes, heads * tokens],
 * the rows of token t in head order at columns t * heads .. t * heads + heads - 1, as the hash
 * addressed them; `embedding` is BF16 [heads * 160, tokens], head h of token t at
 * [h * 160, h * 160 + 160). The oracle decodes each stored row exactly (an FP8 code times its
 * FP16 scale in FP64) and the output is compared after its BF16 store: BF16 rows are copied
 * bit-exactly, FP8 rows round once to nearest even. No workspace or state.
 */
void ngram_embed_rows(const Tensor& rows, NgramRowFormat format, std::int32_t heads,
                      Tensor& embedding, cudaStream_t stream);

} // namespace ninfer::ops
