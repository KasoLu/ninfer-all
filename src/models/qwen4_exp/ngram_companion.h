#pragma once

// The n-gram table of a Qwen3.8-Flash-Next model ships as a companion artifact (tools/convert
// --components ngram): one v3 file set whose text component is the table and the hash constants
// its rows were written for, shared by every quantization of the model. Its rows stay where the
// files store them; NgramTableReader reads them from there or loads them into RAM.

#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/ngram_table.h"
#include "ninfer/ops/ngram_rows.h"

#include <filesystem>
#include <optional>

namespace ninfer::models::qwen4_exp {

struct NgramCompanion {
    NgramTableLayout layout;
    ops::NgramRowFormat format = ops::NgramRowFormat::Bf16;
};

// Opens the companion at `path` and refuses one whose constants differ from those `config` derives.
[[nodiscard]] NgramCompanion open_ngram_companion(const std::filesystem::path& path,
                                                  const TextConfig& config);

// The first companion in `directory` (other than `exclude`) that matches `config`, if any.
[[nodiscard]] std::optional<std::filesystem::path>
find_ngram_companion(const std::filesystem::path& directory, const std::filesystem::path& exclude,
                     const TextConfig& config);

} // namespace ninfer::models::qwen4_exp
