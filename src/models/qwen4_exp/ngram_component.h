#pragma once

// The n-gram table of a Qwen3.8-Flash-Next model is its artifact's `ngram` component: the table
// and the hash constants its rows were written for. The rows stay where the artifact stores them,
// and nothing reads them at load; NgramTableReader reads the rows each token addresses from there,
// or loads the whole table into RAM.

#include "artifact/reader.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/ngram_table.h"
#include "ninfer/ops/ngram_rows.h"

#include <filesystem>

namespace ninfer::models::qwen4_exp {

struct NgramTableSource {
    NgramTableLayout layout;
    ops::NgramRowFormat format = ops::NgramRowFormat::Bf16;
};

// Locates the table of the artifact `reader` opened at `artifact` and refuses one whose constants
// differ from those `config` derives.
[[nodiscard]] NgramTableSource ngram_table_source(const artifact::Reader& reader,
                                                  const std::filesystem::path& artifact,
                                                  const TextConfig& config);

} // namespace ninfer::models::qwen4_exp
