#include "models/qwen4_exp/ngram_component.h"

#include "models/qwen4_exp/ngram_hash.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen4_exp {
namespace {

using artifact::ArtifactError;
using artifact::Json;

std::vector<std::uint64_t> u64_array(const Json& config, const char* name) {
    const auto& value = config.at(name);
    if (!value.is_array()) { throw ArtifactError(std::string(name) + " must be an array"); }
    std::vector<std::uint64_t> out;
    for (const auto& item : value) { out.push_back(artifact::require_u64(item, name)); }
    return out;
}

void require_equal(std::uint64_t actual, std::uint64_t expected, const char* name) {
    if (actual != expected) {
        throw ArtifactError(std::string("the n-gram table's ") + name + " is " +
                            std::to_string(actual) + "; the model needs " +
                            std::to_string(expected));
    }
}

ops::NgramRowFormat row_format(const std::string& format) {
    if (format == "gguf_iq4_nl") { return ops::NgramRowFormat::Iq4Nl; }
    if (format == "bf16") { return ops::NgramRowFormat::Bf16; }
    throw ArtifactError("the n-gram table is stored as " + format +
                        "; the runtime decodes gguf_iq4_nl and bf16 rows");
}

} // namespace

NgramTableSource ngram_table_source(const artifact::Reader& reader,
                                    const std::filesystem::path& artifact,
                                    const TextConfig& config) {
    const auto& directory = reader.directory();
    if (!directory.components.contains("ngram")) {
        throw ArtifactError(artifact.string() +
                            " carries no n-gram table (its `ngram` component); convert the "
                            "model with its table source, --source ngram=PATH");
    }
    const Json& c = directory.component("ngram").config;
    artifact::require_members(c,
                              {"architectures", "model_type", "vocab_size", "eos_token_id",
                               "ngram_size", "heads_per_ngram", "row_width", "rows", "multipliers",
                               "head_vocab", "head_offset"},
                              {}, "n-gram table config");
    if (!c.at("architectures").is_array() || c.at("architectures").size() != 1 ||
        c.at("architectures")[0] != "Qwen4ExpNgramTable") {
        throw ArtifactError(artifact.string() +
                            ": the `ngram` component is not a Qwen3.8-Flash-Next n-gram table");
    }
    const NgramHashConstants expected = derive_ngram_hash_constants(config.ngram);
    require_equal(artifact::require_u64(c.at("vocab_size"), "vocab_size"), config.vocab_size,
                  "vocab_size");
    require_equal(artifact::require_u64(c.at("eos_token_id"), "eos_token_id"),
                  static_cast<std::uint64_t>(config.eos_token_id), "eos_token_id");
    require_equal(artifact::require_u64(c.at("ngram_size"), "ngram_size"), config.ngram.ngram_size,
                  "ngram_size");
    require_equal(artifact::require_u64(c.at("heads_per_ngram"), "heads_per_ngram"),
                  config.ngram.heads_per_ngram, "heads_per_ngram");
    const std::uint64_t width = config.ple_embed_dim / config.ngram_heads();
    require_equal(artifact::require_u64(c.at("row_width"), "row_width"), width, "row_width");
    require_equal(artifact::require_u64(c.at("rows"), "rows"), expected.rows, "rows");
    if (u64_array(c, "multipliers") != expected.multipliers ||
        u64_array(c, "head_vocab") != expected.head_vocab ||
        u64_array(c, "head_offset") != expected.head_offset) {
        throw ArtifactError(artifact.string() +
                            ": the n-gram table's hash constants differ from the model's");
    }
    const auto found = directory.bindings.find("ngram/table");
    if (found == directory.bindings.end() || found->second.parts.size() != 1) {
        throw ArtifactError(artifact.string() + ": the n-gram table must be one stored object");
    }
    const auto& part   = found->second.parts.front();
    const auto& tensor = directory.tensor(part.object);
    NgramTableSource out;
    out.format                      = row_format(tensor.format);
    out.layout.row_bytes            = ops::ngram_row_bytes(out.format);
    out.layout.rows                 = expected.rows;
    const std::uint64_t table_bytes = out.layout.rows * out.layout.row_bytes;
    if (tensor.shape != artifact::Shape{expected.rows, width} || part.begin != 0 ||
        tensor.bytes != table_bytes) {
        throw ArtifactError(artifact.string() + ": the n-gram table has an unexpected geometry");
    }
    for (const auto& segment : reader.segments(tensor.offset, table_bytes)) {
        const auto& record = directory.files.at(segment.file_index);
        const std::filesystem::path file =
            segment.file_index == 0 ? artifact : artifact.parent_path() / record.path.value();
        out.layout.segments.push_back({file, segment.file_offset, segment.bytes});
    }
    return out;
}

} // namespace ninfer::models::qwen4_exp
