#pragma once

#include "artifact/framing.h"

#include <filesystem>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp {

using ExpertRouteCounts = std::vector<std::vector<std::uint64_t>>;

// Counts address local text layers followed by the optional MTP layer. An artifact identity and
// exact bank widths prevent a profile from silently ranking another model's experts.
ExpertRouteCounts read_expert_profile(const std::filesystem::path& path,
    const artifact::ArtifactId& artifact, std::span<const std::size_t> experts);
void write_expert_profile(const std::filesystem::path& path,
    const artifact::ArtifactId& artifact, const ExpertRouteCounts& counts);

} // namespace ninfer::models::qwen4_exp
