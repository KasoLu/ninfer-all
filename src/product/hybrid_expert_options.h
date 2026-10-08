#pragma once

#include "ninfer/types.h"

#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

inline constexpr const char* kHybridExpertHelp =
    "  --expert-dma-share F          native host experts: GPU share of cache misses,\n"
    "                                0..1 (default 1); below 1 enables experimental CPU mixing\n"
    "                                prefill always uses the GPU\n"
    "  --expert-cpu-threads N        CPU workers, 1..256 (default automatic)\n"
    "  --expert-cache-adaptive       replace cold cached experts between calls;\n"
    "                                changes the CPU/GPU arithmetic partition\n"
    "  --expert-profile FILE         fill the cache from recorded expert counts\n"
    "  --expert-profile-out FILE     record counts for this artifact after requests\n";

template<class Value>
bool parse_hybrid_expert_option(std::string_view option, HybridExpertOptions& out, Value&& value) {
    if (option == "--expert-dma-share") {
        const std::string text(value());
        float share = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), share);
        if (error != std::errc{} || end != text.data() + text.size() ||
            !std::isfinite(share) || share < 0 || share > 1) {
            throw std::invalid_argument("--expert-dma-share takes a finite value in 0..1");
        }
        out.dma_share = share;
    } else if (option == "--expert-cpu-threads") {
        const std::string text(value());
        std::uint32_t threads = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), threads);
        if (error != std::errc{} || end != text.data() + text.size() || threads == 0 || threads > 256) {
            throw std::invalid_argument("--expert-cpu-threads takes an integer in 1..256");
        }
        out.cpu_threads = threads;
    } else if (option == "--expert-cache-adaptive") {
        out.adaptive_cache = true;
    } else if (option == "--expert-profile" || option == "--expert-profile-out") {
        const std::string path(value());
        if (path.empty()) { throw std::invalid_argument(std::string(option) + " takes a file path"); }
        (option == "--expert-profile" ? out.routing_profile : out.record_profile) = path;
    } else { return false; }
    return true;
}

} // namespace ninfer::product
