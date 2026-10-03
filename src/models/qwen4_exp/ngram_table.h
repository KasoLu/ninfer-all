#pragma once

// Row reads from the Qwen3.8-Flash-Next n-gram table, which by default stays on disk: rows are
// read where the file stores them, through the OS page cache, as the hash addresses them. The
// RAM residency loads the whole payload once and serves rows from memory.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp {

enum class NgramResidency : std::uint8_t { Disk, Ram };

struct NgramTableLayout {
    std::filesystem::path path;
    std::uint64_t payload_offset = 0; // byte offset of row 0
    std::uint32_t row_bytes      = 0;
    std::uint64_t rows           = 0; // addressable rows
};

class NgramTableReader {
  public:
    // Opens the file and checks that it holds every row; Ram reads the payload now.
    NgramTableReader(NgramTableLayout layout, NgramResidency residency);
    ~NgramTableReader();
    NgramTableReader(const NgramTableReader&)            = delete;
    NgramTableReader& operator=(const NgramTableReader&) = delete;

    [[nodiscard]] const NgramTableLayout& layout() const noexcept { return layout_; }
    [[nodiscard]] NgramResidency residency() const noexcept { return residency_; }

    // Copies the rows `row_ids` addresses, in order, into `out` (row_ids.size() * row_bytes).
    // A row id past the table throws std::out_of_range; a failed read throws std::runtime_error.
    void read_rows(std::span<const std::uint64_t> row_ids, std::span<std::uint8_t> out) const;

  private:
    struct File;
    NgramTableLayout layout_;
    NgramResidency residency_;
    std::unique_ptr<File> file_;
    std::vector<std::uint8_t> resident_;
};

} // namespace ninfer::models::qwen4_exp
