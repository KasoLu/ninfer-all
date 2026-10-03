#include "models/qwen4_exp/ngram_table.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <cerrno>
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace ninfer::models::qwen4_exp {

struct NgramTableReader::File {
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int descriptor = -1;
#endif

    explicit File(const std::filesystem::path& path) {
#ifdef _WIN32
        // Random access: the cache manager does not read ahead past each row.
        handle = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                             FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("n-gram table: cannot open " + path.string());
        }
#else
        descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor < 0) {
            throw std::runtime_error("n-gram table: cannot open " + path.string() + ": " +
                                     std::strerror(errno));
        }
#    if defined(POSIX_FADV_RANDOM)
        (void)::posix_fadvise(descriptor, 0, 0, POSIX_FADV_RANDOM);
#    endif
#endif
    }

    ~File() {
#ifdef _WIN32
        if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
#else
        if (descriptor >= 0) ::close(descriptor);
#endif
    }

    [[nodiscard]] std::uint64_t size() const {
#ifdef _WIN32
        LARGE_INTEGER bytes{};
        if (!GetFileSizeEx(handle, &bytes)) throw std::runtime_error("n-gram table: cannot size");
        return static_cast<std::uint64_t>(bytes.QuadPart);
#else
        struct stat status {};
        if (::fstat(descriptor, &status) != 0) throw std::runtime_error("n-gram table: cannot size");
        return static_cast<std::uint64_t>(status.st_size);
#endif
    }

    // Reads exactly `bytes` at `offset`.
    void read(std::uint64_t offset, std::uint8_t* destination, std::size_t bytes) const {
        while (bytes > 0) {
#ifdef _WIN32
            OVERLAPPED position{};
            position.Offset     = static_cast<DWORD>(offset & 0xffffffffULL);
            position.OffsetHigh = static_cast<DWORD>(offset >> 32U);
            const DWORD request = static_cast<DWORD>(
                std::min<std::size_t>(bytes, std::numeric_limits<DWORD>::max() / 2));
            DWORD done = 0;
            if (!ReadFile(handle, destination, request, &done, &position) || done == 0) {
                throw std::runtime_error("n-gram table: read failed");
            }
#else
            const ssize_t done = ::pread(descriptor, destination, bytes, static_cast<off_t>(offset));
            if (done < 0 && errno == EINTR) continue;
            if (done <= 0) {
                throw std::runtime_error(std::string("n-gram table: read failed: ") +
                                         (done < 0 ? std::strerror(errno) : "end of file"));
            }
#endif
            destination += done;
            offset += static_cast<std::uint64_t>(done);
            bytes -= static_cast<std::size_t>(done);
        }
    }
};

NgramTableReader::NgramTableReader(NgramTableLayout layout, NgramResidency residency)
    : layout_(std::move(layout)), residency_(residency),
      file_(std::make_unique<File>(layout_.path)) {
    if (layout_.row_bytes == 0 || layout_.rows == 0) {
        throw std::invalid_argument("n-gram table: empty layout");
    }
    if (layout_.rows > (std::numeric_limits<std::uint64_t>::max() - layout_.payload_offset) /
                           layout_.row_bytes) {
        throw std::invalid_argument("n-gram table: payload size overflows");
    }
    const std::uint64_t end = layout_.payload_offset + layout_.rows * layout_.row_bytes;
    if (file_->size() < end) {
        throw std::runtime_error("n-gram table: " + layout_.path.string() + " holds " +
                                 std::to_string(file_->size()) + " bytes, the table needs " +
                                 std::to_string(end));
    }
    if (residency_ == NgramResidency::Ram) {
        const std::uint64_t bytes = layout_.rows * layout_.row_bytes;
        if (bytes > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("n-gram table: payload exceeds the address space");
        }
        resident_.resize(static_cast<std::size_t>(bytes));
        file_->read(layout_.payload_offset, resident_.data(), resident_.size());
    }
}

NgramTableReader::~NgramTableReader() = default;

void NgramTableReader::read_rows(std::span<const std::uint64_t> row_ids,
                                 std::span<std::uint8_t> out) const {
    const std::size_t row_bytes = layout_.row_bytes;
    if (out.size() != row_ids.size() * row_bytes) {
        throw std::invalid_argument("n-gram table: output size mismatch");
    }
    for (std::size_t i = 0; i < row_ids.size(); ++i) {
        const std::uint64_t row = row_ids[i];
        if (row >= layout_.rows) {
            throw std::out_of_range("n-gram table: row " + std::to_string(row) + " past " +
                                    std::to_string(layout_.rows));
        }
        std::uint8_t* destination = out.data() + i * row_bytes;
        if (residency_ == NgramResidency::Ram) {
            std::memcpy(destination, resident_.data() + row * row_bytes, row_bytes);
        } else {
            file_->read(layout_.payload_offset + row * row_bytes, destination, row_bytes);
        }
    }
}

} // namespace ninfer::models::qwen4_exp
