#include "models/qwen4_exp/ngram_table.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
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
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace ninfer::models::qwen4_exp {
namespace {

// Direct reads cover whole 4 KiB blocks at 4 KiB-aligned file offsets into 4 KiB-aligned memory.
constexpr std::uint64_t kAlign = 4096;
// A residency's load reads the table in pieces of about this size, several at once.
constexpr std::uint64_t kLoadChunk = 4ULL << 20;
// A load piece holding fewer hot rows than one per this many bytes reads them one by one.
constexpr std::uint64_t kSparseLoad = 8192;
// Direct row reads up to this size use a buffer each thread keeps.
constexpr std::uint64_t kThreadBounce = 64 * 1024;

std::string mib(std::uint64_t bytes) {
    return std::to_string((bytes + (1ULL << 19)) >> 20) + " MiB";
}

#ifndef _WIN32
std::string error_text(int error) { return std::strerror(error); }
#endif

struct AlignedFree {
    void operator()(std::uint8_t* p) const noexcept {
        ::operator delete(p, std::align_val_t{kAlign});
    }
};

using AlignedBytes = std::unique_ptr<std::uint8_t, AlignedFree>;

AlignedBytes aligned_bytes(std::uint64_t bytes) {
    return AlignedBytes(
        static_cast<std::uint8_t*>(::operator new(bytes, std::align_val_t{kAlign})));
}

// The bounce buffer of a direct row read on this thread.
std::uint8_t* thread_bounce() {
    thread_local AlignedBytes buffer = aligned_bytes(kThreadBounce);
    return buffer.get();
}

std::size_t latency_bucket(std::uint64_t nanoseconds) {
    const std::uint64_t micro = nanoseconds / 1000;
    return std::min<std::size_t>(std::bit_width(micro), NgramTableStats::kLatencyBuckets - 1);
}

} // namespace

struct NgramTableReader::File {
    NgramIo io;
    std::filesystem::path path;
    std::uint64_t bytes      = 0;       // the file's size
    const std::uint8_t* view = nullptr; // Mapped: the whole file
#ifdef _WIN32
    HANDLE handle  = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
#else
    int descriptor = -1;
#endif

    File(std::filesystem::path file, NgramIo mode) : io(mode), path(std::move(file)) {
#ifdef _WIN32
        // Random access: the cache manager does not read ahead past each row.
        DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS;
        if (io == NgramIo::Direct) { flags |= FILE_FLAG_NO_BUFFERING; }
        handle = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                             flags, nullptr);
        if (handle == INVALID_HANDLE_VALUE) { fail("cannot open"); }
        LARGE_INTEGER size{};
        if (!GetFileSizeEx(handle, &size)) { fail("cannot size"); }
        bytes = static_cast<std::uint64_t>(size.QuadPart);
        if (io == NgramIo::Mapped) {
            mapping = CreateFileMappingW(handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (mapping == nullptr) { fail("cannot map"); }
            view = static_cast<const std::uint8_t*>(MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0));
            if (view == nullptr) { fail("cannot map"); }
        }
#else
        int flags = O_RDONLY | O_CLOEXEC;
        if (io == NgramIo::Direct) { flags |= O_DIRECT; }
        descriptor = ::open(path.c_str(), flags);
        if (descriptor < 0) {
            const int error = errno;
            if (io == NgramIo::Direct && error == EINVAL) {
                throw std::runtime_error("n-gram table: " + path.string() +
                                         " is on a file system without direct I/O; use "
                                         "--ngram-io buffered or mapped");
            }
            throw std::runtime_error("n-gram table: cannot open " + path.string() + ": " +
                                     error_text(error));
        }
        struct stat status{};
        if (::fstat(descriptor, &status) != 0) { fail("cannot size"); }
        bytes = static_cast<std::uint64_t>(status.st_size);
#    if defined(POSIX_FADV_RANDOM)
        if (io != NgramIo::Direct) { (void)::posix_fadvise(descriptor, 0, 0, POSIX_FADV_RANDOM); }
#    endif
        if (io == NgramIo::Mapped && bytes > 0) {
            void* p = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, descriptor, 0);
            if (p == MAP_FAILED) { fail("cannot map"); }
            (void)::madvise(p, bytes, MADV_RANDOM);
            view = static_cast<const std::uint8_t*>(p);
        }
#endif
    }

    ~File() {
#ifdef _WIN32
        if (view != nullptr) UnmapViewOfFile(view);
        if (mapping != nullptr) CloseHandle(mapping);
        if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
#else
        if (view != nullptr) ::munmap(const_cast<std::uint8_t*>(view), bytes);
        if (descriptor >= 0) ::close(descriptor);
#endif
    }

    File(const File&)            = delete;
    File& operator=(const File&) = delete;

    [[noreturn]] void fail(const char* what) const {
#ifdef _WIN32
        throw std::runtime_error("n-gram table: " + std::string(what) + " " + path.string() +
                                 " (error " + std::to_string(GetLastError()) + ")");
#else
        throw std::runtime_error("n-gram table: " + std::string(what) + " " + path.string() + ": " +
                                 error_text(errno));
#endif
    }

    // One positioned read of up to `bytes` at `offset`; 0 at the end of the file.
    std::size_t read_some(std::uint64_t offset, std::uint8_t* destination,
                          std::size_t bytes) const {
        bytes = std::min<std::size_t>(bytes, 1U << 30U);
        for (;;) {
#ifdef _WIN32
            OVERLAPPED position{};
            position.Offset     = static_cast<DWORD>(offset & 0xffffffffULL);
            position.OffsetHigh = static_cast<DWORD>(offset >> 32U);
            DWORD done          = 0;
            if (!ReadFile(handle, destination, static_cast<DWORD>(bytes), &done, &position)) {
                if (GetLastError() == ERROR_HANDLE_EOF) { return 0; }
                fail("cannot read");
            }
            return done;
#else
            const ssize_t done =
                ::pread(descriptor, destination, bytes, static_cast<off_t>(offset));
            if (done < 0 && errno == EINTR) { continue; }
            if (done < 0) { fail("cannot read"); }
            return static_cast<std::size_t>(done);
#endif
        }
    }

    // Reads exactly `bytes` at `offset`, through the page cache or out of the mapping.
    void read_cached(std::uint64_t offset, std::uint8_t* destination, std::size_t bytes) const {
        if (view != nullptr) {
            std::memcpy(destination, view + offset, bytes);
            return;
        }
        while (bytes > 0) {
            const std::size_t done = read_some(offset, destination, bytes);
            if (done == 0) {
                throw std::runtime_error("n-gram table: " + path.string() + " ends early");
            }
            destination += done;
            offset += done;
            bytes -= done;
        }
    }

    // Reads exactly `bytes` at `offset` past the page cache: the 4 KiB blocks around them into an
    // aligned buffer, then the bytes out of it.
    void read_direct(std::uint64_t offset, std::uint8_t* destination, std::size_t bytes) const {
        const std::uint64_t begin = offset & ~(kAlign - 1);
        const std::uint64_t end   = (offset + bytes + kAlign - 1) & ~(kAlign - 1);
        AlignedBytes owned;
        std::uint8_t* bounce = nullptr;
        if (end - begin <= kThreadBounce) {
            bounce = thread_bounce();
        } else {
            owned  = aligned_bytes(end - begin);
            bounce = owned.get();
        }
        const std::uint64_t need = offset + bytes - begin;
        std::uint64_t have       = 0;
        while (have < need) {
            const std::size_t done = read_some(begin + have, bounce + have, end - begin - have);
            if (done == 0) {
                throw std::runtime_error("n-gram table: " + path.string() + " ends early");
            }
            have += done;
        }
        std::memcpy(destination, bounce + (offset - begin), bytes);
    }

    void read(std::uint64_t offset, std::uint8_t* destination, std::size_t bytes) const {
        if (io == NgramIo::Direct) {
            read_direct(offset, destination, bytes);
        } else {
            read_cached(offset, destination, bytes);
        }
    }
};

// Anonymous memory for the resident rows: 2 MiB pages where Linux offers them, and lockable.
class NgramTableReader::Memory {
public:
    explicit Memory(std::uint64_t bytes) : bytes_(bytes) {
        if (bytes > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("n-gram table: " + mib(bytes) + " exceed the address space");
        }
#ifdef _WIN32
        data_ = static_cast<std::uint8_t*>(
            VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (data_ == nullptr) {
            throw std::runtime_error("n-gram table: cannot allocate " + mib(bytes) +
                                     " of RAM (error " + std::to_string(GetLastError()) + ")");
        }
#else
        void* p =
            ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            throw std::runtime_error("n-gram table: cannot allocate " + mib(bytes) +
                                     " of RAM: " + error_text(errno));
        }
#    if defined(MADV_HUGEPAGE)
        (void)::madvise(p, bytes, MADV_HUGEPAGE);
#    endif
        data_ = static_cast<std::uint8_t*>(p);
#endif
    }

    ~Memory() {
#ifdef _WIN32
        if (locked_) VirtualUnlock(data_, bytes_);
        VirtualFree(data_, 0, MEM_RELEASE);
#else
        if (locked_) ::munlock(data_, bytes_);
        ::munmap(data_, bytes_);
#endif
    }

    Memory(const Memory&)            = delete;
    Memory& operator=(const Memory&) = delete;

    [[nodiscard]] std::uint8_t* data() const noexcept { return data_; }

    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }

    void lock() {
#ifdef _WIN32
        // VirtualLock keeps pages in the working set, whose minimum must make room for them.
        const HANDLE process = GetCurrentProcess();
        SIZE_T minimum = 0, maximum = 0;
        if (!GetProcessWorkingSetSize(process, &minimum, &maximum) ||
            !SetProcessWorkingSetSize(process, minimum + bytes_, maximum + bytes_) ||
            !VirtualLock(data_, bytes_)) {
            throw std::runtime_error("n-gram table: cannot lock " + mib(bytes_) +
                                     " in RAM (error " + std::to_string(GetLastError()) + ")");
        }
#else
        if (::mlock(data_, bytes_) != 0) {
            throw std::runtime_error("n-gram table: cannot lock " + mib(bytes_) +
                                     " in RAM: " + error_text(errno) +
                                     " (the memlock limit, ulimit -l, or CAP_IPC_LOCK)");
        }
#endif
        locked_ = true;
    }

private:
    std::uint8_t* data_ = nullptr;
    std::uint64_t bytes_;
    bool locked_ = false;
};

NgramTableReader::NgramTableReader(NgramTableLayout layout, const NgramReadOptions& options)
    : layout_(std::move(layout)), options_(options) {
    if (layout_.row_bytes == 0 || layout_.rows == 0 || layout_.segments.empty()) {
        throw std::invalid_argument("n-gram table: empty layout");
    }
    if (layout_.rows > std::numeric_limits<std::uint64_t>::max() / layout_.row_bytes) {
        throw std::invalid_argument("n-gram table: payload size overflows");
    }
    if (options_.depth == 0 || options_.depth > 1024) {
        throw std::invalid_argument("n-gram table: 1..1024 reads in flight");
    }
    if (options_.lock && options_.residency == NgramResidency::Disk) {
        throw std::invalid_argument("n-gram table: the disk residency keeps no rows to lock");
    }
    const std::uint64_t table_bytes = layout_.rows * layout_.row_bytes;
    std::uint64_t covered           = 0;
    for (const auto& segment : layout_.segments) {
        files_.push_back(std::make_unique<File>(segment.path, options_.io));
        if (segment.bytes == 0 ||
            segment.file_offset > std::numeric_limits<std::uint64_t>::max() - segment.bytes ||
            files_.back()->bytes < segment.file_offset + segment.bytes) {
            throw std::runtime_error("n-gram table: " + segment.path.string() + " holds " +
                                     std::to_string(files_.back()->bytes) +
                                     " bytes, short of its table segment");
        }
        starts_.push_back(covered);
        covered += segment.bytes;
    }
    if (covered < table_bytes) {
        throw std::runtime_error("n-gram table: the files hold " + std::to_string(covered) +
                                 " table bytes, the table needs " + std::to_string(table_bytes));
    }
    pool_ = std::make_unique<ReadPool>(options_.depth);
    if (options_.residency == NgramResidency::Ram) {
        load_resident();
    } else if (options_.residency == NgramResidency::RamHot) {
        load_hot();
    }
    // The profile has served its purpose.
    options_.hot_rows = {};
    options_.hot_rows.shrink_to_fit();
}

NgramTableReader::~NgramTableReader() {
    try {
        wait();
    } catch (...) {}
}

std::uint64_t NgramTableReader::resident_bytes() const noexcept {
    return (resident_ ? resident_->bytes() : 0) + hot_bits_.size() * 8 + hot_blocks_.size() * 4;
}

void NgramTableReader::read(std::uint64_t offset, std::uint8_t* destination,
                            std::size_t bytes) const {
    auto segment = static_cast<std::size_t>(
        std::upper_bound(starts_.begin(), starts_.end(), offset) - starts_.begin() - 1);
    while (bytes > 0) {
        const auto& layout        = layout_.segments[segment];
        const std::uint64_t local = offset - starts_[segment];
        const std::size_t count =
            static_cast<std::size_t>(std::min<std::uint64_t>(bytes, layout.bytes - local));
        files_[segment]->read(layout.file_offset + local, destination, count);
        destination += count;
        offset += count;
        bytes -= count;
        ++segment;
    }
}

void NgramTableReader::load(std::uint64_t first, std::uint64_t count,
                            std::uint8_t* destination) const {
    const std::uint64_t rb    = layout_.row_bytes;
    const std::uint64_t piece = std::max<std::uint64_t>(1, kLoadChunk / rb);
    pool_->run(static_cast<std::size_t>((count + piece - 1) / piece), [&](std::size_t i) {
        const std::uint64_t row = first + i * piece;
        const std::uint64_t n   = std::min(piece, first + count - row);
        read(row * rb, destination + (row - first) * rb, static_cast<std::size_t>(n * rb));
    });
}

void NgramTableReader::load_resident() {
    resident_ = std::make_unique<Memory>(layout_.rows * layout_.row_bytes);
    load(0, layout_.rows, resident_->data());
    if (options_.lock) { resident_->lock(); }
    resident_rows_ = layout_.rows;
}

void NgramTableReader::load_hot() {
    const std::uint64_t rows   = layout_.rows;
    const std::uint64_t rb     = layout_.row_bytes;
    const std::uint64_t words  = (rows + 63) / 64;
    const std::uint64_t blocks = (rows + 511) / 512;
    const std::uint64_t index  = words * 8 + blocks * 4;
    if (options_.budget_bytes < index + rb) {
        throw std::invalid_argument("n-gram table: a RAM budget of " + mib(options_.budget_bytes) +
                                    " does not cover the hot rows' index of " + mib(index));
    }
    if (rows > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("n-gram table: a hot-row profile addresses 2^32 rows");
    }
    const std::uint64_t fit = (options_.budget_bytes - index) / rb;
    const std::size_t take =
        static_cast<std::size_t>(std::min<std::uint64_t>(options_.hot_rows.size(), fit));
    hot_bits_.assign(static_cast<std::size_t>(words), 0);
    for (std::size_t i = 0; i < take; ++i) {
        const std::uint64_t row = options_.hot_rows[i];
        if (row >= rows) {
            throw std::invalid_argument("n-gram table: the hot-row profile names row " +
                                        std::to_string(row) + " past the table");
        }
        std::uint64_t& word     = hot_bits_[row >> 6U];
        const std::uint64_t bit = 1ULL << (row & 63U);
        if ((word & bit) != 0) {
            throw std::invalid_argument("n-gram table: the hot-row profile names row " +
                                        std::to_string(row) + " twice");
        }
        word |= bit;
    }
    hot_blocks_.resize(static_cast<std::size_t>(blocks));
    std::uint64_t running = 0;
    for (std::uint64_t b = 0; b < blocks; ++b) {
        hot_blocks_[b] = static_cast<std::uint32_t>(running);
        for (std::uint64_t w = b * 8; w < std::min(words, b * 8 + 8); ++w) {
            running += static_cast<std::uint64_t>(std::popcount(hot_bits_[w]));
        }
    }
    resident_rows_ = running;
    if (running == 0) { return; }
    resident_ = std::make_unique<Memory>(running * rb);
    // The hot rows in row order, which is their order in RAM.
    std::vector<std::uint32_t> by_slot;
    by_slot.reserve(static_cast<std::size_t>(running));
    for (std::uint64_t w = 0; w < words; ++w) {
        for (std::uint64_t bits = hot_bits_[w]; bits != 0; bits &= bits - 1) {
            by_slot.push_back(static_cast<std::uint32_t>(w * 64 + std::countr_zero(bits)));
        }
    }
    // Pieces of the table that hold hot rows: a dense piece is read whole and its hot rows copied
    // out, a sparse one reads its hot rows one by one.
    const std::uint64_t piece = std::max<std::uint64_t>(1, kLoadChunk / rb);
    std::uint8_t* const out   = resident_->data();
    pool_->run(static_cast<std::size_t>((rows + piece - 1) / piece), [&](std::size_t i) {
        const std::uint64_t first = i * piece;
        const std::uint64_t end   = std::min(rows, first + piece);
        const auto lo             = std::lower_bound(by_slot.begin(), by_slot.end(), first);
        const auto hi             = std::lower_bound(lo, by_slot.end(), end);
        if (lo == hi) { return; }
        if (std::uint64_t(hi - lo) * kSparseLoad < (end - first) * rb) {
            for (auto it = lo; it != hi; ++it) {
                read(*it * rb, out + std::uint64_t(it - by_slot.begin()) * rb,
                     static_cast<std::size_t>(rb));
            }
            return;
        }
        std::vector<std::uint8_t> rows_read(static_cast<std::size_t>((end - first) * rb));
        read(first * rb, rows_read.data(), rows_read.size());
        for (auto it = lo; it != hi; ++it) {
            std::memcpy(out + std::uint64_t(it - by_slot.begin()) * rb,
                        rows_read.data() + (*it - first) * rb, static_cast<std::size_t>(rb));
        }
    });
    if (options_.lock) { resident_->lock(); }
}

// The row's place among the resident rows, or -1 when it is not hot.
std::int64_t NgramTableReader::hot_slot(std::uint64_t row) const noexcept {
    const std::uint64_t word  = row >> 6U;
    const std::uint64_t bits  = hot_bits_[word];
    const std::uint64_t below = (1ULL << (row & 63U)) - 1;
    if (((bits >> (row & 63U)) & 1U) == 0) { return -1; }
    std::uint64_t slot = hot_blocks_[row >> 9U];
    for (std::uint64_t w = (row >> 9U) << 3U; w < word; ++w) {
        slot += static_cast<std::uint64_t>(std::popcount(hot_bits_[w]));
    }
    return static_cast<std::int64_t>(slot) + std::popcount(bits & below);
}

void NgramTableReader::submit(std::span<const std::uint64_t> row_ids, std::span<std::uint8_t> out) {
    if (pending_) { throw std::logic_error("n-gram table: a batch is in flight"); }
    const std::size_t rb = layout_.row_bytes;
    if (out.size() != row_ids.size() * rb) {
        throw std::invalid_argument("n-gram table: output size mismatch");
    }
    for (const std::uint64_t row : row_ids) {
        if (row >= layout_.rows) {
            throw std::out_of_range("n-gram table: row " + std::to_string(row) + " past " +
                                    std::to_string(layout_.rows));
        }
    }
    batch_start_ = std::chrono::steady_clock::now();
    misses_.clear();
    std::uint64_t hits = 0;
    for (std::size_t i = 0; i < row_ids.size(); ++i) {
        std::int64_t slot = -1;
        if (options_.residency == NgramResidency::Ram) {
            slot = static_cast<std::int64_t>(row_ids[i]);
        } else if (options_.residency == NgramResidency::RamHot && resident_) {
            slot = hot_slot(row_ids[i]);
        }
        if (slot < 0) {
            misses_.push_back(static_cast<std::uint32_t>(i));
            continue;
        }
        std::memcpy(out.data() + i * rb, resident_->data() + std::uint64_t(slot) * rb, rb);
        ++hits;
    }
    rows_.fetch_add(row_ids.size(), std::memory_order_relaxed);
    resident_hits_.fetch_add(hits, std::memory_order_relaxed);
    batch_ids_ = row_ids;
    batch_out_ = out;
    pending_   = true;
    if (misses_.empty()) { return; }
    // Groups of rows, a few per thread, so a prompt chunk's thousands of rows share the threads
    // without a hand-off per row.
    const std::size_t groups = std::min<std::size_t>(misses_.size(), 4 * options_.depth);
    const std::size_t per    = (misses_.size() + groups - 1) / groups;
    pool_->start(groups, [this, per](std::size_t group) {
        const std::size_t row_bytes = layout_.row_bytes;
        const std::size_t end       = std::min(misses_.size(), (group + 1) * per);
        for (std::size_t m = group * per; m < end; ++m) {
            const std::size_t i = misses_[m];
            read(batch_ids_[i] * row_bytes, batch_out_.data() + i * row_bytes, row_bytes);
        }
    });
}

void NgramTableReader::wait() {
    if (!pending_) { return; }
    pending_ = false;
    std::exception_ptr failure;
    try {
        pool_->finish();
    } catch (...) { failure = std::current_exception(); }
    const auto nanoseconds =
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - batch_start_)
                                       .count());
    batches_.fetch_add(1, std::memory_order_relaxed);
    latency_ns_.fetch_add(nanoseconds, std::memory_order_relaxed);
    histogram_[latency_bucket(nanoseconds)].fetch_add(1, std::memory_order_relaxed);
    if (failure) { std::rethrow_exception(failure); }
}

void NgramTableReader::read_rows(std::span<const std::uint64_t> row_ids,
                                 std::span<std::uint8_t> out) {
    submit(row_ids, out);
    wait();
}

NgramTableStats NgramTableReader::counters() const {
    NgramTableStats out;
    out.rows          = rows_.load(std::memory_order_relaxed);
    out.resident_rows = resident_hits_.load(std::memory_order_relaxed);
    out.batches       = batches_.load(std::memory_order_relaxed);
    out.read_seconds  = static_cast<double>(latency_ns_.load(std::memory_order_relaxed)) * 1e-9;
    for (std::size_t b = 0; b < out.latency_histogram.size(); ++b) {
        out.latency_histogram[b] = histogram_[b].load(std::memory_order_relaxed);
    }
    return out;
}

} // namespace ninfer::models::qwen4_exp
