#include "models/qwen4_exp/ngram_profile.h"

#include <bit>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen4_exp {
namespace {

static_assert(std::endian::native == std::endian::little, "profiles are little-endian");

constexpr char kMagic[8]             = {'N', 'F', 'N', 'G', 'H', 'O', 'T', '1'};
constexpr std::uint64_t kHeaderBytes = 8 + 4 * 8;

} // namespace

NgramProfile read_ngram_profile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open the hot-row profile " + path.string()); }
    char magic[8]           = {};
    std::uint64_t header[4] = {};
    file.read(magic, sizeof magic);
    file.read(reinterpret_cast<char*>(header), sizeof header);
    if (!file || std::memcmp(magic, kMagic, sizeof magic) != 0) {
        throw std::runtime_error(path.string() + " is not an n-gram hot-row profile");
    }
    NgramProfile out{.table_rows = header[0], .fingerprint = header[1], .tokens = header[2]};
    const std::uint64_t count = header[3];
    std::error_code error;
    const auto bytes = std::filesystem::file_size(path, error);
    if (error || bytes != kHeaderBytes + count * 4) {
        throw std::runtime_error("the hot-row profile " + path.string() + " is truncated");
    }
    out.rows.resize(static_cast<std::size_t>(count));
    file.read(reinterpret_cast<char*>(out.rows.data()),
              static_cast<std::streamsize>(out.rows.size() * 4));
    if (!file) { throw std::runtime_error("cannot read the hot-row profile " + path.string()); }
    return out;
}

void write_ngram_profile(const std::filesystem::path& path, const NgramProfile& profile) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    const std::uint64_t header[4] = {profile.table_rows, profile.fingerprint, profile.tokens,
                                     profile.rows.size()};
    file.write(kMagic, sizeof kMagic);
    file.write(reinterpret_cast<const char*>(header), sizeof header);
    file.write(reinterpret_cast<const char*>(profile.rows.data()),
               static_cast<std::streamsize>(profile.rows.size() * 4));
    file.close();
    if (!file) { throw std::runtime_error("cannot write the hot-row profile " + path.string()); }
}

void check_ngram_profile(const NgramProfile& profile, const NgramHashConstants& constants,
                         const std::filesystem::path& path) {
    if (profile.table_rows != constants.rows ||
        profile.fingerprint != ngram_hash_fingerprint(constants)) {
        throw std::invalid_argument("the hot-row profile " + path.string() +
                                    " was counted for another n-gram hash than the model's");
    }
}

} // namespace ninfer::models::qwen4_exp
