// ninfer-ngram-profile: the hot-row profile of a Qwen3.8-Flash-Next n-gram table, counted from a
// text corpus's tokens alone (the rows a token addresses depend on the token and its predecessors
// only, so no model runs), and the share of a held-out corpus's row reads a profile's leading rows
// serve at a range of RAM budgets.

#include "artifact/reader.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/ngram_hash.h"
#include "models/qwen4_exp/ngram_profile.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninfer;
using models::qwen4_exp::NgramContext;
using models::qwen4_exp::NgramHashConstants;
using models::qwen4_exp::NgramProfile;

constexpr const char* kUsage =
    "usage: ninfer-ngram-profile <model.ninfer> --out PROFILE [--max-rows N] TEXT...\n"
    "       ninfer-ngram-profile <model.ninfer> --evaluate PROFILE [--row-bytes B] TEXT...\n"
    "\n"
    "Counts the n-gram table rows a corpus addresses, from its tokens alone (no model runs), and\n"
    "writes them, the most frequent first, as a hot-row profile for --ngram-residency ram-hot.\n"
    "A TEXT file is one document, or a .jsonl file one document per line: its \"text\", or its\n"
    "\"messages\" rendered as the chat template's <|im_start|>role ... <|im_end|> turns.\n"
    "\n"
    "  --out PROFILE        write the profile\n"
    "  --max-rows N         keep the N most frequent rows (default: every row the corpus reads)\n"
    "  --evaluate PROFILE   report the share of the corpus's row reads that the profile's leading\n"
    "                       rows serve at RAM budgets of 256 MiB to 32 GiB, how many misses\n"
    "                       repeat a row their document read before, and what a page cache of\n"
    "                       the budget would serve at best (the 4 KiB pages these reads read\n"
    "                       most)\n"
    "  --row-bytes B        bytes per stored row the budgets are reckoned in (default 90, the\n"
    "                       published gguf_iq4_nl table; 320 for bf16)\n";

struct Options {
    std::filesystem::path model, out, evaluate;
    std::vector<std::filesystem::path> texts;
    std::uint64_t max_rows  = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t row_bytes = 90;
};

std::uint64_t parse_count(std::string_view text, const char* option) {
    std::uint64_t value     = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0) {
        throw std::invalid_argument(std::string(option) + " takes a positive integer");
    }
    return value;
}

Options parse(int argc, char** argv) {
    if (argc < 2 || std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h") {
        std::cout << kUsage;
        std::exit(argc < 2 ? 2 : 0);
    }
    Options out;
    out.model = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto value           = [&]() -> std::string_view {
            if (++i >= argc) { throw std::invalid_argument(std::string(arg) + " needs a value"); }
            return argv[i];
        };
        if (arg == "--out") {
            out.out = std::filesystem::path(value());
        } else if (arg == "--evaluate") {
            out.evaluate = std::filesystem::path(value());
        } else if (arg == "--max-rows") {
            out.max_rows = parse_count(value(), "--max-rows");
        } else if (arg == "--row-bytes") {
            out.row_bytes = parse_count(value(), "--row-bytes");
        } else if (arg.starts_with("--")) {
            throw std::invalid_argument("unknown option " + std::string(arg));
        } else {
            out.texts.emplace_back(arg);
        }
    }
    if (out.out.empty() == out.evaluate.empty()) {
        throw std::invalid_argument("give one of --out and --evaluate");
    }
    if (out.texts.empty()) { throw std::invalid_argument("give the corpus's TEXT files"); }
    return out;
}

// The model's tokenizer and n-gram hash, read from its artifact without its weights.
struct Hash {
    std::vector<std::byte> tokenizer_json, tokenizer_config, generation_config;
    std::unique_ptr<models::qwen3_5::frontend::Tokenizer> tokenizer;
    NgramHashConstants constants;
    std::int32_t eos    = 0;
    std::uint32_t vocab = 0;

    explicit Hash(const std::filesystem::path& model) {
        const artifact::Reader reader(model);
        const auto& text  = reader.directory().component("text");
        const auto config = models::qwen4_exp::parse_text_config(text.config);
        const auto object = [&](const char* role) {
            const auto found = text.resources.find(role);
            if (found == text.resources.end()) {
                throw std::runtime_error(model.string() + " has no " + role);
            }
            return reader.read_object(found->second);
        };
        tokenizer_json    = object("tokenizer.json");
        tokenizer_config  = object("tokenizer_config.json");
        generation_config = object("generation_config.json");
        const auto view   = [](const std::vector<std::byte>& bytes) {
            return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        };
        tokenizer = std::make_unique<models::qwen3_5::frontend::Tokenizer>(
            models::qwen3_5::frontend::TokenizerResources{
                view(tokenizer_json), view(tokenizer_config), view(generation_config)});
        constants = models::qwen4_exp::derive_ngram_hash_constants(config.ngram);
        eos       = config.eos_token_id;
        vocab     = config.vocab_size;
        if (constants.rows > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("the table has more rows than a profile addresses");
        }
    }

    // The rows each token of `text` addresses, document start to end, in order.
    void rows(std::string_view text, std::vector<std::uint64_t>& out) const {
        const auto ids = tokenizer->encode(text);
        std::vector<std::int32_t> tokens(ids.begin(), ids.end());
        NgramContext context = NgramContext::sequence_start(constants, eos);
        out.resize(tokens.size() * constants.heads());
        models::qwen4_exp::ngram_row_ids(constants, tokens, eos, vocab, context, out);
    }
};

// Every document of the corpus, in file order.
void for_each_document(const std::vector<std::filesystem::path>& texts,
                       const std::function<void(std::string_view)>& visit) {
    for (const auto& path : texts) {
        std::ifstream file(path, std::ios::binary);
        if (!file) { throw std::runtime_error("cannot open " + path.string()); }
        if (path.extension() != ".jsonl") {
            std::ostringstream all;
            all << file.rdbuf();
            visit(all.str());
            continue;
        }
        std::string line;
        std::size_t number = 0;
        while (std::getline(file, line)) {
            ++number;
            if (line.empty()) { continue; }
            const auto json = nlohmann::json::parse(line, nullptr, false);
            if (json.is_discarded() || !json.is_object()) {
                throw std::runtime_error(path.string() + ":" + std::to_string(number) +
                                         " is not a JSON object");
            }
            if (json.contains("text") && json["text"].is_string()) {
                visit(json["text"].get<std::string>());
                continue;
            }
            if (!json.contains("messages") || !json["messages"].is_array()) {
                throw std::runtime_error(path.string() + ":" + std::to_string(number) +
                                         " has neither text nor messages");
            }
            std::string chat;
            for (const auto& message : json["messages"]) {
                if (!message.is_object() || !message.contains("content") ||
                    !message["content"].is_string()) {
                    continue;
                }
                const std::string role = message.contains("role") && message["role"].is_string()
                                             ? message["role"].get<std::string>()
                                             : std::string("user");
                chat += "<|im_start|>" + role + "\n" + message["content"].get<std::string>() +
                        "<|im_end|>\n";
            }
            visit(chat);
        }
    }
}

int profile(const Options& options, const Hash& hash) {
    std::vector<std::uint32_t> counts(static_cast<std::size_t>(hash.constants.rows), 0);
    std::vector<std::uint64_t> rows;
    std::uint64_t documents = 0, tokens = 0;
    for_each_document(options.texts, [&](std::string_view text) {
        hash.rows(text, rows);
        for (const auto row : rows) {
            if (counts[row] != std::numeric_limits<std::uint32_t>::max()) { ++counts[row]; }
        }
        ++documents;
        tokens += rows.size() / hash.constants.heads();
    });
    std::vector<std::uint64_t> ranked; // count << 32 | ~row: descending count, then ascending row
    for (std::uint64_t row = 0; row < counts.size(); ++row) {
        if (counts[row] != 0) {
            ranked.push_back((std::uint64_t(counts[row]) << 32U) | (~row & 0xffffffffULL));
        }
    }
    std::sort(ranked.begin(), ranked.end(), std::greater<>());
    NgramProfile out{.table_rows  = hash.constants.rows,
                     .fingerprint = models::qwen4_exp::ngram_hash_fingerprint(hash.constants),
                     .tokens      = tokens};
    const auto keep =
        static_cast<std::size_t>(std::min<std::uint64_t>(ranked.size(), options.max_rows));
    out.rows.reserve(keep);
    for (std::size_t i = 0; i < keep; ++i) {
        out.rows.push_back(static_cast<std::uint32_t>(~ranked[i] & 0xffffffffULL));
    }
    models::qwen4_exp::write_ngram_profile(options.out, out);
    std::printf("%llu documents, %llu tokens, %llu row reads, %zu distinct rows; wrote %zu rows "
                "to %s\n",
                static_cast<unsigned long long>(documents), static_cast<unsigned long long>(tokens),
                static_cast<unsigned long long>(tokens * hash.constants.heads()), ranked.size(),
                keep, options.out.string().c_str());
    return 0;
}

int evaluate(const Options& options, const Hash& hash) {
    const NgramProfile profile = models::qwen4_exp::read_ngram_profile(options.evaluate);
    models::qwen4_exp::check_ngram_profile(profile, hash.constants, options.evaluate);
    constexpr std::uint32_t kAbsent = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> rank(static_cast<std::size_t>(hash.constants.rows), kAbsent);
    for (std::size_t i = 0; i < profile.rows.size(); ++i) {
        rank[profile.rows[i]] = static_cast<std::uint32_t>(i);
    }
    // The RAM budgets, and the profile rows each holds beside the hot set's index.
    const std::uint64_t index =
        (hash.constants.rows + 63) / 64 * 8 + (hash.constants.rows + 511) / 512 * 4;
    std::vector<std::uint64_t> budgets;
    for (std::uint64_t mib = 256; mib <= 32768; mib *= 2) { budgets.push_back(mib << 20U); }
    std::vector<std::uint64_t> capacity;
    for (const auto budget : budgets) {
        capacity.push_back(
            budget > index
                ? std::min<std::uint64_t>((budget - index) / options.row_bytes, profile.rows.size())
                : 0);
    }
    std::vector<std::uint64_t> hits(budgets.size(), 0), repeat_misses(budgets.size(), 0);
    // The last document that read each row, for the reads that repeat within a document, and the
    // reads of each 4 KiB page of the table, for the page cache's best case at a budget.
    std::vector<std::uint32_t> last(static_cast<std::size_t>(hash.constants.rows), 0);
    std::vector<std::uint32_t> pages(
        static_cast<std::size_t>((hash.constants.rows * options.row_bytes + 4095) / 4096), 0);
    std::vector<std::uint64_t> rows;
    std::uint64_t documents = 0, reads = 0, repeats = 0, in_profile = 0;
    for_each_document(options.texts, [&](std::string_view text) {
        hash.rows(text, rows);
        ++documents;
        const auto document = static_cast<std::uint32_t>(documents);
        for (const auto row : rows) {
            const bool repeat = last[row] == document;
            last[row]         = document;
            repeats += repeat;
            ++pages[row * options.row_bytes / 4096];
            const std::uint32_t r = rank[row];
            in_profile += r != kAbsent;
            for (std::size_t b = 0; b < budgets.size(); ++b) {
                if (r != kAbsent && r < capacity[b]) {
                    ++hits[b];
                } else if (repeat) {
                    ++repeat_misses[b];
                }
            }
        }
        reads += rows.size();
    });
    if (reads == 0) { throw std::runtime_error("the corpus has no tokens"); }
    const auto share = [&](std::uint64_t n) { return 100.0 * double(n) / double(reads); };
    // A page cache of the budget at best: the pages these very reads read most, a page per row
    // read (a row across a page boundary counted in its first page).
    std::sort(pages.begin(), pages.end(), std::greater<>());
    std::vector<std::uint64_t> page_hits;
    for (const auto budget : budgets) {
        const auto held =
            static_cast<std::size_t>(std::min<std::uint64_t>(budget / 4096, pages.size()));
        std::uint64_t sum = 0;
        for (std::size_t i = 0; i < held; ++i) { sum += pages[i]; }
        page_hits.push_back(sum);
    }
    std::printf("held-out: %llu documents, %llu tokens, %llu row reads; profile of %zu rows "
                "from %llu tokens\n",
                static_cast<unsigned long long>(documents),
                static_cast<unsigned long long>(reads / hash.constants.heads()),
                static_cast<unsigned long long>(reads), profile.rows.size(),
                static_cast<unsigned long long>(profile.tokens));
    std::printf("reads of rows the profile lists: %.1f%%; reads that repeat a row of their "
                "document: %.1f%%\n",
                share(in_profile), share(repeats));
    std::printf("%10s %12s %10s %22s %20s\n", "budget", "rows held", "hit rate",
                "misses that repeat", "page cache at best");
    for (std::size_t b = 0; b < budgets.size(); ++b) {
        std::printf("%6llu MiB %12llu %9.1f%% %21.1f%% %19.1f%%\n",
                    static_cast<unsigned long long>(budgets[b] >> 20U),
                    static_cast<unsigned long long>(capacity[b]), share(hits[b]),
                    share(repeat_misses[b]), share(page_hits[b]));
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        const auto start      = std::chrono::steady_clock::now();
        const Hash hash(options.model);
        const int result = options.out.empty() ? evaluate(options, hash) : profile(options, hash);
        std::fprintf(
            stderr, "%.1f s\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        return result;
    } catch (const std::exception& error) {
        std::cerr << "ninfer-ngram-profile: " << error.what() << '\n';
        return 1;
    }
}
