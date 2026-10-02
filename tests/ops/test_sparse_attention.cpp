// sparse_softmax_attention against an FP64 oracle over the selected blocks and the query's own
// incomplete block, through a fragmented BF16 paged cache: a full 512-block selection at long
// context, short contexts where every block is selected, and every tail length 0..3.
#include "core/device.h"
#include "ninfer/ops/sparse_attention.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kDim = 256, kQHeads = 24, kKvHeads = 2, kTop = 512, kPage = 64;

int run(int first_position, int tokens, std::uint32_t seed) {
    const int length = first_position + tokens;
    const int pages  = (length + kPage - 1) / kPage;
    // Fragmented table: logical page i lives at physical page perm[i].
    std::vector<int> table(pages);
    std::iota(table.begin(), table.end(), 0);
    std::mt19937 random(seed);
    std::shuffle(table.begin(), table.end(), random);
    std::vector<float> k(static_cast<std::size_t>(pages) * kPage * kKvHeads * kDim),
        v(k.size()), q(static_cast<std::size_t>(tokens) * kQHeads * kDim);
    fill_uniform(k, seed + 1, -1.0f, 1.0f);
    fill_uniform(v, seed + 2, -2.0f, 2.0f);
    fill_uniform(q, seed + 3, -1.0f, 1.0f);
    round_to_bf16(k);
    round_to_bf16(v);
    round_to_bf16(q);
    const auto cache_index = [&](int position, int head, int d) {
        const int page = table[position / kPage], slot = position % kPage;
        return ((static_cast<std::size_t>(page) * kKvHeads + head) * kPage + slot) * kDim + d;
    };
    // Selections: every block when there are at most 512, else 512 distinct random ones.
    std::vector<int> selected(static_cast<std::size_t>(kTop) * tokens, -1), counts(tokens);
    for (int t = 0; t < tokens; ++t) {
        const int blocks = (first_position + t + 1) / 4;
        std::vector<int> all(blocks);
        std::iota(all.begin(), all.end(), 0);
        if (blocks > kTop) {
            std::shuffle(all.begin(), all.end(), random);
            all.resize(kTop);
            std::sort(all.begin(), all.end());
        }
        counts[t] = static_cast<int>(all.size());
        std::copy(all.begin(), all.end(), selected.begin() + static_cast<std::ptrdiff_t>(t) * kTop);
    }
    const float scale = 1.0f / 16.0f;
    std::vector<double> expected(q.size());
    for (int t = 0; t < tokens; ++t) {
        const int p = first_position + t, tail = (p + 1) / 4 * 4;
        std::vector<int> positions;
        for (int i = 0; i < counts[t]; ++i)
            for (int j = 0; j < 4; ++j) positions.push_back(selected[static_cast<std::size_t>(t) * kTop + i] * 4 + j);
        for (int j = tail; j <= p; ++j) positions.push_back(j);
        for (int h = 0; h < kQHeads; ++h) {
            const int kv = h / 12;
            std::vector<double> logits(positions.size());
            double maximum = -INFINITY;
            for (std::size_t i = 0; i < positions.size(); ++i) {
                double dot = 0;
                for (int d = 0; d < kDim; ++d) dot += double(q[(static_cast<std::size_t>(t) * kQHeads + h) * kDim + d]) * k[cache_index(positions[i], kv, d)];
                logits[i] = dot * scale;
                maximum   = std::max(maximum, logits[i]);
            }
            double sum = 0;
            for (double& l : logits) sum += (l = std::exp(l - maximum));
            for (int d = 0; d < kDim; ++d) {
                double o = 0;
                for (std::size_t i = 0; i < positions.size(); ++i) o += logits[i] * v[cache_index(positions[i], kv, d)];
                expected[(static_cast<std::size_t>(t) * kQHeads + h) * kDim + d] = o / sum;
            }
        }
    }
    const auto encode = [](const std::vector<float>& values) {
        std::vector<std::uint16_t> bits(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
        return bits;
    };
    GuardedDeviceBuffer d_k(k.size() * 2), d_v(v.size() * 2), d_q(q.size() * 2), d_out(q.size() * 2),
        d_selected(selected.size() * 4), d_counts(counts.size() * 4), d_table(table.size() * 4);
    const auto kb = encode(k), vb = encode(v), qb = encode(q);
    d_k.copy_from_host(kb.data(), d_k.bytes());
    d_v.copy_from_host(vb.data(), d_v.bytes());
    d_q.copy_from_host(qb.data(), d_q.bytes());
    d_selected.copy_from_host(selected.data(), d_selected.bytes());
    d_counts.copy_from_host(counts.data(), d_counts.bytes());
    d_table.copy_from_host(table.data(), d_table.bytes());
    PagedKVLayerView cache{};
    cache.k_pages      = Tensor(d_k.data(), DType::BF16, {kDim, kPage, kKvHeads, pages});
    cache.v_pages      = Tensor(d_v.data(), DType::BF16, {kDim, kPage, kKvHeads, pages});
    cache.block_table  = Tensor(d_table.data(), DType::I32, {pages});
    cache.head_dim     = kDim;
    cache.num_kv_heads = kKvHeads;
    cache.storage      = KvCacheStorage::BFloat16;
    Tensor t_q(d_q.data(), DType::BF16, {kDim, kQHeads, tokens});
    Tensor t_out(d_out.data(), DType::BF16, {kDim, kQHeads, tokens});
    Tensor t_selected(d_selected.data(), DType::I32, {kTop, tokens});
    Tensor t_counts(d_counts.data(), DType::I32, {tokens});
    ops::sparse_softmax_attention(t_q, first_position, t_selected, t_counts, cache, scale, t_out, nullptr);
    cuda_synchronize();
    const std::string label = "sparse attention p=" + std::to_string(first_position) + " T=" + std::to_string(tokens);
    int failures = verify_reduction(label, from_device_bf16(d_out.data(), q.size()), expected,
                                    {4.0e-3, 1.0e-4, 2.0 * 3.90625e-3});
    for (auto* buffer : {&d_k, &d_v, &d_q, &d_out, &d_selected, &d_counts, &d_table}) {
        failures += buffer->verify_guards(label.c_str());
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run(0, 1, 9100u);       // one key: the query itself
    failures += run(5, 4, 9101u);       // short context, tails 2,3,0,1
    failures += run(2047, 9, 9102u);    // across the 512-block edge
    failures += run(4096, 3, 9103u);    // a full selection
    failures += run(9000, 1, 9104u);
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " sparse_softmax_attention\n";
    return failures == 0 ? 0 : 1;
}
