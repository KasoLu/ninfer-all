// The expert cache must never change what a kernel reads: after any sequence of routes and
// rebalances, every table entry points either at the expert's bytes in the pinned host block or at
// a device slot holding the same bytes. It must also admit the experts a layer routes to most, keep
// within its slots, and leave a layer without slots untouched.
#include "core/arena.h"
#include "core/device.h"
#include "models/qwen4_exp/expert_cache.h"

#include <cuda_runtime.h>

#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::models::qwen4_exp;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

constexpr int kExperts = 64;

struct Bank {
    std::int64_t bytes;
    std::unique_ptr<PinnedHostBuffer> host;
    DeviceBuffer table;
    std::vector<const void*> pointers;

    Bank(std::int64_t expert_bytes, std::mt19937& random)
        : bytes(expert_bytes), host(std::make_unique<PinnedHostBuffer>(expert_bytes * kExperts)),
          table(kExperts * sizeof(void*)) {
        auto* data = static_cast<std::uint8_t*>(host->data());
        for (std::int64_t i = 0; i < expert_bytes * kExperts; ++i) {
            data[i] = std::uint8_t(random());
        }
        for (int e = 0; e < kExperts; ++e) { pointers.push_back(data + e * expert_bytes); }
        table.copy_from_host(pointers.data(), pointers.size() * sizeof(void*));
    }

    [[nodiscard]] ExpertBank view() const {
        return {.expert_bytes = bytes, .host = pointers, .table = table.p};
    }

    // Every entry's bytes equal the expert's; returns how many entries point at device memory.
    int verify(const std::string& label) const {
        std::vector<const void*> entries(kExperts);
        table.copy_to_host(entries.data(), entries.size() * sizeof(void*));
        int cached = 0;
        std::vector<std::uint8_t> bytes_read(static_cast<std::size_t>(bytes));
        for (int e = 0; e < kExperts; ++e) {
            if (entries[e] == pointers[e]) { continue; }
            ++cached;
            require(cudaMemcpy(bytes_read.data(), entries[e], bytes_read.size(),
                               cudaMemcpyDeviceToHost) == cudaSuccess,
                    label + ": a cached entry is not readable");
            require(std::memcmp(bytes_read.data(), pointers[e], bytes_read.size()) == 0,
                    label + ": a slot differs from its expert's bytes");
        }
        return cached;
    }
};

int run() {
    DeviceContext device;
    std::mt19937 random(9100u);
    // Two layers; their banks have different expert sizes, as mixed block types do.
    Bank g0(4096, random), u0(4096, random), d0(2048, random);
    Bank g1(1536, random), u1(3072, random), d1(1024, random);
    std::vector<ExpertCacheLayer> layers = {
        {.rank = 0, .stream = device.stream, .gate = g0.view(), .up = u0.view(), .down = d0.view()},
        {.rank = 0, .stream = device.stream, .gate = g1.view(), .up = u1.view(), .down = d1.view()},
    };
    // Room for about 10 experts of layer 0 and 20 of layer 1 (even split of 2 * 102,400 bytes).
    const std::vector<std::uint64_t> budget = {2 * 102400};
    ExpertCache cache(device, layers, budget);
    const auto slots = cache.stats().slots;
    require(slots == 10 + 18,
            "slots per layer follow each layer's expert size: " + std::to_string(slots));

    // Layer 0 routes heavily to experts 5 and 7; layer 1 spreads.
    std::vector<std::int32_t> hot;
    for (int t = 0; t < 40; ++t) {
        hot.push_back(5);
        hot.push_back(7);
        hot.push_back(t % kExperts);
    }
    for (int round = 0; round < 6; ++round) {
        cache.observe(0, hot, 40);
        std::vector<std::int32_t> spread;
        for (int i = 0; i < 120; ++i) { spread.push_back(int(random() % kExperts)); }
        cache.observe(1, spread, 40);
        cache.rebalance(round % 2 == 0 ? (1u << 20) : 30000);
        device.synchronize();
        const int cached0 = g0.verify("gate 0") + 0;
        require(u0.verify("up 0") == cached0 && d0.verify("down 0") == cached0,
                "a layer's three banks cache the same experts");
        require(cached0 <= 10, "layer 0 keeps within its slots");
        const int cached1 = g1.verify("gate 1");
        require(u1.verify("up 1") == cached1 && d1.verify("down 1") == cached1,
                "layer 1's banks agree");
        require(cached1 <= 18, "layer 1 keeps within its slots");
    }
    std::vector<const void*> entries(kExperts);
    g0.table.copy_to_host(entries.data(), entries.size() * sizeof(void*));
    require(entries[5] != g0.pointers[5] && entries[7] != g0.pointers[7],
            "the most routed experts are cached");
    const auto stats = cache.stats();
    require(stats.routes == 6 * 240 && stats.hits > 0 && stats.admitted > 0,
            "the cache counts routes, hits and admissions");
    return 0;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        run();
        std::cout << "PASS qwen4_exp expert cache\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
