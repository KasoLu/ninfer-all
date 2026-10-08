// Public HC write/read composition versus the fused public Op, at the model's real geometry.
#include "ninfer_bench_common.h"
#include "ninfer/ops/hyper_connection.h"

using namespace ninfer;
using namespace ninfer::bench;

namespace {

void run(int tokens, bool fp32, cudaStream_t stream, DeviceBuffer& flush) {
    constexpr int hidden = 2560, streams = 4, width = hidden * streams, lowrank = 320;
    auto initial = make_f32(std::size_t(width) * tokens, 101, -2.0f, 2.0f);
    DeviceBuffer stack(initial.bytes), mixed(std::size_t(hidden) * tokens * 2);
    auto previous = make_f32(std::size_t(streams) * tokens, 102, 0.0f, 2.0f);
    DeviceBuffer injection(previous.bytes);
    auto y = fp32 ? make_f32(std::size_t(hidden) * tokens, 103)
                  : make_bf16(std::size_t(hidden) * tokens, 103);
    auto norm = make_bf16(width, 104);
    auto down = make_bf16(std::size_t(width) * lowrank, 105);
    auto up = make_bf16(std::size_t(lowrank) * width, 106);
    auto inject = make_bf16(std::size_t(width) * streams, 107);
    Tensor xs(stack.p, DType::FP32, {hidden, streams, tokens});
    Tensor output(y.p, fp32 ? DType::FP32 : DType::BF16, {hidden, tokens});
    Tensor iw(injection.p, DType::FP32, {streams, tokens});
    Tensor out(mixed.p, DType::BF16, {hidden, tokens});
    Tensor wn(norm.p, DType::BF16, {width}), wd(down.p, DType::BF16, {width, lowrank});
    Tensor wu(up.p, DType::BF16, {lowrank, width}), wi(inject.p, DType::BF16, {width, streams});
    const ops::HyperConnectionWeights weights{&wn, &wd, &wu, &wi};
    WorkspaceArena workspace(ops::hyper_connection_read_workspace_bytes(streams, hidden, lowrank, tokens));
    const auto reset = [&] {
        CUDA_CHECK(cudaMemcpyAsync(stack.p, initial.p, initial.bytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(injection.p, previous.p, previous.bytes, cudaMemcpyDeviceToDevice, stream));
    };
    const auto separate = [&](cudaStream_t s) {
        ops::hyper_connection_write(xs, output, iw, s);
        ops::hyper_connection_read(xs, weights, 1e-6f, workspace, out, &iw, s);
    };
    const auto fused = [&](cudaStream_t s) {
        ops::hyper_connection_write_read(xs, output, iw, weights, 1e-6f, workspace, out, &iw, s);
    };
    // Initialize cuBLAS and both kernel sets before capture and measurement.
    reset();
    separate(stream);
    reset();
    fused(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TimedGraph baseline, candidate;
    baseline.capture(stream, separate);
    candidate.capture(stream, fused);
    for (int sample = 0; sample < 10; ++sample) {
        reset();
        baseline.launch(stream);
        reset();
        candidate.launch(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (const bool cold : {false, true}) {
        std::vector<double> before, after;
        for (int sample = 0; sample < 61; ++sample) {
            // Alternate order; restore identical inputs and optionally evict L2 outside timing.
            for (int turn = 0; turn < 2; ++turn) {
                const bool use_fused = (sample + turn) % 2 != 0;
                reset();
                if (cold) { flush_l2(flush, stream); }
                (use_fused ? after : before).push_back(
                    (use_fused ? candidate : baseline).launch_timed(stream));
            }
        }
        const auto a = summarize_timings(before), b = summarize_timings(after);
        std::printf("{\"tokens\":%d,\"y\":\"%s\",\"cache\":\"%s\","
                    "\"separate_us\":%.3f,\"fused_us\":%.3f,\"delta_pct\":%.3f,"
                    "\"separate_p95_us\":%.3f,\"fused_p95_us\":%.3f,"
                    "\"separate_nodes\":%zu,\"fused_nodes\":%zu}\n",
                    tokens, fp32 ? "fp32" : "bf16", cold ? "evicted" : "reused",
                    a.median_us, b.median_us, 100.0 * (b.median_us / a.median_us - 1.0),
                    a.p95_us, b.p95_us, baseline.nodes(), candidate.nodes());
    }
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
    std::printf("HC write/read: %s; CUDA %d; paired CUDA Graph samples=61; reset outside timing\n",
                device.name, CUDART_VERSION);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto flush = make_f32(16 * 1024 * 1024, 108);
    for (const bool fp32 : {false, true}) {
        for (const int tokens : {1, 4, 8, 9, 16, 128, 512}) { run(tokens, fp32, stream, flush); }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
