#include "models/qwen4_exp/executor.h"

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/weight_view.h"
#include "models/qwen4_exp/ngram_hash.h"
#include "models/qwen4_exp/ngram_table.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/moe_experts.h"
#include "ninfer/ops/moe_route.h"
#include "ninfer/ops/ngram_rows.h"
#include "ninfer/ops/ple_inject.h"
#include "ninfer/ops/qsa_indexer.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/sparse_attention.h"
#include "ninfer/ops/weight_input.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen4_exp {
namespace {

constexpr std::int32_t kAlign = 256;

std::uint64_t round_up(std::uint64_t value) { return (value + kAlign - 1) / kAlign * kAlign; }

// A logical projection's output rows, from one or more native operands: contiguous runs of one
// parent are fused into one operand, other parts write their rows separately.
struct Projection {
    struct Piece {
        ops::SingleProjectionWeight weight;
        std::int32_t row  = 0;
        std::int32_t rows = 0;
    };

    std::vector<Piece> pieces;
    std::int32_t rows = 0;
    std::int32_t k    = 0;
};

Projection make_projection(const Model& model, std::initializer_list<WeightId> ids) {
    std::vector<ops::WeightInput> inputs;
    for (const WeightId id : ids) { inputs.push_back(model.input(id)); }
    Projection out;
    std::size_t begin = 0;
    while (begin < inputs.size()) {
        std::size_t end = begin + 1;
        while (end < inputs.size() && ops::joins(std::span<const ops::WeightInput>(
                                          inputs.data() + begin, end + 1 - begin))) {
            ++end;
        }
        const std::span<const ops::WeightInput> run(inputs.data() + begin, end - begin);
        Projection::Piece piece{ops::prepare_linear_weight(run), out.rows, 0};
        piece.rows = piece.weight.weight.n;
        if (out.k == 0) { out.k = piece.weight.weight.k; }
        if (piece.weight.weight.k != out.k) {
            throw std::invalid_argument("projection parts read different input widths");
        }
        out.rows += piece.rows;
        out.pieces.push_back(std::move(piece));
        begin = end;
    }
    return out;
}

std::size_t projection_workspace(const Projection& p, std::int32_t tokens) {
    std::size_t bytes = 0;
    for (const auto& piece : p.pieces) {
        std::size_t need = ops::linear_workspace_capacity_bytes(
            piece.weight.weight.qtype, piece.rows, p.k, piece.weight.policy, 1, tokens);
        if (p.pieces.size() > 1) { need += round_up(std::uint64_t(piece.rows) * tokens * 2); }
        bytes = std::max(bytes, need);
    }
    return bytes;
}

// out (BF16 [p.rows, T]) <- the projection of x (BF16 [p.k, T]).
void project(const Projection& p, const Tensor& x, Tensor& out, WorkspaceArena& workspace,
             cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (p.pieces.size() == 1) {
        ops::linear(x, p.pieces.front().weight.weight, out, p.pieces.front().weight.policy,
                    workspace, stream);
        return;
    }
    for (const auto& piece : p.pieces) {
        auto scope  = workspace.scope();
        Tensor part = workspace.alloc(DType::BF16, {piece.rows, tokens});
        ops::linear(x, piece.weight.weight, part, piece.weight.policy, workspace, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::byte*>(out.data) + std::size_t(piece.row) * 2,
                                     std::size_t(p.rows) * 2, part.data,
                                     std::size_t(piece.rows) * 2, std::size_t(piece.rows) * 2,
                                     tokens, cudaMemcpyDeviceToDevice, stream));
    }
}

Tensor direct(const Model& model, WeightId id, std::initializer_list<std::int32_t> shape,
              DType dtype = DType::BF16) {
    const auto& view = model.weight(id).view;
    Tensor tensor    = weight_tensor(view, shape);
    if (tensor.dtype != dtype) {
        throw std::invalid_argument(model.weight(id).name + " must be stored as " +
                                    (dtype == DType::BF16 ? "bf16" : "fp32"));
    }
    return tensor;
}

struct HcPlan {
    Tensor norm, down, up, inject;
    bool has_inject = false;

    [[nodiscard]] ops::HyperConnectionWeights weights() const {
        return {&norm, &down, &up, has_inject ? &inject : nullptr};
    }
};

HcPlan make_hc(const Model& model, const HyperConnectionWeights& w, const TextConfig& c) {
    const auto width = static_cast<std::int32_t>(c.hc_count * c.hidden_size);
    const auto low   = static_cast<std::int32_t>(c.hc_lowrank);
    HcPlan out;
    out.norm = direct(model, w.norm, {width});
    out.down = direct(model, w.down, {width, low});
    out.up   = direct(model, w.up, {low, width});
    if (w.inject) {
        out.inject     = direct(model, *w.inject, {width, static_cast<std::int32_t>(c.hc_count)});
        out.has_inject = true;
    }
    return out;
}

struct GdnPlan {
    Projection qkv, z, a, b, output;
    Tensor convolution, a_log, dt_bias, norm;
};

struct QsaPlan {
    Projection query, gate, key, value, index, output;
    Tensor query_norm, key_norm, index_query_norm, index_key_norm;
};

struct PlePlan {
    Projection key, value;
    Tensor norm_key, norm_query, norm_conv, convolution;
};

// One projection of every expert of a layer, as a device table of expert base pointers on the
// layer's device.
struct ExpertTable {
    QType format           = QType::GGUF_Q8_0;
    std::int64_t row_bytes = 0;
    DeviceBuffer table;

    [[nodiscard]] ops::GgufExpertTable view() const {
        return {format, static_cast<const void* const*>(table.p), row_bytes};
    }
};

ExpertTable make_table(const Model& model, std::span<const WeightId> ids) {
    ExpertTable out;
    std::vector<const void*> pointers;
    for (const WeightId id : ids) {
        const Weight w = native_weight(model.weight(id).view);
        if (!is_gguf(w.qtype)) {
            throw std::invalid_argument(model.weight(id).name +
                                        ": the expert kernels take GGUF block banks");
        }
        const auto block       = gguf_block_shape(w.qtype);
        const std::int64_t row = std::int64_t(w.k / block.elements) * block.bytes;
        if (pointers.empty()) {
            out.format    = w.qtype;
            out.row_bytes = row;
        } else if (w.qtype != out.format || row != out.row_bytes) {
            throw std::invalid_argument(model.weight(id).name +
                                        ": every expert of a bank must share its block type");
        }
        pointers.push_back(w.qdata);
    }
    out.table = DeviceBuffer(pointers.size() * sizeof(void*));
    out.table.copy_from_host(pointers.data(), pointers.size() * sizeof(void*));
    return out;
}

struct MoePlan {
    Tensor router, shared_gate;
    ExpertTable gate, up, down, shared_gate_table, shared_up_table, shared_down_table;

    [[nodiscard]] ops::GgufMoeWeights banks() const {
        return {gate.view(),
                up.view(),
                down.view(),
                shared_gate_table.view(),
                shared_up_table.view(),
                shared_down_table.view()};
    }
};

struct LayerPlan {
    std::size_t rank = 0;
    HcPlan attn_hc, mlp_hc;
    std::optional<GdnPlan> gdn;
    std::optional<QsaPlan> qsa;
    std::optional<PlePlan> ple;
    MoePlan moe;
};

// One sequence's mutable state of one layer, on that layer's device.
struct LayerState {
    DeviceBuffer ssm, conv;                      // Gated DeltaNet
    DeviceBuffer k_pages, v_pages, pooled, tail; // sparse attention
    DeviceBuffer history;                        // PLE
};

struct SequenceState {
    std::uint32_t position = 0;
    NgramContext context;
    std::vector<LayerState> layers;
};

// What one device holds for the layers it runs: activation planes sized for a full chunk and a
// scoped workspace.
struct RankState {
    std::size_t rank    = 0;
    cudaStream_t stream = nullptr;
    DeviceBuffer activations;
    std::unique_ptr<DeviceArena> workspace;
    DeviceBuffer block_table;
    std::unique_ptr<PinnedHostBuffer> staging;
    cudaEvent_t staged = nullptr;
    cudaEvent_t done   = nullptr;
    // Activation planes (capacity: prefill_chunk tokens).
    float* stack            = nullptr;
    std::int32_t* ids       = nullptr;
    std::int32_t* positions = nullptr;
    void* mixed             = nullptr; // BF16 [H, T]
    float* inject           = nullptr; // [4, T]
    void* y                 = nullptr; // BF16 or FP32 [H, T]
    void* a                 = nullptr; // generic planes
    void* b                 = nullptr;
    void* c                 = nullptr;
    void* d                 = nullptr;
    void* e                 = nullptr;
    void* f                 = nullptr;
    void* g                 = nullptr;
    std::int32_t* route_ids = nullptr;
    float* route_weights    = nullptr;
    float* route_shared     = nullptr;
    std::int32_t* selected  = nullptr;
    std::int32_t* counts    = nullptr;
    void* rows              = nullptr; // staged n-gram rows
    void* logits            = nullptr; // head rank only
};

} // namespace

struct Executor::Impl {
    const Model& model;
    DeviceContext& device;
    ExecutorOptions options;
    TextConfig config;
    NgramHashConstants ngram;
    std::unique_ptr<NgramTableReader> table;
    Weight embedding_table;
    Projection head;
    HcPlan final_mixer;
    std::vector<LayerPlan> layers;
    std::vector<RankState> ranks;
    std::vector<SequenceState> sequences;
    std::uint32_t pages          = 0; // KV pages per sequence and attention layer
    std::uint32_t pooled_slots   = 0; // indexer blocks per sequence and attention layer
    std::uint32_t max_logit_rows = 0;
    ExecutorMemory memory;
    std::vector<std::uint64_t> row_ids;
    std::vector<std::uint8_t> row_bytes_host;

    Impl(const Model& m, DeviceContext& d, ExecutorOptions o)
        : model(m), device(d), options(std::move(o)), config(m.config()),
          ngram(derive_ngram_hash_constants(config.ngram)) {
        if (options.prefill_chunk == 0 || options.prefill_chunk > 4096 || options.sequences == 0 ||
            options.max_context == 0) {
            throw std::invalid_argument("qwen4_exp executor: chunk must be in [1, 4096]");
        }
        table = std::make_unique<NgramTableReader>(options.ngram.layout, options.ngram_residency);
        max_logit_rows = std::min<std::uint32_t>(options.prefill_chunk, 512);
        pages          = (options.max_context + kPagedKVPageSize - 1) / kPagedKVPageSize;
        pooled_slots   = options.max_context / config.indexer_compress_ratio + 1;
        plan_weights();
        allocate_ranks();
        allocate_sequences();
    }

    ~Impl() {
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            if (rank.staged != nullptr) { cudaEventDestroy(rank.staged); }
            if (rank.done != nullptr) { cudaEventDestroy(rank.done); }
        }
    }

    void plan_weights() {
        const auto& w   = model.weights();
        embedding_table = native_weight(model.weight(w.token_embedding).view);
        {
            RankBinding bind(device, model.head_rank());
            head        = make_projection(model, {w.output_head});
            final_mixer = make_hc(model, w.final_mixer, config);
        }
        for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
            const LayerWeights& lw = w.layers[i];
            LayerPlan plan;
            plan.rank = model.stages().placement(i).stage;
            RankBinding bind(device, plan.rank);
            plan.attn_hc = make_hc(model, lw.attn_hc, config);
            plan.mlp_hc  = make_hc(model, lw.mlp_hc, config);
            if (const auto* g = std::get_if<GdnWeights>(&lw.mixer)) {
                GdnPlan gdn;
                gdn.qkv             = make_projection(model, {g->query, g->key, g->value});
                gdn.z               = make_projection(model, {g->z});
                gdn.a               = make_projection(model, {g->a_projection});
                gdn.b               = make_projection(model, {g->b_projection});
                gdn.output          = make_projection(model, {g->output});
                const auto channels = static_cast<std::int32_t>(gdn.qkv.rows);
                gdn.convolution =
                    direct(model, g->convolution,
                           {channels, static_cast<std::int32_t>(config.linear_conv_kernel_dim)});
                const auto heads = static_cast<std::int32_t>(config.linear_num_value_heads);
                gdn.a_log        = direct(model, g->a_log, {heads}, DType::FP32);
                gdn.dt_bias      = direct(model, g->dt_bias, {heads}, DType::FP32);
                gdn.norm = direct(model, g->norm,
                                  {static_cast<std::int32_t>(config.linear_value_head_dim)});
                plan.gdn = std::move(gdn);
            } else {
                const auto& a = std::get<SparseAttentionWeights>(lw.mixer);
                QsaPlan qsa;
                qsa.query            = make_projection(model, {a.query});
                qsa.gate             = make_projection(model, {a.gate});
                qsa.key              = make_projection(model, {a.key});
                qsa.value            = make_projection(model, {a.value});
                qsa.index            = make_projection(model, {a.index_query, a.index_key});
                qsa.output           = make_projection(model, {a.output});
                const auto d         = static_cast<std::int32_t>(config.head_dim);
                const auto di        = static_cast<std::int32_t>(config.indexer_head_dim);
                qsa.query_norm       = direct(model, a.query_norm, {d});
                qsa.key_norm         = direct(model, a.key_norm, {d});
                qsa.index_query_norm = direct(model, a.index_query_norm, {di});
                qsa.index_key_norm   = direct(model, a.index_key_norm, {di});
                plan.qsa             = std::move(qsa);
            }
            if (lw.ple) {
                PlePlan ple;
                ple.key          = make_projection(model, {lw.ple->key});
                ple.value        = make_projection(model, {lw.ple->value});
                const auto width = static_cast<std::int32_t>(config.hc_count * config.hidden_size);
                ple.norm_key     = direct(model, lw.ple->norm_key, {width});
                ple.norm_query   = direct(model, lw.ple->norm_query, {width});
                ple.norm_conv    = direct(model, lw.ple->norm_conv, {width});
                ple.convolution =
                    direct(model, lw.ple->convolution,
                           {static_cast<std::int32_t>(config.ple_conv_kernel_size), width});
                plan.ple = std::move(ple);
            }
            const auto h = static_cast<std::int32_t>(config.hidden_size);
            plan.moe.router =
                direct(model, lw.moe.router, {h, static_cast<std::int32_t>(config.num_experts)});
            plan.moe.shared_gate       = direct(model, lw.moe.shared_score, {h});
            plan.moe.gate              = make_table(model, lw.moe.gate);
            plan.moe.up                = make_table(model, lw.moe.up);
            plan.moe.down              = make_table(model, lw.moe.down);
            plan.moe.shared_gate_table = make_table(model, std::span(&lw.moe.shared_gate, 1));
            plan.moe.shared_up_table   = make_table(model, std::span(&lw.moe.shared_up, 1));
            plan.moe.shared_down_table = make_table(model, std::span(&lw.moe.shared_down, 1));
            layers.push_back(std::move(plan));
        }
    }

    [[nodiscard]] std::size_t workspace_bytes(std::size_t rank) const {
        const std::int32_t t = static_cast<std::int32_t>(options.prefill_chunk);
        std::size_t bytes    = ops::hyper_connection_read_workspace_bytes(
            static_cast<std::int32_t>(config.hc_count),
            static_cast<std::int32_t>(config.hidden_size),
            static_cast<std::int32_t>(config.hc_lowrank), t);
        bytes = std::max(bytes, ops::moe_experts_gguf_workspace_bytes(t));
        for (const auto& plan : layers) {
            if (plan.rank != rank) { continue; }
            if (plan.gdn) {
                for (const Projection* p : {&plan.gdn->qkv, &plan.gdn->z, &plan.gdn->a,
                                            &plan.gdn->b, &plan.gdn->output}) {
                    bytes = std::max(bytes, projection_workspace(*p, t));
                }
                bytes =
                    std::max(bytes, ops::gated_delta_net_workspace_capacity_bytes(
                                        static_cast<std::int32_t>(config.linear_num_key_heads),
                                        static_cast<std::int32_t>(config.linear_num_value_heads),
                                        true, 1, t));
            }
            if (plan.qsa) {
                for (const Projection* p :
                     {&plan.qsa->query, &plan.qsa->gate, &plan.qsa->key, &plan.qsa->value,
                      &plan.qsa->index, &plan.qsa->output}) {
                    bytes = std::max(bytes, projection_workspace(*p, t));
                }
                bytes = std::max(bytes, ops::qsa_indexer_select_workspace_bytes(
                                            t, static_cast<std::int32_t>(pooled_slots)));
            }
            if (plan.ple) {
                bytes = std::max({bytes, projection_workspace(plan.ple->key, t),
                                  projection_workspace(plan.ple->value, t),
                                  ops::ple_inject_workspace_bytes(t)});
            }
        }
        if (rank == model.head_rank()) {
            bytes = std::max(bytes,
                             projection_workspace(head, static_cast<std::int32_t>(max_logit_rows)));
        }
        return bytes + (16U << 20);
    }

    void allocate_ranks() {
        const std::uint64_t t     = options.prefill_chunk;
        const std::uint64_t h     = config.hidden_size;
        const std::uint64_t width = std::uint64_t(config.hc_count) * h;
        const std::uint64_t plane = std::max<std::uint64_t>(width, 6144); // widest BF16 plane rows
        const std::uint64_t row_b = ops::ngram_row_bytes(options.ngram.format);
        for (std::size_t r = 0; r < device.size(); ++r) {
            RankBinding bind(device, r);
            RankState rank;
            rank.rank                                            = r;
            rank.stream                                          = device.rank(r).stream;
            std::vector<std::pair<void**, std::uint64_t>> planes = {
                {reinterpret_cast<void**>(&rank.stack), width * t * 4},
                {reinterpret_cast<void**>(&rank.ids), t * 4},
                {reinterpret_cast<void**>(&rank.positions), t * 4},
                {&rank.mixed, h * t * 2},
                {reinterpret_cast<void**>(&rank.inject), std::uint64_t(config.hc_count) * t * 4},
                {&rank.y, h * t * 4},
                {&rank.a, plane * t * 2},
                {&rank.b, plane * t * 2},
                {&rank.c, plane * t * 2},
                {&rank.d, plane * t * 2},
                {&rank.e, plane * t * 2},
                {&rank.f, plane * t * 2},
                {&rank.g, plane * t * 4},
                {reinterpret_cast<void**>(&rank.route_ids),
                 std::uint64_t(config.num_experts_per_tok) * t * 4},
                {reinterpret_cast<void**>(&rank.route_weights),
                 std::uint64_t(config.num_experts_per_tok) * t * 4},
                {reinterpret_cast<void**>(&rank.route_shared), t * 4},
                {reinterpret_cast<void**>(&rank.selected),
                 std::uint64_t(config.indexer_block_budget()) * t * 4},
                {reinterpret_cast<void**>(&rank.counts), t * 4},
                {&rank.rows, std::uint64_t(config.ngram_heads()) * t * row_b},
            };
            if (r == model.head_rank()) {
                planes.push_back(
                    {&rank.logits, std::uint64_t(config.vocab_size) * max_logit_rows * 2});
            }
            std::uint64_t total = 0;
            for (const auto& [slot, bytes] : planes) { total += round_up(bytes); }
            rank.activations     = DeviceBuffer(total);
            std::uint64_t offset = 0;
            for (const auto& [slot, bytes] : planes) {
                *slot = static_cast<std::byte*>(rank.activations.p) + offset;
                offset += round_up(bytes);
            }
            rank.workspace = std::make_unique<DeviceArena>(workspace_bytes(r));
            std::vector<std::int32_t> identity(pages);
            std::iota(identity.begin(), identity.end(), 0);
            rank.block_table = DeviceBuffer(identity.size() * 4);
            rank.block_table.copy_from_host(identity.data(), identity.size() * 4);
            rank.staging = std::make_unique<PinnedHostBuffer>(
                round_up(t * 8) + round_up(std::uint64_t(config.ngram_heads()) * t * row_b));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.staged, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.done, cudaEventDisableTiming));
            memory.workspace_bytes += total + rank.workspace->capacity();
            ranks.push_back(std::move(rank));
        }
    }

    void allocate_sequences() {
        const std::uint64_t width = std::uint64_t(config.hc_count) * config.hidden_size;
        for (std::uint32_t s = 0; s < options.sequences; ++s) {
            SequenceState sequence;
            for (const auto& plan : layers) {
                RankBinding bind(device, plan.rank);
                LayerState state;
                if (plan.gdn) {
                    const std::uint64_t dk = config.linear_key_head_dim,
                                        dv = config.linear_value_head_dim;
                    state.ssm  = DeviceBuffer(dk * dv * config.linear_num_value_heads * 4);
                    state.conv = DeviceBuffer(std::uint64_t(plan.gdn->qkv.rows) *
                                              (config.linear_conv_kernel_dim - 1) * 2);
                    memory.state_bytes += state.ssm.bytes + state.conv.bytes;
                }
                if (plan.qsa) {
                    const std::uint64_t page = std::uint64_t(config.head_dim) * kPagedKVPageSize *
                                               config.num_key_value_heads * 2;
                    state.k_pages            = DeviceBuffer(page * pages);
                    state.v_pages            = DeviceBuffer(page * pages);
                    state.pooled =
                        DeviceBuffer(std::uint64_t(config.indexer_head_dim) * pooled_slots * 4);
                    state.tail = DeviceBuffer(std::uint64_t(config.indexer_head_dim) *
                                              (config.indexer_compress_ratio - 1) * 4);
                    memory.state_bytes += state.k_pages.bytes + state.v_pages.bytes +
                                          state.pooled.bytes + state.tail.bytes;
                }
                if (plan.ple) {
                    state.history = DeviceBuffer(width * (config.ple_conv_kernel_size - 1) *
                                                 config.ngram.ngram_size * 4);
                    memory.state_bytes += state.history.bytes;
                }
                sequence.layers.push_back(std::move(state));
            }
            sequences.push_back(std::move(sequence));
            reset(s);
        }
    }

    void reset(std::uint32_t s) {
        auto& sequence    = sequences.at(s);
        sequence.position = 0;
        sequence.context  = NgramContext::sequence_start(ngram, config.eos_token_id);
        for (std::size_t i = 0; i < layers.size(); ++i) {
            RankBinding bind(device, layers[i].rank);
            const cudaStream_t stream = ranks[layers[i].rank].stream;
            for (DeviceBuffer* buffer :
                 {&sequence.layers[i].ssm, &sequence.layers[i].conv, &sequence.layers[i].pooled,
                  &sequence.layers[i].tail, &sequence.layers[i].history}) {
                if (buffer->p != nullptr) {
                    CUDA_CHECK(cudaMemsetAsync(buffer->p, 0, buffer->bytes, stream));
                }
            }
        }
    }

    // Moves the stack's first `tokens` columns from one rank to another, ordered after the
    // source's work.
    void cross(std::size_t from, std::size_t to, std::int32_t tokens) {
        RankState& source = ranks[from];
        RankState& target = ranks[to];
        {
            RankBinding bind(device, from);
            CUDA_CHECK(cudaEventRecord(source.done, source.stream));
        }
        RankBinding bind(device, to);
        CUDA_CHECK(cudaStreamWaitEvent(target.stream, source.done, 0));
        const std::size_t bytes =
            std::size_t(config.hc_count) * config.hidden_size * std::size_t(tokens) * 4;
        CUDA_CHECK(cudaMemcpyPeerAsync(target.stack, device.rank(to).device, source.stack,
                                       device.rank(from).device, bytes, target.stream));
    }

    void stage_inputs(SequenceState& sequence, std::span<const std::int32_t> tokens) {
        const auto t = static_cast<std::int32_t>(tokens.size());
        // Host side first: positions, and the n-gram rows of the PLE layer's tokens.
        std::vector<std::int32_t> positions(tokens.size());
        std::iota(positions.begin(), positions.end(), static_cast<std::int32_t>(sequence.position));
        const std::size_t heads = config.ngram_heads();
        row_ids.resize(tokens.size() * heads);
        ngram_row_ids(ngram, tokens, config.eos_token_id, config.vocab_size, sequence.context,
                      row_ids);
        const std::size_t row_b = ops::ngram_row_bytes(options.ngram.format);
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            // The staging buffer is rewritten only once the previous upload has left it.
            CUDA_CHECK(cudaEventSynchronize(rank.staged));
            auto* base = static_cast<std::byte*>(rank.staging->data());
            std::memcpy(base, tokens.data(), tokens.size() * 4);
            std::memcpy(base + t * 4, positions.data(), positions.size() * 4);
            CUDA_CHECK(cudaMemcpyAsync(rank.ids, base, tokens.size() * 4, cudaMemcpyHostToDevice,
                                       rank.stream));
            CUDA_CHECK(cudaMemcpyAsync(rank.positions, base + t * 4, positions.size() * 4,
                                       cudaMemcpyHostToDevice, rank.stream));
            const bool ple_here =
                std::any_of(layers.begin(), layers.end(),
                            [&](const LayerPlan& p) { return p.ple && p.rank == rank.rank; });
            if (ple_here) {
                auto* rows = base + round_up(std::uint64_t(options.prefill_chunk) * 8);
                table->read_rows(row_ids, std::span(reinterpret_cast<std::uint8_t*>(rows),
                                                    row_ids.size() * row_b));
                CUDA_CHECK(cudaMemcpyAsync(rank.rows, rows, row_ids.size() * row_b,
                                           cudaMemcpyHostToDevice, rank.stream));
            }
            CUDA_CHECK(cudaEventRecord(rank.staged, rank.stream));
        }
    }

    void run_gdn(const LayerPlan& plan, LayerState& state, RankState& rank, std::int32_t t) {
        const GdnPlan& g     = *plan.gdn;
        WorkspaceArena& ws   = *rank.workspace;
        const cudaStream_t s = rank.stream;
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const auto kd =
            static_cast<std::int32_t>(config.linear_num_key_heads * config.linear_key_head_dim);
        const auto vd =
            static_cast<std::int32_t>(config.linear_num_value_heads * config.linear_value_head_dim);
        const auto nk = static_cast<std::int32_t>(config.linear_num_key_heads);
        const auto nv = static_cast<std::int32_t>(config.linear_num_value_heads);
        const auto dh = static_cast<std::int32_t>(config.linear_value_head_dim);
        const Tensor mixed(rank.mixed, DType::BF16, {h, t});
        Tensor qkv(rank.a, DType::BF16, {g.qkv.rows, t});
        Tensor z(rank.b, DType::BF16, {vd, t});
        Tensor ga(rank.c, DType::BF16, {nv, t});
        Tensor gb(static_cast<std::byte*>(rank.c) + round_up(std::uint64_t(nv) * t * 2),
                  DType::BF16, {nv, t});
        project(g.qkv, mixed, qkv, ws, s);
        project(g.z, mixed, z, ws, s);
        project(g.a, mixed, ga, ws, s);
        project(g.b, mixed, gb, ws, s);
        // Convolution in place of the projections' q/k/v planes.
        auto* conv_out = static_cast<std::byte*>(rank.d);
        Tensor q(conv_out, DType::BF16, {kd, t});
        Tensor k(conv_out + round_up(std::uint64_t(kd) * t * 2), DType::BF16, {kd, t});
        Tensor v(rank.e, DType::BF16, {vd, t});
        Tensor conv_state(
            state.conv.p, DType::BF16,
            {g.qkv.rows, static_cast<std::int32_t>(config.linear_conv_kernel_dim - 1)});
        ops::causal_conv1d_silu_split(qkv, g.convolution, conv_state, conv_state, q, k, v, s);
        Tensor gate(rank.g, DType::FP32, {nv, t});
        Tensor beta(static_cast<std::byte*>(rank.g) + round_up(std::uint64_t(nv) * t * 4),
                    DType::FP32, {nv, t});
        ops::gdn_gating(ga, gb, g.a_log, g.dt_bias, gate, beta, s);
        Tensor q3(q.data, DType::BF16, {dh, nk, t}), k3(k.data, DType::BF16, {dh, nk, t});
        Tensor v3(v.data, DType::BF16, {dh, nv, t});
        Tensor ssm(state.ssm.p, DType::FP32, {dh, dh, nv});
        Tensor o(rank.f, DType::BF16, {dh, nv, t});
        {
            auto scope = ws.scope();
            ops::gated_delta_net(q3, k3, v3, gate, beta, 1.0F / std::sqrt(float(dh)), true, ws, ssm,
                                 o, s);
        }
        Tensor rows_o(rank.f, DType::BF16, {dh, nv * t});
        Tensor rows_z(rank.b, DType::BF16, {dh, nv * t});
        Tensor gated_rows(rank.a, DType::BF16, {dh, nv * t});
        ops::gated_rmsnorm(rows_o, g.norm, rows_z, ops::GateActivation::Sigmoid,
                           config.rms_norm_eps, gated_rows, s);
        const Tensor gated(rank.a, DType::BF16, {vd, t});
        Tensor y(rank.y, DType::BF16, {h, t});
        project(g.output, gated, y, ws, s);
    }

    void run_qsa(const LayerPlan& plan, LayerState& state, RankState& rank, std::int32_t t,
                 std::int32_t first) {
        const QsaPlan& a     = *plan.qsa;
        WorkspaceArena& ws   = *rank.workspace;
        const cudaStream_t s = rank.stream;
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const auto d         = static_cast<std::int32_t>(config.head_dim);
        const auto nq        = static_cast<std::int32_t>(config.num_attention_heads);
        const auto nk        = static_cast<std::int32_t>(config.num_key_value_heads);
        const Tensor mixed(rank.mixed, DType::BF16, {h, t});
        Tensor q(rank.a, DType::BF16, {d * nq, t});
        Tensor gate(rank.b, DType::BF16, {d * nq, t});
        auto* kv = static_cast<std::byte*>(rank.c);
        Tensor k(kv, DType::BF16, {d * nk, t});
        Tensor v(kv + round_up(std::uint64_t(d) * nk * t * 2), DType::BF16, {d * nk, t});
        Tensor index(kv + 2 * round_up(std::uint64_t(d) * nk * t * 2), DType::BF16,
                     {a.index.rows, t});
        project(a.query, mixed, q, ws, s);
        project(a.gate, mixed, gate, ws, s);
        project(a.key, mixed, k, ws, s);
        project(a.value, mixed, v, ws, s);
        project(a.index, mixed, index, ws, s);
        Tensor q3(rank.a, DType::BF16, {d, nq, t}), k3(k.data, DType::BF16, {d, nk, t});
        Tensor qo(rank.d, DType::BF16, {d, nq, t});
        Tensor ko(rank.e, DType::BF16, {d, nk, t});
        const Tensor positions(rank.positions, DType::I32, {t});
        ops::rmsnorm_rope(positions, a.query_norm, a.key_norm, q3, k3, qo, ko, s);
        PagedKVLayerView cache{};
        cache.k_pages = Tensor(
            state.k_pages.p, DType::BF16,
            {d, static_cast<std::int32_t>(kPagedKVPageSize), nk, static_cast<std::int32_t>(pages)});
        cache.v_pages = Tensor(
            state.v_pages.p, DType::BF16,
            {d, static_cast<std::int32_t>(kPagedKVPageSize), nk, static_cast<std::int32_t>(pages)});
        cache.block_table =
            Tensor(rank.block_table.p, DType::I32, {static_cast<std::int32_t>(pages)});
        cache.head_dim     = d;
        cache.num_kv_heads = nk;
        cache.storage      = KvCacheStorage::BFloat16;
        const Tensor v3(v.data, DType::BF16, {d, nk, t});
        ops::kv_cache_append(ko, v3, positions, cache, s);
        const auto di = static_cast<std::int32_t>(config.indexer_head_dim);
        Tensor pooled(state.pooled.p, DType::FP32, {di, static_cast<std::int32_t>(pooled_slots)});
        Tensor tail(state.tail.p, DType::FP32,
                    {di, static_cast<std::int32_t>(config.indexer_compress_ratio - 1)});
        const ops::QsaIndexerWeights iw{&a.index_query_norm, &a.index_key_norm};
        ops::qsa_indexer_append(index, first, iw, config.rms_norm_eps, pooled, tail, s);
        Tensor selected(rank.selected, DType::I32,
                        {static_cast<std::int32_t>(config.indexer_block_budget()), t});
        Tensor counts(rank.counts, DType::I32, {t});
        {
            auto scope = ws.scope();
            ops::qsa_indexer_select(index, first, iw, config.rms_norm_eps, pooled, ws, selected,
                                    counts, s);
        }
        Tensor attention(rank.f, DType::BF16, {d, nq, t});
        ops::sparse_softmax_attention(qo, first, selected, counts, cache,
                                      1.0F / std::sqrt(float(d)), attention, s);
        Tensor flat(rank.f, DType::BF16, {d * nq, t});
        ops::sigmoid_mul(gate, flat, s);
        Tensor y(rank.y, DType::BF16, {h, t});
        project(a.output, flat, y, ws, s);
    }

    void run_ple(const LayerPlan& plan, LayerState& state, RankState& rank, std::int32_t t,
                 Tensor& stack) {
        const PlePlan& p     = *plan.ple;
        WorkspaceArena& ws   = *rank.workspace;
        const cudaStream_t s = rank.stream;
        const auto heads     = static_cast<std::int32_t>(config.ngram_heads());
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const auto width     = static_cast<std::int32_t>(config.hc_count * config.hidden_size);
        const Tensor rows(
            rank.rows, DType::U8,
            {static_cast<std::int32_t>(ops::ngram_row_bytes(options.ngram.format)), heads * t});
        Tensor embedding(rank.a, DType::BF16, {static_cast<std::int32_t>(config.ple_embed_dim), t});
        ops::ngram_embed_rows(rows, options.ngram.format, heads, embedding, s);
        Tensor key(rank.g, DType::BF16, {width, t});
        Tensor value(rank.b, DType::BF16, {h, t});
        project(p.key, embedding, key, ws, s);
        project(p.value, embedding, value, ws, s);
        Tensor history(state.history.p, DType::FP32,
                       {width, static_cast<std::int32_t>((config.ple_conv_kernel_size - 1) *
                                                         config.ngram.ngram_size)});
        auto scope = ws.scope();
        ops::ple_inject(stack, key, value,
                        {&p.norm_key, &p.norm_query, &p.norm_conv, &p.convolution},
                        config.rms_norm_eps, history, ws, s);
    }

    void run_moe(const LayerPlan& plan, RankState& rank, std::int32_t t) {
        const MoePlan& m     = plan.moe;
        WorkspaceArena& ws   = *rank.workspace;
        const cudaStream_t s = rank.stream;
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const auto top       = static_cast<std::int32_t>(config.num_experts_per_tok);
        const Tensor mixed(rank.mixed, DType::BF16, {h, t});
        Tensor ids(rank.route_ids, DType::I32, {top, t});
        Tensor weights(rank.route_weights, DType::FP32, {top, t});
        Tensor shared(rank.route_shared, DType::FP32, {t});
        ops::moe_route(mixed, m.router, m.shared_gate, ids, weights, shared, s);
        Tensor y(rank.y, DType::FP32, {h, t});
        auto scope = ws.scope();
        ops::moe_experts_gguf(mixed, ids, weights, shared, m.banks(), ws, y, s);
    }

    void forward(std::uint32_t s, std::span<const std::int32_t> tokens, std::uint32_t logit_rows) {
        auto& sequence = sequences.at(s);
        const auto t   = static_cast<std::int32_t>(tokens.size());
        if (t == 0 || tokens.size() > options.prefill_chunk) {
            throw std::invalid_argument("qwen4_exp forward: 1..prefill_chunk tokens per call");
        }
        if (logit_rows == 0 || logit_rows > tokens.size() || logit_rows > max_logit_rows) {
            throw std::invalid_argument("qwen4_exp forward: invalid logit rows");
        }
        if (sequence.position + tokens.size() > options.max_context) {
            throw std::invalid_argument("qwen4_exp forward: the sequence exceeds max_context");
        }
        for (const auto token : tokens) {
            if (token < 0 || std::uint32_t(token) >= config.vocab_size) {
                throw std::invalid_argument("qwen4_exp forward: token outside the vocabulary");
            }
        }
        const auto first = static_cast<std::int32_t>(sequence.position);
        stage_inputs(sequence, tokens);
        const auto h        = static_cast<std::int32_t>(config.hidden_size);
        const auto hc       = static_cast<std::int32_t>(config.hc_count);
        std::size_t current = 0;
        {
            RankState& rank = ranks[0];
            RankBinding bind(device, 0);
            const Tensor ids(rank.ids, DType::I32, {t});
            Tensor x(rank.mixed, DType::BF16, {h, t});
            ops::embedding(ids, embedding_table, x, rank.stream);
            Tensor stack(rank.stack, DType::FP32, {h, hc, t});
            ops::hyper_connection_expand(x, stack, rank.stream);
        }
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const LayerPlan& plan = layers[i];
            if (plan.rank != current) {
                cross(current, plan.rank, t);
                current = plan.rank;
            }
            RankState& rank = ranks[current];
            RankBinding bind(device, current);
            LayerState& state = sequence.layers[i];
            Tensor stack(rank.stack, DType::FP32, {h, hc, t});
            if (plan.ple) { run_ple(plan, state, rank, t, stack); }
            Tensor mixed(rank.mixed, DType::BF16, {h, t});
            Tensor inject(rank.inject, DType::FP32, {hc, t});
            {
                auto scope = rank.workspace->scope();
                ops::hyper_connection_read(stack, plan.attn_hc.weights(), config.rms_norm_eps,
                                           *rank.workspace, mixed, &inject, rank.stream);
            }
            if (plan.gdn) {
                run_gdn(plan, state, rank, t);
            } else {
                run_qsa(plan, state, rank, t, first);
            }
            ops::hyper_connection_write(stack, Tensor(rank.y, DType::BF16, {h, t}), inject,
                                        rank.stream);
            {
                auto scope = rank.workspace->scope();
                ops::hyper_connection_read(stack, plan.mlp_hc.weights(), config.rms_norm_eps,
                                           *rank.workspace, mixed, &inject, rank.stream);
            }
            run_moe(plan, rank, t);
            ops::hyper_connection_write(stack, Tensor(rank.y, DType::FP32, {h, t}), inject,
                                        rank.stream);
        }
        const std::size_t head_rank = model.head_rank();
        if (current != head_rank) { cross(current, head_rank, t); }
        RankState& rank = ranks[head_rank];
        RankBinding bind(device, head_rank);
        const auto n = static_cast<std::int32_t>(logit_rows);
        const Tensor last(rank.stack + std::size_t(hc) * h * (t - n), DType::FP32, {h, hc, n});
        Tensor mixed(rank.mixed, DType::BF16, {h, n});
        {
            auto scope = rank.workspace->scope();
            ops::hyper_connection_read(last, final_mixer.weights(), config.rms_norm_eps,
                                       *rank.workspace, mixed, nullptr, rank.stream);
        }
        Tensor logits(rank.logits, DType::BF16, {static_cast<std::int32_t>(config.vocab_size), n});
        project(head, mixed, logits, *rank.workspace, rank.stream);
        sequence.position += static_cast<std::uint32_t>(t);
    }
};

Executor::Executor(const Model& model, DeviceContext& device, ExecutorOptions options)
    : impl_(std::make_unique<Impl>(model, device, std::move(options))) {}

Executor::~Executor() = default;

const ExecutorOptions& Executor::options() const noexcept { return impl_->options; }

ExecutorMemory Executor::memory() const noexcept { return impl_->memory; }

void Executor::reset(std::uint32_t sequence) { impl_->reset(sequence); }

std::uint32_t Executor::position(std::uint32_t sequence) const {
    return impl_->sequences.at(sequence).position;
}

void Executor::forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                       std::uint32_t logit_rows) {
    impl_->forward(sequence, tokens, logit_rows);
}

Tensor Executor::logits(std::uint32_t rows) const {
    return Tensor(
        impl_->ranks[impl_->model.head_rank()].logits, DType::BF16,
        {static_cast<std::int32_t>(impl_->config.vocab_size), static_cast<std::int32_t>(rows)});
}

std::size_t Executor::head_rank() const noexcept { return impl_->model.head_rank(); }

cudaStream_t Executor::head_stream() const noexcept {
    return impl_->ranks[impl_->model.head_rank()].stream;
}

} // namespace ninfer::models::qwen4_exp
