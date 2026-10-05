# NInfer-all

One line of [NInfer](https://github.com/Neroued/ninfer) for the RTX 3090, RTX 4090, RTX 5090 and RTX
PRO 6000 Blackwell, consolidated from the forks that carry it and extended with this repository's
own work. The base is the `master` of
[ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090): v0.12.0 (with its prompt
grafts, `/slots` session persistence, the effective thinking budget and worker recovery) and
the multi-GPU pipeline stages, most of both written by [Warlax](https://github.com/WarlaxZ), on the
line [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) started from Neroued's NInfer.
On top of it come patches from [TertiumOrganum1/ninfer-3090](https://github.com/TertiumOrganum1/ninfer-3090),
ideas from [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) and its
contributors, open pull requests to [Neroued/ninfer](https://github.com/Neroued/ninfer), and work by
[IMGillusion](https://github.com/IMGillusion/ninfer-disk-kv),
[Mirko Covizzi](https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile),
Ian Ranson ([Wallawalla47](https://github.com/Wallawalla47/ninfer-custom)),
[tmark00](https://github.com/tmark00/ninfer) and David Oelfke
([gzenz/ninfer](https://github.com/gzenz/ninfer)). Each change keeps its
author; the [maintainer map](docs/maintainer/consolidated-line.md) lists them with the files they
touch.

Everything the engine does beyond the list below (building, packages, serving APIs, supported models,
flags) is described in the original READMEs of
**[NInfer-3090](https://github.com/ashalliants/ninfer-3090#readme)** and
**[NInfer-4090](https://github.com/UDPSendToFailed/ninfer-4090#readme)**. New features that change
numbers or serving behaviour are opt-in, with three exceptions: the routes a card's measured device
profile picks (`--device-profile off` keeps the compiled tables), prefill chunks rounded to whole
waves of the card's SMs (`NINFER_PREFILL_ALIGN=0` keeps the requested chunk), and
TertiumOrganum1's ternary prefill tile (`NINFER_T2_A8_TILE=off` restores the kernel it replaces).

## Highlights

Measured in September 2026 on one card each, greedy, one request at a time unless the row says
otherwise; each number's setup and the full tables are in the
[reference measurements](docs/performance/reference-2026-09.md).

| | RTX 3090 | RTX 4090 | RTX 5090 | RTX PRO 6000 |
|---|---:|---:|---:|---:|
| **Ternary Bonsai 2 27B**, short chat (DFlash2, 7 drafts) | 202 tok/s | 256 tok/s | 397 tok/s | 381 tok/s |
| decode after a 261K-token document (fastest drafter) | 90 tok/s | 123 tok/s | 218 tok/s | 218 tok/s |
| time to first token for a 261K-token prompt | 215 s | 102 s | 82 s | 78 s |
| largest context, filled and all three needles found | 970,752 | 958,464 | 978,944 | 1,048,576\* |
| eight requests at once (MTP, 3 drafts), total | 551 tok/s | 824 tok/s | 1,063 tok/s | 1,155 tok/s |
| **Qwen3.8-27B**, short chat (DFlash2, 7 drafts) | 118 tok/s | 149 tok/s | 236 tok/s | 237 tok/s |
| largest context, filled and all three needles found | 417,792 | 405,504 | 872,448 | 1,048,576\* |
| eight requests at once (MTP, 3 drafts), total | 329 tok/s | 442 tok/s | 690 tok/s | 739 tok/s |

\* The engine's ceiling, which the RTX PRO 6000 (96 GB) starts with every KV storage and drafter;
filled to it, both models find two of the three needles.

- **Against the previous `master` on the same card**, a 261K-token Bonsai prompt takes 215 s instead
  of 315 s on the RTX 3090, 102 s instead of 138 s on the RTX 4090 and 82 s instead of 115 s on the
  RTX 5090, and `rk4v4` decode after it is 11 to 13% faster on the 24 GB cards. Decode at short
  context is unchanged, since it is bound by reading the weights, and Qwen3.8's 8K to 32K prompts on
  the RTX 5090 take 8 to 10% longer.
- **The RTX 3090's device profile** runs the `rk4v4` verify attention at 262K 3.2 times faster than
  the compiled route, and the fast prompt kernel with FP16 P·V takes 19 to 30% less prompt-attention
  time on all three cards.
- **Draft length.** DFlash2 with seven drafts is fastest on short answers, while after long
  documents the best count lies between three and seven; MTP runs up to fifteen drafts now but is
  fastest at three to five.
- **Past the native window.** Filled to about 880K tokens, Bonsai 2 returned all three planted codes
  on every card; at 1,048,576 tokens, which only the RTX 5090 and the RTX PRO 6000 hold, it misses the
  one at 943K.
- **RTX PRO 6000.** Its 96 GB start every configuration at the engine's 1,048,576-token ceiling with
  at least 50 GiB to spare. Against the RTX 5090 it prefills 4 to 6% faster and decodes 2 to 3% slower.

## What this line adds

- **Model suspend.** With `--model-suspend` an idle server gives its device memory back without
  exiting -- `POST /v1/models/{id}/suspend` -- and takes it again on the next request or on
  `/resume`, with its retained conversations intact: every device allocation sits at a fixed
  address, so the CUDA Graphs and caches survive, and only the live persistent bytes travel to host
  memory (free KV pages stay behind). On an RTX 3090 serving Ternary Bonsai 2 27B, a suspend takes
  the card from 7.9 GiB in use to 0.3 GiB (the CUDA context) in 0.33 s, and a resume takes 1.1 s,
  almost all of it the weights read again from a warm page cache (0.8 s with `--suspend-weights
  host`). Greedy output and prefix reuse are identical before and after, on one device, across
  pipeline stages, with MTP, with the hybrid prefix cache and with overlay Vision. See
  [Model suspend](docs/serving.md#model-suspend).
- **Several models behind one server.** Started with `--models-dir` or a llama.cpp-style
  `--models-preset` INI instead of an artifact, `ninfer-serve` is a router with llama.cpp's API:
  `GET /models`, `POST /models/load` and `/models/unload`, `GET /models/sse`, the `model` field
  choosing the model of each request, and `--models-max` loaded at once. A model started with
  `--model-suspend` makes room by going to sleep rather than unloading, and wakes with its retained
  conversations: on an RTX 3090 with two 27B models swapping in one 24 GB card, a wake took 1.8 s
  where a cold load took 11.6 s, with identical output. See
  [Several models](docs/serving.md#several-models-router).
- **llama.cpp's native endpoints.** `POST /completion` and OpenAI's legacy `POST /v1/completions`
  continue a raw prompt (text, token ids, or both mixed) with llama.cpp's response objects and
  streams, and `POST /tokenize`, `/detokenize` and `/apply-template` expose the tokenizer and the
  chat template. Greedy completion of the prompt `/apply-template` renders gives the very text
  `/v1/chat/completions` gives for the same messages. See
  [Raw-prompt completion](docs/serving.md#raw-prompt-completion-and-the-tokenizer).
- **Rerank.** `POST /v1/rerank` (Jina's and llama.cpp's shape, TEI's too) ranks documents with the
  served model as the judge, in Qwen3-Reranker's yes/no formulation, scoring each by
  P(yes) / (P(yes) + P(no)) from the answer token's exact log probabilities. See
  [Rerank](docs/serving.md#rerank).
- **GGUF block formats.** Qwen3.8-27B GGUF releases that choose a ggml quantization type per tensor,
  such as ISTA-DASLab's GSQ-RCO models, import without requantization: the converter recipe
  `qwen3_8_27b_gguf` copies every quantized tensor's blocks unchanged, and the runtime multiplies
  all fifteen dense ggml block types in place, decode and verification through a vector kernel that
  decodes each weight once for every column, prompts through llama.cpp's integer tensor-core kernel.
  MTP, DFlash2 and Vision work as with the official artifact. The 3.5-bit GSQ-RCO IQ3_S model scores
  the WikiText-2 perplexity its card states (7.071 against 7.07; the official artifact scores 7.286)
  at 10.95 GiB of weights instead of 15.9, scores 80.3% on IFBench, 100% on AIME 2025 and 2026 and
  88.4% on GPQA-Diamond (the official artifact: 77.7, 96.7, 96.7 and 87.4), and on the same card it
  decodes faster than the official artifact: 59.9 against 40.3 tok/s on an RTX 3090 and 107.5
  against 88.1 on an RTX 5090 without speculation, 146 against 109 on an RTX 4090 with MTP. See
  [GGUF block formats](docs/gguf.md).
- **Device route profiles for every GPU.** Which kernel schedule serves each operation and width
  is looked up in the card's measured profile before the compiled tables, which were tuned on one
  card. Profiles measured on the RTX 3090, 4090, 5090 and all three RTX PRO 6000 Blackwell editions
  (Workstation, Max-Q, Server) are built in; any other GPU is calibrated
  once at first start (20 to 40 seconds) and the result is saved, and `ninfer-calibrate`
  re-measures on demand. On the RTX 3090 the profile makes the `rk4v4` verify attention at 262K
  3.2 times faster; on all three it turns on FP16 accumulation of P·V (15 to 17% less time in that
  attention, perplexity unchanged) and the fast prompt kernel (19 to 30% less prompt-attention
  time). A greedy answer can then differ between a request served alone and the same request
  batched with others, at near-tied tokens, more often than before; `--device-profile off` keeps
  the compiled schedules of the previous master. See
  [device profiles](docs/device-profiles.md).
- **FP8 and NVFP4 at full speed on the default Blackwell build.** Every `120a` build compiles the FP8
  A8 and NVFP4 W4A4 tensor-core units, so FP8 and NVFP4 weights run their own routes on the
  `mma.sync` compatibility path as well. Before, an NVFP4 artifact failed at startup there and FP8
  weights prefilled through a dequantizing route. On an RTX PRO 6000 the Qwen3.8-27B NVFP4/FP8
  artifact prefills 4,096 tokens at 11,822 tok/s, within 1.4% of a native build, and the
  Qwen3.6-35B-A3B NVFP4 artifact at 30,938 tok/s.
- **Faster attention at long context.** The INT8-family small-T kernel gains tiers that split the
  QK product across producer warps and fetch the next key tile a whole iteration ahead; the fast
  prompt kernel now serves `rk8v4`, `rk4v4`, `rk4v4-e8` and `rk2v4-e8`; every prompt kernel's
  prefill chunks are sized to whole waves of the card's SMs, as Ian Ranson's fast kernel did for its
  own. On an RTX 3090 a 131K prompt with `rk8v4` takes 76 s instead of 101 s on the previous master.
- **MTP up to fifteen drafts.** Draft windows past eight verify columns build one CUDA Graph per
  context band, so `--spec mtp --draft-tokens 10..15` starts (it failed on graph update before).
- **BF16 KV with graphs.** MTP with the default BF16 KV cache failed at startup on the previous
  master, and so did Qwen3.8 without speculation at 512 and 1,024 tokens of context: the CUDA Graph
  planner shared one executable between windows where BF16 takes the prompt kernel (up to 128 keys)
  and windows where it takes small-T. The planner now asks the attention op which route each
  captured call takes.
- **Parallel query tiles (opt-in).** A single-row verification or short prefill over an INT8-family
  cache can run its 9 to 64 columns as parallel tiles of one split-KV launch after one batched KV
  append, instead of serial chunks: the device profile key `attn_parallel_tiles` (calibrated) or
  `NINFER_ATTN_PARALLEL_TILES=1`.
- **Branch anchors (opt-in).** `--branch-anchors` captures a request where its prompt stops
  matching a retained conversation, so an edited or forked conversation resumes from there.
- **Blackwell kernels.** FP8 A8 projections use the block-scaled MX FP8 MMA with TMA split-K
  schedules; an NVFP4 KV cache prefills past 2048 visible keys with QK on FP4 tensor cores under
  `--fast-prefill-kernel` or the profile's `attn_prompt_fast`; the unified kernels launch as
  programmatic dependents in captured graphs; with CUDA 13.2 the FP8 and NVFP4 A16 operands widen
  natively. All of it compiles only into `120a` builds.
- **Qwen3.8-Flash-Next.** ISTA-DASLab's GSQ-RCO GGUF releases of the 125B-parameter MoE (512
  experts, about 6B active) convert without requantization (`qwen3_8_flash_next_gguf`), with their
  n-gram table in the same file or in a table artifact every release shares, and run as their own
  model family: experts on the GPUs of a `--devices` pipeline (the 37.6 GB Q2_0 release decodes at 90 tok/s
  on two RTX 3090 Ti), in pinned host memory with the most used of them cached on one GPU
  (`--expert-residency host`, 48 tok/s on one RTX 3090), or left in the artifact's files and
  streamed into a GPU cache (`--expert-residency disk`, under 1 GB of RAM). It serves up to eight
  requests at once with prompt-prefix reuse and structured output, and reads images and video with
  `--vision`; MTP is not available (no release carries its layer). See
  [Qwen3.8-Flash-Next](docs/qwen3-8-flash-next.md).
- **Reference measurements** of Ternary Bonsai 2 27B and Qwen3.8-27B on the RTX 3090, 4090 and
  5090 up to the full window, the largest context each card serves and fills, every draft length
  from one to fifteen, several requests at once, and the previous master on the same hosts:
  [September 2026](docs/performance/reference-2026-09.md).
- **Ternary Bonsai 2 27B.** PrismML's [ternary Qwen3.8-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
  runs from `t2_g128_fp16` weights: 2.125 bits per weight, imported from the PQ2_0 GGUF without
  rounding, with the checkpoint's Hadamard rotations fused into the norms and gates that produce
  each projection input. The token table and the heads stay ternary, and the converter recipe
  `bonsai2_27b_ternary` adds ProCreations' Bonsai-trained MTP head and DFlash2 adapter and an exact
  proposal head.
- **Integer activations for ternary projections.** Decode, speculative verification and prompts up
  to 192 tokens use a small-T kernel over s8 activations. Longer prompts use the int8-activation
  GEMM, which pads a ragged prompt to its cheapest tile. The output, draft and proposal heads take
  the same route.
- **RTX 3090 tuning.** The rotating producers run four warps per 1024-point transform. The small-T
  attention over the INT8-family caches launches its splits in whole waves of the card's SMs. The GDN record stages its
  window in shared memory. The DFlash2 adapter of a ternary target runs in Q4.
- **DFlash2 with Vision in overlay.** An image encode can borrow the drafter's memory, so DFlash2,
  Vision and the model's whole 262,144-token window fit on one 24 GB card.
- **Serving fixes.**
  - A forced `tool_choice` opens the named call in the generation prompt, and the template's
    default thinking yields to it.
  - A context-cache store that cannot place a request fails only that request (HTTP 429) instead of
    the engine.
  - A Paged KV exhaustion names its page numbers, and three in a row mark the engine unhealthy.
  - Several context-cache fixes keep long agent sessions from re-prefilling: private reclamation,
    the demand window, and the capture search for a zero-value candidate.
- **Build.** Tests build against CUDA 13's `cudaGraphGetEdges`. Device code is compressed for size
  (CUDA 12.8 and newer), which keeps the linked binaries under 2 GiB.

Taken from [TertiumOrganum1's fork](https://github.com/TertiumOrganum1/ninfer-3090):

- **`rk4v4-e8` KV cache.** Keys are rotated as in `rk8v4` and snapped per octet to the E8 lattice
  in int4, and values keep `rk8v4`'s int4 plane (the E8-lattice KV codecs first appeared in
  NInfer-4090, by UDPSendToFailed with Daniel Parker). That is 280 bytes per token and KV head against
  408. On Ternary Bonsai 2 the whole 262,144-token window takes 2.0 GiB less, two lanes get a
  whole window each (524,288 tokens, where `rk8v4` fits 519,744) with 5.7 GiB to spare, the three
  codes planted at 131K and 250K are still found, and quick-corpus perplexity moves from 5.631 to
  5.650.
- **A 128x64 int8 tile for the ternary prefill route.** Activations are quantised per token and
  128-column group, so the int32 sum runs over a whole weight group. On Ternary Bonsai 2 prefill
  runs 33% faster at 8K, 21% at 32K and 15% at 64K than with the kernel it replaces
  (`NINFER_T2_A8_TILE=off`), and quick-corpus perplexity stays at 5.631.
- **Tool calls.** A malformed tool-call region is recovered as far as it reads, instead of leaking
  its markup into the answer.
- **Shared captures.** A shared-prefix capture whose replacement releases less than was assessed is
  abandoned. Before, the engine failed for good and answered 503 until a restart.
- **Build.** `sm_120a` (RTX 50-series) builds on the `mma.sync` compatibility path.

Taken from [NInfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) (UDPSendToFailed unless
named), re-implemented here:

- **Whole-program CUDA build** (Matt Anderson). The core and ops archives no longer build relocatable device code,
  so ptxas keeps shuffles from a computed lane inline and pipelines loads across loops: a fifth
  fewer kernels need a stack frame, and the server binary grows by a quarter.
- **Shared-memory scale reads.** The INT8-family attention kernels read their query, key and value
  scales from shared memory instead of shuffling them from a computed lane. With the whole-program
  build, the `rk8v4` attention of a verify step takes 9 to 21% less time, so on Ternary Bonsai 2
  an MTP step at 64K of context is 6.6% shorter and prefill 2 to 7% faster from 8K up, with the
  same answers.
- **`TCP_NODELAY`** on the server socket, so a streamed token leaves as soon as it is written.
- **Sigmoid, SiLU and softplus on the SFU, opt-in.** A build with `-DNINFER_SFU_SIGMOID_SILU=ON`
  evaluates sigmoid and SiLU with `ex2.approx` and a correctly rounded reciprocal instead of `expf`
  and a divide: on Ternary Bonsai 2 prefill measured 2% faster from 8K up, quick-corpus perplexity
  moves from 5.6306 to 5.6309, and MTP decode is unchanged. `-DNINFER_SFU_SOFTPLUS=ON` evaluates
  the GDN decay gate's softplus the same way, switching to a log1p series where e^x is below 1/16
  so the slow decays of long-memory heads keep their precision (quick-corpus perplexity 5.6302).
- **Keys past 262,144.** The small-T attention kernels read each page's physical index from the
  block table once a split spans more than the 64 page IDs it stages, and the visible-key limit
  rises to 1,048,576.
- **Four times the native window, with YaRN.** `--max-context` accepts up to 1,048,576 tokens on
  the 262,144-token models. Past the native window positions run plain RoPE, or YaRN with
  `--rope-yarn`, at Qwen's documented factor (`--max-context` / 262,144) and computed as Hugging
  Face and vLLM do. On Ternary Bonsai 2 with `rk2v4-e8`, a needle test (three codes at 33, 66 and
  90% of a prose document) finds all three at 500,000 tokens without the flag and two of three
  with it at 131,072, 500,000 and 1,000,000 tokens, so YaRN stays off unless plain RoPE stops
  answering. `--rope-yarn-factor F` fixes the factor instead, for every position whatever the
  window.
- **`rk2v4-e8` KV cache** (with Daniel Parker, who also proposed it upstream as Neroued/ninfer#173).
  Each 8-dimension block of a rotated, G64-scaled key is stored in two
  bytes: the nearest of E8's 240 roots, and a byte holding a 4-bit log-radius and a signed
  residual axis. That is 216 bytes per token and KV head, against 280 for `rk4v4-e8` and 408 for
  `rk8v4`, and the one format that holds 1,048,576 tokens beside Ternary Bonsai 2 on a 24 GB card
  (2.8 GiB spare without speculation, 1.3 GiB with MTP). Two lanes over the 262,144-token window
  leave 7.7 GiB spare. The price is quality: quick-corpus perplexity rises from 5.631 to 5.820
  (`rk4v4-e8`: 5.651), and DFlash2 accepts fewer drafts (51.8% against 54.4%), so decode is 4%
  slower. The three planted codes are all found at 131,072 and 250,000 tokens, and at 500,000 in
  a 1,048,576-token window.
- **D3D12-resident arenas on Windows** (with keylimesoda). A build with `-DNINFER_D3D12_RESIDENCY=ON` offers
  `--wddm-evictable-budget`: the device arenas come from a shared D3D12 heap made resident at the
  highest priority and imported into CUDA, and the KV cache is sized as if WDDM will evict other
  allocations. Untested here, since this line has no Windows machine; the code only passes a
  MinGW syntax check.

Further 4090 ideas: a server default reasoning effort, MTP draft windows up to 15, `/metrics` and
`/slots` (Sergiusz Michalik) and `/props`, a WebUI compiled in from `NINFER_WEBUI_DIR`, the block sampler's candidates in shared memory, an opt-in bf16 residual add
(`-DNINFER_BF16_RESIDUAL_ADD=ON`), vector stores in the chunked GDN prefill, and bounded split
compilation with ptxas reports as build options.

From other forks:

- **A disk tier under the Host tier** ([IMGillusion](https://github.com/IMGillusion/ninfer-disk-kv)).
  `--disk-kv-path DIR` writes an evicted conversation's KV pages and state images to CRC-checked,
  LRU files keyed by the prompt digest, which survive restarts; with `--disk-kv-restore` a new
  request whose prompt starts with a stored prefix is seeded from disk and prefills only the rest.
  On Ternary Bonsai 2 with MTP, a 17,444-token prompt evicted by two others comes back from disk
  in 1.2 s instead of 9.6 s, and in 1.0 s after a restart, with the same answer; DFlash2 and no
  speculation restore the same way. On Windows, a build with `-DNINFER_DIRECTSTORAGE=ON` reads
  those restores through DirectStorage (`--disk-kv-directstorage`; untested, as the D3D12
  option).
- **Adaptive MTP** ([Mirko Covizzi](https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile)).
  `--adaptive-mtp` lets each round verify 3..K of the K drafts, the width that measured draft
  survival and measured round cost favour, with CUDA Graphs for each width. On Ternary Bonsai 2
  on an RTX 3090 with K=5 it verified five drafts in 59% of rounds, four in 23% and three in 18%,
  and did not beat the card's fixed K=3: 200 against 204 tok/s on short prompts, 150 against 163
  at 8K. The graphs for every width cost memory too, so Huihui with a 198,400-token cache no
  longer fits a 24 GB card with it. A near-tied token can come out differently at another width,
  as it does between two fixed windows.
- **A fast INT8 prompt-attention kernel** (Ian Ranson, [Wallawalla47](https://github.com/Wallawalla47/ninfer-custom)).
  Every warp keeps its query rows, scores and output in registers and accumulates P·V in FP16 per
  64-key tile. This line extends it to `rk8v4` and the packed key codings and lets the device
  profile turn it on where it is faster (all three measured cards: 19 to 30% less prompt-attention
  time); `--fast-prefill-kernel` forces it. Quick-corpus perplexity at 64K on Ternary Bonsai 2 moves
  from 5.2074 to 5.2079 (`rk8v4`), and the three needles at 131K are all found. An `nvfp4` KV
  cache on Blackwell has its own fast prompt kernel under the same switch, with QK on block-scaled
  FP4 Tensor Cores straight from the stored codes (a two-term NVFP4 Q) past 2048 visible keys:
  0.35-0.67x the tiled kernel's time per layer on RTX 5090, 3.5-14.4% faster prefill at 16K-64K.
- **Agent-harness tool calls.** `<function name=...>`, `<invoke name=...>`, `<function_calls>` and
  `<param name=...>` are read as tool calls (upstream PR #300 by Pavel Kochubey, via Wallawalla47), next
  to the Qwen form, and go through the same recovery pass.
- **Structured output** through xgrammar, speculative decoding included, opt-in with
  `--structured-output` (upstream PR #294 by Andrey Shvartsman).
- **First-token log probabilities.** With `--first-token-logprobs`, a Chat Completions request may ask
  for `top_logprobs` and gets the first generated token's log probability with its alternatives
  (IMGillusion).
- **Rolling retention.** `--context-cache-policy rolling` lets one long conversation keep rolling
  its cached frontier forward (IMGillusion).
- **Diverged-branch release.** `--release-diverged-checkpoints` lets the cache drop first a private
  checkpoint that its own conversation has moved away from (Ian Ranson, after pkochubey's upstream
  PR #300).
- **NVFP4 expert banks on Blackwell** (upstream PRs #286-#290 by Mykhailo Dementii). The published
  Qwen3.6-35B-A3B NVFP4 checkpoint converts with `--recipe qwen3_6_35b_a3b_nvfp4` and runs on any
  `120a` build: on an RTX 5090 (native build, `-DNINFER_SM120_NATIVE=ON`) the 20.6 GB text
  artifact prefilled 27,663 tok/s at 4K and decoded 397 tok/s. Its prefill quantizes activations
  to four bits for W4A4, which only Blackwell has, so sm_8x builds refuse the banks.
- **N-gram copy drafting** (remesis, Ian Ranson). Beside any drafter, a round may verify up to
  15 tokens copied from earlier prompt, tool-result or output text that the last 12 tokens match
  (`--ngram-draft-tokens`, `--ngram-min-match`); it is on by default with `--spec` and exact, since
  the target verifies every copy. An optional RAM archive keeps finished requests' copy sources for
  later requests of the same `X-NInfer-Draft-Session` (`--ngram-archive-mib`).
- **A hybrid prefix cache** (Ian Ranson, [Wallawalla47](https://github.com/Wallawalla47/ninfer-custom)).
  `--use-alt-prefix-caching` swaps the checkpoint catalog for content-addressed 64-token KV blocks
  shared across requests plus sparse state snapshots, sized from free memory and one
  `--host-cache-mib` Host budget. Around the default catalog the same work adds opt-in
  recency eviction with demotion to Host first (`--recency-eviction`), on-demand growth of an
  answer's Device KV lease (`--kv-lease-growth`), one Host budget for the retention tier
  (`--host-cache-mib`), automatic message-boundary anchors for rewritten transcripts
  (`--auto-long-anchors`), and, on by default, the reuse of what an aborted request prefilled and
  least-recently-used replacement of automatic shared prefixes when the catalog is full.
- **Admission and eviction** (Gideon Zenz, David Oelfke, Ian Ranson). `--thorough-admission-search`
  gives a new request's reuse plan up to 250 ms and every candidate, `--value-aware-demote` ranks
  eviction by what a checkpoint would cost to rebuild, `--concurrent-prefill` admits while others
  prefill, and `--recover-invariant-failures` keeps serving after an internal invariant fails.
- **Drafting and sampling** (Gideon Zenz). `--mtp-attention-window N` lets the MTP draft head
  attend to its first 64 keys and the newest `N` instead of the whole history, so the draft's read
  stops growing with the context while verification still decides every token. Post-thinking
  sampling switches a thinking request to its own preset (temperature 0.2) once the reasoning
  closes, per server (`--post-thinking*`) or per request (a `post_thinking` object).
- **Serving** (Gideon Zenz, Ian Ranson). `GET /stats` with every Engine counter and the waiting
  queue, on the main port or a separate `--stats-port`; a terminal dashboard and a wedge watchdog in
  [`tools/monitor`](tools/monitor/README.md); request-log rotation (`--request-log-max-mib`);
  `--assistant-prefill`, `--unconstrained-response-format`, `--lenient-assistant-history` and
  `--derive-session-keys` for clients that need them; Anthropic streams carry the protocol's `ping`
  event with each heartbeat; grouped `--help` screens, `--log-colours` and a statistics panel
  (`--log-stats-panel`); the build id in every product binary.
- **Vision on CPU and position interpolation** (David Oelfke). `--vision-residency cpu` encodes
  images on CPU threads from host FP32 weights with no device Vision memory, and
  `--rope-scaling-factor` with `--rope-scaling-original-context` interpolates positions past the
  native window.
- **Kernels and conversion** (Ian Ranson, Duncan Betts). Programmatic dependent launches in decode
  graphs (`-DNINFER_PDL=ON` on compatibility builds), split-KV attention for short prefill steps
  over long contexts, a general BF16 GEMM fallback, MTP banks of mixed formats, the fused RMSNorm and
  NVFP4 attention input at every width, and converters for ModelOpt NVFP4/FP8 checkpoints, the
  Quasar NVFP4 checkpoint and a least-squares scale search (now part of `grouped_search`). A native
  Windows build against a prebuilt vcpkg tree.
- **Unified Linear templates** (Neroued). The Q4, Q5, Q6 and Q8 A16 Linear templates with sliced-K
  schedules sit beside this line's routes, and each card takes them only at the widths where two
  sweeps on an RTX 3090, 4090 and 5090 measured them faster: Q5 from about 8 columns up to 96 (RTX
  3090), 128 (4090) or 1,024 (5090), 1.5 to 1.7 times as fast over the shapes; Q6 from 4 to 32
  columns; Q4 from 25 columns; Q8 at widths that differ per card. Q4 decode and verification widths
  keep this line's kernels. `NINFER_LINEAR_ROUTES=legacy|unified` forces one table.
- **FP8, NVFP4 and BF16 templates and the fused projections** (Neroued). Upstream's unified FP8,
  NVFP4 and BF16 Linear templates, and its moves of the fused projections of every format onto
  them (attention and GDN inputs with their conv forms, LinearAdd, SwiGLU, the Q8 pair, the top-k
  heads, the Q8 grouped convolution and context-KV materialization), sit beside this line's routes
  by the same rule: a card takes them at the widths where two rounds of every shape and Op
  benchmark measured them faster. Where this line's FP8 and NVFP4 A16 routes loop a small-T kernel
  over wide inputs, the unified ones run 1.7 to 7 (FP8) and 2.5 to 44 (NVFP4) times as fast at
  verification and prefill widths on an RTX 3090, and similarly on a 4090; on an RTX 5090 the FP8,
  NVFP4 and BF16 routes gain 1.1 to 3 times at most widths under A16, A8 and A4. The Q8
  projections keep this line's routes at most widths. The unified SwiGLU keeps its gate and up
  projections in FP32 through the activation.
- **Two-stage GDN prefill** (Neroued). A prompt chunk of 16 tokens or more can run the gated delta
  rule as one preparation pass (Q/K normalization, the gate factors and each chunk's solve) and one
  FP32-state recurrence that also writes the output, in place of the WY, state-passing and output
  kernels. The GDN op ran 1.4 to 4.3 times as fast at every width from 16 to 8,192 tokens on an
  RTX 3090, 4090 and 5090, so their built-in profiles take it; Engine prefill of the Qwen3.8 27B
  artifact on the 3090 moved by about 1 %, the recurrence being a small part of a prefill step.
  `NINFER_GDN_TWO_STAGE=0|1` forces either.
- **PackGQA** (Gideon Zenz). The INT8 prompt kernel can pack each KV head's query heads into its
  tiles (`NINFER_PROMPT_PACK_GQA=1`, or a profile's `attn_pack_gqa`). A 1024-token chunk at 32K
  and 131K of context ran 2.7 % faster on an RTX 3090 and 0.5 % and 3.9 % slower on a 4090 and
  5090, so no built-in profile turns it on.
- **Engine and serving fixes**: out-of-memory recovery of the worker (David Oelfke's, ported by
  Ian Ranson), `--kv-headroom-mib`, `--cuda-graph-allowance-mib`, `--thinking-budget-message` (Ian
  Ranson); the WebUI's MCP traffic relayed behind `--webui-mcp-proxy`, E8 root codes decoded from
  tables and an SM-count RMSNorm cutoff ([tmark00](https://github.com/tmark00/ninfer));
  MTP graph profiles with topology classes (Mykhailo Dementii, upstream PR #221); openable server URLs and CORS
  preflight echoes (pelebel, natpate); and the upstream pull requests listed in the map, among them
  GGUF as a conversion source (giveen), a Q6 recipe (bingchengcc), sparse-MoE, NVFP4 and
  attention-epilogue tuning (Mykhailo Dementii, Duncan Betts, MOVIBALE), quoted-marker and
  duplicate-parameter tool-call fixes (Fedor Suchkov, adubkov) and Copilot tool shapes (Damian
  Sromek).

The [maintainer map](docs/maintainer/consolidated-line.md) lists each change with the files it
touches and the tests that cover it.

## Docker

`ghcr.io/iamwavecut/ninfer-all:latest` is built from every master commit that passes CI, on CUDA
13.4 and Ubuntu 26.04. It carries two builds and starts the one that matches the GPU: `sm_86` for the
RTX 30 series (compute capability 8.6), which also runs the RTX 40 series (8.9), and `sm_120a` for the
RTX 50 series and the RTX PRO 6000 Blackwell (12.0). The host needs an NVIDIA driver of the CUDA 13
branch (580 or newer) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Other tags: `sha-<commit>`, and the `VERSION` of the newest one; the registry keeps `latest` and the
image published before it, which stays available under its `sha-` tag. The image
is 1.6 GB compressed: Ubuntu, cuBLAS and the CUDA runtime from NVIDIA's repository, FFmpeg, and one
multi-call executable per architecture that all four programs are names of. On an RTX 3090 and an
RTX 5090 with driver 580.159.03 it downloaded the 27B and answered chat completions through
`run qwen38-27b`; the 5090 started at the full 262,144-token context, the 3090 at 196,608 after two
of the launcher's step-downs.

```bash
docker pull ghcr.io/iamwavecut/ninfer-all:latest
# The model goes to ./models (19 GiB for the 27B).
docker run --rm -v "$PWD/models:/models" ghcr.io/iamwavecut/ninfer-all download qwen38-27b
# Serve it with the measured `tuned` profile on http://localhost:8080/v1.
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all run qwen38-27b
```

Or with [compose.yaml](compose.yaml), which wires the GPU, the port and the volumes:

```bash
docker compose run --rm ninfer download qwen38-27b
docker compose up -d
```

The container's command chooses what runs:

| command | runs |
|---|---|
| `run <model> [profile]` (default `run qwen38-27b`) | `scripts/run.sh`: the launcher profiles, with the same `NINFER_*` overrides (`-e NINFER_SPEC=mtp`, `-e NINFER_CONTEXT=131072`, ...) |
| `download <model>` | `scripts/download-model.sh` into `/models` |
| `serve`, `ninfer`, `perplexity`, `calibrate` `[args]` | that binary, for any artifact and flags: `serve /models/my.ninfer --max-context 65536 ...` |

Volumes: `/models` holds artifacts, `/cache` the device profile the engine measures on first start,
and `/grafts/<model>/` optional [prompt grafts](docs/serving.md#prompt-grafts). The server listens on
port 8080. `NINFER_IMAGE_ARCH=sm86|sm120a` overrides the GPU detection. `docker build -t ninfer .`
builds the same image from source (`--build-arg ARCHS=86` for one architecture).

## Running

Download an artifact from the table below and point `ninfer-serve` at it. The server speaks the
OpenAI and Anthropic APIs on `127.0.0.1:8080` by default. The card's device profile is picked up
on its own; the configurations below are the ones the [reference tables](docs/performance/reference-2026-09.md)
measure.

<details>
<summary>Ternary Bonsai 2 27B, fastest single stream: DFlash2 with five drafts over the full 262,144-token window</summary>

```bash
ninfer-serve Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 5
```

Five drafts are the all-round choice: seven are faster on short answers, and three to seven win
after long documents ([draft length](docs/performance/reference-2026-09.md#draft-length)). Add
`--vision --vision-residency overlay --vision-max-merged 12288` for images: the encode borrows the
drafter's memory, so the whole window still fits a 24 GB card.
</details>

<details>
<summary>Ternary Bonsai 2 27B with MTP drafting through the proposal head</summary>

```bash
ninfer-serve Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```
</details>

<details>
<summary>Ternary Bonsai 2 27B with the largest context a 24 GB card holds (958,464 tokens, <code>rk4v4</code>)</summary>

```bash
ninfer-serve Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 958464 --kv-capacity 958464 --kv-dtype rk4v4 --gdn-state-fp16 --rope-yarn
```

An RTX 5090 holds the 1,048,576-token maximum with `rk4v4`, DFlash2 or MTP included, and
978,944 tokens with `rk8v4`. Filled to 1,048,576 tokens, the model still finds codes planted at 33
and 66% but misses the one at 90% (about 943K), with YaRN or plain RoPE; up to about 880K it found
every code on all three cards.
</details>

<details>
<summary>Ternary Bonsai 2 27B with adaptive MTP and a disk tier that keeps evicted conversations</summary>

```bash
ninfer-serve Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 198400 --kv-capacity 198400 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 5 --lm-head-draft --adaptive-mtp \
  --disk-kv-path /var/cache/ninfer --disk-kv-gib 64 --disk-kv-restore
```
</details>

<details>
<summary>Qwen3.8-27B on a 24 GB card: DFlash2 with five drafts over 245,760 tokens of <code>rk4v4</code></summary>

```bash
ninfer-serve Qwen3.8-27B-NInfer/qwen3_8_27b.ninfer --model-id qwen3.8-27b \
  --max-context 245760 --kv-capacity 245760 --kv-dtype rk4v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 5
```

With `rk8v4` the same speculation fits 167,936 tokens on an RTX 4090 and 176,128 on an RTX 3090;
an RTX 5090 takes the full 262,144 with either.
</details>

<details>
<summary>Measuring a card that has no built-in profile</summary>

```bash
ninfer-calibrate --print > my-gpu.json
```

The engine does this by itself at first start; running it by hand refreshes the profile after a
driver or clock change. See [device profiles](docs/device-profiles.md).
</details>

## Artifacts

| model | artifact | notes |
|---|---|---|
| Ternary Bonsai 2 27B | [WaveCut/Ternary-Bonsai-2-27B-NInfer-v3](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3) | 8.87 GiB. Ternary text tower, token table and head, Vision, Bonsai-trained MTP head and DFlash2 adapter, and an exact proposal head. Runs only on this line. |
| Qwen3.8-27B GSQ-RCO IQ3_S | [WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3) | 13.99 GiB. ISTA-DASLab's 3.5-bit GGUF blocks kept byte for byte, their Q6_K MTP head, Vision, the DFlash2 adapter and a proposal head. Runs only on this line. |
| Qwen3.8-Flash-Next n-gram table | [WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3) | 26.82 GiB: the IQ4_NL n-gram table every Flash-Next artifact below reads (`--ngram-table`). Runs only on this line. |
| Qwen3.8-Flash-Next GSQ-RCO Q2_0 | [WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3) | 35.89 GiB: ISTA-DASLab's 2.4-bit GGUF blocks kept byte for byte, without the n-gram table, with the Vision tower. Experts on two 24 GB GPUs, in host memory or on disk. Runs only on this line. |
| Qwen3.8-Flash-Next GSQ-RCO IQ3_S | [WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3) | 51.90 GiB: ISTA-DASLab's 3.5-bit GGUF blocks kept byte for byte, without the n-gram table, with the Vision tower. Experts in host memory or on disk with one 24 GB GPU. Runs only on this line. |
| Qwen3.8-Flash-Next Coder GSQ-RCO IQ1_M | [WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-NInfer-v3) | 28.42 GiB: ISTA-DASLab's expert-pruned coding build (256 experts per layer) in its GGUF blocks, without the n-gram table, with the Vision tower. Runs only on this line. |
| Qwen3.8-27B | [neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 19 GiB, `groupwise-int` (Q4/Q5), the upstream artifact the reference tables use |
| Qwen3.8-27B, abliterated | [WaveCut/Huihui-Qwen3.8-27B-abliterated-NInfer-v3](https://huggingface.co/WaveCut/Huihui-Qwen3.8-27B-abliterated-NInfer-v3) | 19.03 GiB, official `qwen3_8_27b` recipe with MTP, DFlash2 and a proposal head |
| Qwen3.6-35B-A3B NVFP4 | [WaveCut/Qwen3.6-35B-A3B-NVFP4-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.6-35B-A3B-NVFP4-NInfer-v3) | 20.39 GiB. RedHatAI's NVFP4 experts kept code for code, Q8 projections, Vision, MTP and a proposal head. Needs an `sm_120a` GPU. |

The official NInfer artifacts listed in the original READMEs load here too.
Weight conversion shows how the [Bonsai](docs/weight-conversion.md#ternary-bonsai-2-27b) and
[GSQ-RCO](docs/weight-conversion.md#a-mixed-precision-qwen38-27b-gguf) artifacts are built.

## Building

<details>
<summary>Linux with CUDA 13.1</summary>

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target ninfer-serve ninfer-calibrate
```

`CMAKE_CUDA_ARCHITECTURES` is `86` for the RTX 30 series, `89` for the RTX 40 series and `120a`
for the RTX 50 series and the RTX PRO 6000 Blackwell (on the `mma.sync` compatibility path, which the
ternary route needs). The
opt-in build options are listed in the [Linux build guide](docs/rtx-3090-linux.md#build-options).
Windows builds, release packages, tests and benchmarks work as in the
[NInfer-3090 README](https://github.com/ashalliants/ninfer-3090#readme).
</details>

## License

Apache-2.0, as upstream. The Bonsai artifact's weights come from PrismML, ProCreations and Qwen,
all Apache-2.0; its card lists the notices.
