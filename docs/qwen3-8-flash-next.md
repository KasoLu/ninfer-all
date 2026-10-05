# Qwen3.8-Flash-Next

NInfer runs Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`, 125B text parameters, about 6B
active per token) from ISTA-DASLab's GSQ-RCO GGUF releases, converted without requantization:

- [Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF):
  Q2_0 (37.6 GB), IQ2_XS (39.2 GB), IQ3_XXS (47.0 GB), IQ3_S (54.8 GB);
- [Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF).

Each release is two GGUF shards: the model, and the 28.8 GB n-gram table of the model's per-layer
embedding (PLE), identical across releases. A NInfer artifact is one file that holds both: the
table is its `ngram` component, which nothing reads at load; each token reads the 16 rows it
addresses from the file, or `--ngram-ram` loads the table into RAM.

Two conversions are published, with their measurements:
[Q2_0](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3) and
[IQ3_S](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3).

The text model runs; Vision, MTP drafting, the context cache and structured output are not
available for this family yet (the [plan](maintainer/qwen3-8-flash-next-plan.md) tracks them).

## Convert

`--model` needs only the HF checkpoint's configuration and tokenizer files (`config.json`,
`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`) from
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

```bash
python3 -m tools.convert --model /path/to/Qwen3.8-Flash-Next \
  --recipe qwen3_8_flash_next_gguf \
  --source gguf=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --source ngram=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --device cpu --rows-per-chunk 65536 \
  --name qwen3.8-flash-next --out models/flash-next-q2_0.ninfer
```

Every matrix keeps the block type the release chose (see [GGUF block formats](gguf.md)); the expert
banks keep the exporter's expert-major layout, so one expert is one contiguous range of bytes. The
recipe undoes llama.cpp's exporter conventions as the Qwen3.8-27B GGUF recipe does (grouped GDN
value heads, `1 + w` norms, `A_log`, the head-interleaved query and gate). The n-gram table keeps
the release's IQ4_NL rows, and its component carries the hash constants they were written for; the
runtime derives them again from the model's configuration and refuses a table that disagrees.

## Run

```bash
# Two 24 GB GPUs: every expert in device memory, one pipeline stage per GPU (Linux).
./build/apps/ninfer-serve models/flash-next-q2_0.ninfer --devices 0,1 --max-context 32768

# One GPU: the experts in pinned host memory, the most used of them cached on the GPU.
./build/apps/ninfer-serve models/flash-next-q2_0.ninfer --expert-residency host --max-context 32768

# One GPU and little RAM: the experts stay in the artifact's files and stream into a GPU cache.
./build/apps/ninfer-serve models/flash-next-q2_0.ninfer --expert-residency disk --max-context 32768
```

| Option | Meaning |
|---|---|
| `--expert-residency device\|host\|disk` | expert banks in the stage devices' memory (default); in page-locked host memory that the expert kernels read across the bus; or left in the artifact's files, each layer's routed experts read into a device cache before they run |
| `--expert-cache-mib N\|auto` | with host or disk experts, device memory for the most used experts: `auto` (default) takes what each device has free after startup less a margin; `0` disables the host-mode cache (disk mode needs one) |
| `--ngram-ram` | load the 28.8 GB n-gram table into RAM instead of reading the 16 rows each token needs from the artifact |
| `--devices A,B,...` | one pipeline stage per GPU; layers are split so that every stage holds about the same stored bytes (`--stage-layers` overrides) |

With host experts the GPU holds only the dense weights (3.7 GB for the Q2_0 release), the
per-request state and the expert cache; the host needs the expert banks in page-locked memory
(34 GB for Q2_0). With disk experts the host needs no copy of the banks at all: the expert cache
reads the missing experts from the artifact's files through the OS page cache, eight reads in
flight, into a 256 MB page-locked staging ring, so the page cache keeps whatever the system can
spare and the rest comes from the disk. The n-gram rows are read the same way unless `--ngram-ram`
is given.

The KV cache of the 12 sparse-attention layers is BF16 whatever `--kv-dtype` asks (the engine says
so at startup, and reports BF16).

Requests run one at a time in arrival order; `--max-concurrency` above one queues.

## Execution

- The residual is a four-stream hyper-connection stack kept in FP32 between layers.
- The 36 Gated DeltaNet layers run the Qwen3.5 GDN kernels with a sigmoid output gate; the 12 sparse
  attention layers run the block indexer and attend only to the blocks it selects (plain dense
  attention below 2,051 positions).
- The PLE layer reads its 16 n-gram rows per token from the artifact (IQ4_NL rows decoded on the
  GPU).
- The 512-expert MoE groups each layer's (token, expert) pairs by expert on the GPU and runs one
  pass over each selected expert's rows for all of its tokens, through device tables of expert
  base pointers: an expert is read wherever the table points, in device memory, a cache slot or the
  pinned host block. Up to eight tokens run vector products; wider calls run ggml's integer
  tensor-core matrix kernel over the routed pairs from device memory. That kernel reads a 640-value
  down row in 256-value steps, so it decodes the bytes after a down matrix as the down's blocks:
  in a bank they are the next expert's down or zeros after the last one, and every cache or stream
  slot keeps zeros after its down, which a smaller down from another layer does not uncover (another
  format's bytes there can hold a non-finite scale, and NaN follows). With host experts, a wide call
  first copies the routed experts the cache does not hold into a device pool (one slot per expert on
  each GPU, 0.7 GB for Q2_0), so each expert crosses the bus once per chunk. Weighted expert outputs
  are summed in fixed point, so the result does not depend on the order experts finish in.
- The expert cache counts the routes each forward pass took (decayed per token) and, between
  passes, copies the experts it needed most into its slots and points the tables at them. With
  disk experts the cache works per layer instead: once a layer has routed its tokens, the experts
  it lacks are read into the slots least recently used (never one the same call needs), and only
  then do the layer's experts run.
- A decode step (one token) replays a CUDA graph per pipeline stage, captured at a sequence's
  second decode step; the token's position reaches the sparse-attention kernels in device memory.
  Disk experts need the host between a layer's routing and its experts, so their steps stay eager.
  `--no-cuda-graph` decodes eagerly everywhere.

## Measurements

Q2_0 release, greedy decoding, CUDA 12.8, 2026-10-05. Decode is measured over the 78 tokens of a
short answer (prompt of 20 tokens) and over the first tokens after a 4,463-token prompt; prefill is
that prompt in 512-token chunks.

| Hardware and placement | Device memory | Host memory | Decode, short answer | Decode after 4,463 tokens | Prefill |
|---|---:|---:|---:|---:|---:|
| 2× RTX 3090 Ti (PCIe, no P2P), experts on the GPUs (`--devices 0,1`) | 18.4 + 19.4 GB | 0.7 GB | 90.2 tok/s | 107 tok/s | 1,504 tok/s |
| RTX 3090, experts in pinned host memory, 17.7 GB expert cache | 22.6 GB | 34 GB pinned | 48.9 tok/s | 42.1 tok/s | 842 tok/s |
| RTX 3090, experts on disk, artifact in the page cache | 22.6 GB | 0.9 GB + page cache | 47.0 tok/s | 39.1 tok/s | 630 tok/s |
| RTX 3090, experts on disk, page cache dropped every second (NVMe) | 22.6 GB | 0.9 GB | 17.2 tok/s | 11.3 tok/s | 195 tok/s |

Host memory is the process's peak resident set (the pinned bank for host experts). Decode after
the long prompt covers its first five tokens only. CUDA graphs add 11% to the short-answer decode on
the two GPUs (81.2 tok/s eager) and 7% with host experts (44.3 tok/s); disk experts decode eagerly.
Prefill touches nearly every expert of every layer in each chunk, and with host or disk experts
each of them crosses the bus or comes off the disk once per chunk, so larger `--prefill-chunk`
values serve more tokens per copy: with host experts on an RTX 3090 Ti the long prompt takes 6.05 s
in 512-token chunks and 4.58 s in 2,048 (10.97 s before the experts went through device slots).

The IQ3_S release on one RTX 3090 (310 W power limit, PCIe 4.0 x16) in a host with 62 GB of RAM,
23 cores of an AMD EPYC 7663 and an NVMe drive, 2026-10-05, with the Q2_0 release in the same
sitting; the artifacts are single files with their n-gram tables, and decode counts the 86 tokens of
the short answer:

| Release and placement | Device memory | Host memory | Decode, short answer | Decode after 4,463 tokens | Prefill |
|---|---:|---:|---:|---:|---:|
| IQ3_S, experts in pinned host memory, 16.2 GB expert cache | 22.1 GiB | 50.3 GB pinned | 34.4 tok/s | 30.2 tok/s | 533 tok/s |
| IQ3_S, experts on disk, the page cache holding what of the 83.6 GB file fits | 22.1 GiB | page cache | 19.0 tok/s | 21.6 tok/s | 209 tok/s |
| IQ3_S, experts on disk, the file's pages evicted every second | 22.1 GiB | — | 10.7 tok/s | 5.4 tok/s | 47 tok/s |
| Q2_0, experts in pinned host memory | 22.1 GiB | 34.0 GB pinned | 50.2 tok/s | 43.7 tok/s | 847 tok/s |

On the current code every GSQ-RCO release, converted into one file with its table, answers the
generate test's prompts (the facts and the 4,463-token needle) on that card with disk experts and
with host experts, with CUDA graphs and without: Q2_0, IQ2_XS (39.2 GB, 35.5 GB pinned), IQ3_XXS
(47.0 GB, 42.9 GB pinned), IQ3_S, and the Coder build's IQ1_M (29.6 GB, 256 experts, 25.1 GB pinned).
Q2_0 also ran with its experts on two GPUs before the table moved into the artifact. With disk
experts the process peaked at 1.05 to 1.19 GB of RAM for IQ2_XS, IQ3_XXS and IQ1_M.

llama.cpp runs the same GGUFs with the experts on the CPU (`--n-cpu-moe 48`). On the second
machine above (23 threads) llama-bench gives 29.6 tok/s decode (tg128) and 312 tok/s prefill
(pp512) for IQ3_S, and 12.7 and 368 tok/s for Q2_0; the first RTX 3090's machine (32 threads) gave
11.2 and 267 tok/s for Q2_0. Its speed follows the host CPU, and its Q2_0 CPU path is the slower one.

Perplexity agrees with llama.cpp: over the first 72 KB of the WikiText sample in
`eval/corpora/perplexity-1m`, 2,560-token windows advancing by 1,024 targets (llama.cpp's
`--ppl-stride 1024 -c 2048`, which widens the window to 2,560), the fourteen windows both evaluate
identically (14,336 targets) give, with a BF16 KV cache in both:

| Release | NInfer | llama.cpp | Window by window |
|---|---:|---:|---|
| Q2_0 | 2.6579 | 2.6502 | -0.014 to +0.016 nats |
| IQ3_S | 2.1317 | 2.1278 | -0.022 to +0.016 nats |
