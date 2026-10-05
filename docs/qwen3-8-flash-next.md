# Qwen3.8-Flash-Next

NInfer runs Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`, 125B text parameters, about 6B
active per token) from ISTA-DASLab's GSQ-RCO GGUF releases, converted without requantization:

- [Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF):
  Q2_0 (37.6 GB), IQ2_XS (39.2 GB), IQ3_XXS (47.0 GB), IQ3_S (54.8 GB);
- [Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF).

Each release is two GGUF shards: the model, and the 28.8 GB n-gram table of the model's per-layer
embedding (PLE), identical across releases. NInfer keeps them apart too: the model artifact, and an
n-gram companion artifact that every quantization of the model shares.

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
  --device cpu --rows-per-chunk 4096 \
  --name Qwen3.8-Flash-Next-GSQ-RCO-Q2_0 --out models/flash-next-q2_0.ninfer

python3 -m tools.convert --model /path/to/Qwen3.8-Flash-Next --components ngram \
  --recipe qwen3_8_flash_next_ngram \
  --source ngram=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --device cpu --rows-per-chunk 1048576 \
  --name Qwen3.8-Flash-Next-ngram-IQ4_NL --out models/flash-next-ngram-iq4nl.ninfer
```

Every matrix keeps the block type the release chose (see [GGUF block formats](gguf.md)); the expert
banks keep the exporter's expert-major layout, so one expert is one contiguous range of bytes. The
recipe undoes llama.cpp's exporter conventions as the Qwen3.8-27B GGUF recipe does (grouped GDN
value heads, `1 + w` norms, `A_log`, the head-interleaved query and gate). The companion carries the
hash constants its rows were written for; the runtime derives them again from the model's
configuration and refuses a companion that disagrees.

## Run

The companion is found next to the model artifact, or named with `--ngram-table`:

```bash
# Two 24 GB GPUs: every expert in device memory, one pipeline stage per GPU (Linux).
./build/apps/ninfer-serve models/flash-next-q2_0.ninfer --devices 0,1 --max-context 32768

# One GPU: the experts in pinned host memory, the most used of them cached on the GPU.
./build/apps/ninfer-serve models/flash-next-q2_0.ninfer --expert-residency host --max-context 32768
```

| Option | Meaning |
|---|---|
| `--expert-residency device\|host` | expert banks in the stage devices' memory (default), or in page-locked host memory that the expert kernels read across the bus |
| `--expert-cache-mib N\|auto` | with host experts, device memory for the most used experts: `auto` (default) takes what each device has free after startup less a margin, `0` disables the cache |
| `--ngram-table PATH` | the n-gram companion artifact |
| `--ngram-ram` | load the 28.8 GB table into RAM instead of reading the 16 rows each token needs from its file |
| `--devices A,B,...` | one pipeline stage per GPU; layers are split so that every stage holds about the same stored bytes (`--stage-layers` overrides) |

With host experts the GPU holds only the dense weights (3.7 GB for the Q2_0 release), the
per-request state and the expert cache; the host needs the expert banks in page-locked memory
(34 GB for Q2_0). The n-gram table is read from the file's page cache unless `--ngram-ram` is given.

Requests run one at a time in arrival order; `--max-concurrency` above one queues.

## Execution

- The residual is a four-stream hyper-connection stack kept in FP32 between layers.
- The 36 Gated DeltaNet layers run the Qwen3.5 GDN kernels with a sigmoid output gate; the 12 sparse
  attention layers run the block indexer and attend only to the blocks it selects (plain dense
  attention below 2,051 positions).
- The PLE layer reads its 16 n-gram rows per token from the companion (IQ4_NL rows decoded on the
  GPU).
- The 512-expert MoE groups each layer's (token, expert) pairs by expert on the GPU and runs one
  pass over each selected expert's rows for all of its tokens, through device tables of expert
  base pointers: an expert is read wherever the table points, in device memory, a cache slot or the
  pinned host block. Weighted expert outputs are summed in fixed point, so the result does not
  depend on the order experts finish in.
- The expert cache counts the routes each forward pass took (decayed per token) and, between
  passes, copies the experts it needed most into its slots and points the tables at them.

## Measurements

RTX 3090 (24 GB, PCIe 4.0), Q2_0 release, greedy decoding, prompts of 20 to 29 tokens:

| Placement | Device memory | Decode |
|---|---:|---:|
| Experts in pinned host memory, no cache | 3.7 GB | 16.8 tok/s |
