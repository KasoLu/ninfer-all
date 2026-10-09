# DOING.md — rk6v4-e8 移植实施记录

本文件是 `RK6V4E8.md`（移植计划文档）实施过程的逐步事实记录，完成后按计划移除。
每一步记录：改动文件、改动内容、依据（RK6V4E8.md 的章节）与验证状态。

## 环境

- 构建：docker 镜像 `ninfer-local-build:latest`（CUDA 13.1.2 工具链，CMake 3.28.3，Ninja，16 核）。
- **本机无 NVIDIA 显卡，容器内只能 CPU 编译，CUDA 测试可编译不能执行**（无设备时按
  `cuda_unavailable()` 惯例 exit 77 SKIP）。RK6V4E8.md §7 的 4090 验收（PPL / 检索门 /
  200K 容量）留待有 GPU 的机器执行。
- 编译目标：`CMAKE_CUDA_ARCHITECTURES=89`（用户 m00314 指示本地构建只针对 sm_89/4090 提速；
  m00317 指示手动停掉构建容器后由我重新配置并执行）。产品多架构目标不变，仅本地构建树单架构。

## 1. 核心 codec（RK6V4E8.md §4.1）

- [x] `include/ninfer/types.h`：`KvCacheStorage` 末尾追加 `RotatedInt6KeyInt4ValueE8`（保既有 ordinal）。
- [x] `src/ops/kv_cache/int8_g64_codec.cuh`：
  - `KvKeyCoding` 追加 `K6E8`；
  - i6 原语照抄 4090 仓库修复后版本：`kKVCacheI6HeadExtent=192`、
    `kv_cache_i6_code_from_int`、`kv_cache_i6_quant_code`、`kv_cache_unpack_i6`、
    `kv_cache_pack_i6_quad`、`kv_cache_lanes_write_i6_quad`（uniform FullMask 显式源 gather，
    §5.3 死锁修复）、`kv_cache_unpack_i6x16`（q2 行 `((b2 & 0xffu) << 16)` 修复版，§6.2）；
  - `KvPackedKeyChunk<K6E8>` = 12 B（3×u32 struct）；
  - `kv_cache_packed_key_chunk_index<K6E8>`：`paged_kv_page_head_offset<192> + 192*page_off + (d*3)/4`；
  - `kv_cache_packed_key_expand16<K6E8>` 调 `kv_cache_unpack_i6x16`；
  - `kv_cache_i8_family_store_key_group` 加 `K6E8` 支（§5.2：scale absmax/31、E8 投影复用
    `kv_cache_e8_nearest`、clamp ±32/31、`kv_cache_lanes_write_i6_quad` 写回；V 走既有 PackedValues）。
- [x] `src/core/paged_kv_storage.h`：`paged_kv_storage_layout` 加 case：
  K `{U8, head_dim*3/4, FP16, 4}`，V `{U8, head_dim/2, FP16, 8}`。
- [x] `src/ops/kv_cache/d256_profile.h`：`kv_cache_is_int8_family` 加入新 storage；
  `d256_kv_cache_profile` 加 case `{U8, U8, 192, 128, 64, FP16, 4, FP16, 8}`（真值 344 B，见文末修正）。

## 2. 分发（RK6V4E8.md §4.2，3 行）

- [x] `src/ops/kv_cache/append/launch.cu`：storage → Keys 分发链加 `RK6V4E8 → issue<true, KvKeyCoding::K6E8>()`。
- [x] `src/ops/softmax_attention/dense/causal_cache/small_t_i8_launch.cuh`：同上（decode 侧分发）。
- [x] `src/ops/softmax_attention/dense/causal_cache/prompt.cu`：标准 i8 prompt 分发加两支
  （pv_f16 true/false，照 Int4E8 口径）；fast 路径保持 rk4v4-e8 专属不动。
- [x] 连带分发点：`small_t.cu` 路线键 `"rk6v4-e8"`；`causal_softmax_attention.cpp` INT8 家族
  prompt 截止分支加 case；`sparse_attention.cu` `decode_key` 加 K6E8 支（`plane_offset<192>` +
  `kv_cache_i6_unpack_i8x16`）、`decode_value` packed int4 列表加新 storage、`NINFER_SPARSE_ATTENTION_CASE`
  实例化加支；`device_calibration.cu` `--attention` 校准列表加 `rk6v4-e8`。

## 3. 接线（RK6V4E8.md §4.3，~15 处单行）

- [x] `apps/cli/options.cpp`（parse + help）
- [x] `apps/cli/main.cpp`（`format_kv_cache`：`rotated-k6e8g64-v4g32`）
- [x] `apps/perplexity/main.cpp`（parse + 名表 + usage）
- [x] `src/serve/serve_options.cpp`（parse + help）
- [x] `src/serve/request_log.cpp`（`kv_cache_name`）
- [x] `src/serve/operational_log.cpp`（`kv_cache_name`）
- [x] `bench/inference/ninfer_bench_support.cpp`（parse + 名表 + usage）
- [x] `bench/ops/causal_softmax_attention_bench.cu`、`bench/ops/kv_cache_append_bench.cu`
  （KvChoice + 名表 + usage + selected_storages + all 列表）
- [x] `src/calibration/route_catalog.cpp`：coding 列表加 `"rk6v4-e8"`
- [x] `include/ninfer/ops/kv_cache_append.h`：存储合同注释补 rk6v4-e8 段

## 4. 测试（RK6V4E8.md §4.4）

- [x] `tests/ops/test_kv_plane_types.cpp`：RK6V4E8 布局断言（K U8 LE=192 / V U8 LE=128 /
  scale FP16 LE=4&8；**344** B/token/head；页 stride 256 对齐）——**docker 内已跑，PASS**。
- [x] 新 `tests/ops/test_i6_roundtrip.cu`（注册进 tests.cmake）：生产
  `kv_cache_pack_i6_quad` → `kv_cache_unpack_i6x16` 往返硬门禁：16×64 全码值全位置扫描 + 4096
  随机块，任何一位不一致 exit 1——已编译；执行待 GPU。
- [x] `tests/ops/test_kv_cache_append.cpp`：`full_append_case_i6`（host oracle：既有 E8 host
  replica + i6 打包，bit-exact 对照）——已编译；执行待 GPU。
- [x] `tests/ops/softmax_attention/causal_cache.cpp`：K6E8 全套 storage 接线 +
  `encode_rotated_i6_key_row`（logical-row oracle）+ `run_rk6v4_e8_cases`（路线覆盖照
  rk4v4-e8）+ criterion（照 rk2v4-e8 = rk4v4-e8 口径）+ `main.cpp` `--rk6v4-e8-only` +
  `tests.cmake` add_test——已编译；执行待 GPU。
- [x] `tests/test_serve_options.cpp`：`--kv-dtype rk6v4-e8 → 枚举` 解析 + help 串——**docker 内已跑，PASS**。
- [x] `tests/models/qwen3_5/test_runtime_mechanisms.cpp`：`test_decoder_layout` 加 rk6v4-e8
  引擎布局用例（192/128/4/8 平面 + 容量排序 rk4v4-e8 < rk6v4-e8 < rk8v4）——**docker 内已跑，PASS**。

## 5. 文档（RK6V4E8.md §4.5）
- [x] `docs/cli.md`（`--kv-dtype` 表行、Context and memory 十档列表与容量说明：344 B，介于
  rk4v4-e8 280 与 rk8v4 408 之间）、`docs/serving.md`（`--kv-dtype` 表行）、
  `docs/device-profiles.md`（`attn_i8_small` 覆盖列表）。
  `README.md` 未改：其容量表全是 4090 实测数据（PPL / 检索门 / 容量），留待 GPU 验收后更新，
  容量档文字说明在 docs/cli.md 的 Context and memory 节（serving.md 表行链接到它）。
- [ ] 完成后移除 `RK6V4E8.md`（AGENTS.md：临时计划完成即删）——待 GPU 验收（§7）通过后移除。

## 6. 验证

- [x] docker 内配置 + 全量编译：`cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89`，`cmake --build build -j3`
  全量通过（-j6/-j16 被 OOM SIGKILL，exit 137；容器 15 GB RAM）。
- [x] ctest：容器无 libcuda.so.1 → 用 stub 库（从 cuda.h 提取 508 个 `cu*` 符号、全返回 103、
  `-Wl,-soname,libcuda.so.1` + LD_PRELOAD）满足动态加载。303/307 CPU 可执行测试 Passed 或 77 Skipped。
  4 个失败均为环境因素非回归：`ninfer_chat_templates_test`（容器缺 jinja2）、
  `ninfer_qwen4_exp_ngram_writer_interop_test`（缺 pytest）、`ninfer_qwen4_exp_peer_copy_order_test` /
  `ninfer_qwen4_exp_expert_cache_visibility_test`（自带 LD_PRELOAD 覆盖 stub 且需 GPU）。
- [ ] 待 GPU：i6 roundtrip 硬门禁、kv_cache_append bit-exact、causal_cache rk6v4-e8 全套、
  4090 上 PPL（quick→full，判据 +0.1%~+0.2%）、检索门（A 门单针 ≥200K + C 门 code-detail）、
  200K 容量验证（RK6V4E8.md §7）。
## 2026-07-11 修正：RK6V4E8.md 的容量算术错误（336 → 344 B）

首次全量编译失败暴露两处错误：

1. `tests/ops/test_i6_roundtrip.cu` include 写的是 `ops/int8_g64_codec.cuh`——4090 仓库有
   `src/ops/int8_g64_codec.cuh`，本仓库只有 `src/ops/kv_cache/int8_g64_codec.cuh`（RK6V4E8.md
   §5 的 include 按 4090 路径写的，本仓库树不同）。已改 include。
2. RK6V4E8.md:17 声称 **336 B**（"K 192 + V 128 + 2×8 B FP16"），但该文档自己的 profile 表
   （:88 `{U8, U8, 192, 128, 64, FP16, 4, FP16, 8}`）给出 192 + 128 + 4×2 + 8×2 = **344 B**：
   V 是 G32 → 8 个 FP16 scale = 16 B，不是 8 B。计划文档自相矛盾，真值是 344。已把
   `test_kv_plane_types.cpp` 的 static_assert 与运行时断言改成 344，`test_runtime_mechanisms.cpp`
   的容量排序断言改为 rk4v4-e8(280) < rk6v4-e8(344) < rk8v4(408)。RK6V4E8.md 按 AGENTS.md 在完成
   后移除，不单独修订。

## 长上下文 PPL 矩阵脚本（2026-03-03，提交 95aef1d1）

- 交付 `tools/long_context_ppl.sh`：在单个超长 UTF-8 文件上跑深度矩阵——bf16 @ 64K
  （24 GB 卡上 27B 权重旁 bf16 KV 放得下的最深度）、rk8v4 与 rk6v4-e8 @ 128K / 200K。
- 每档报告落在 `profiles/perplexity/long_ctx/<kv-dtype>.<context>/report.json`；bf16 参照行
  停在 64K，与量化档深度不对称（与 needle 检索门同款口径），脚本结尾强制打印该注记。
- 选项：`--dry-run`（只打印命令）、`--skip-bf16`、`--out DIR`、`--contextNNN/--strideNNN`
  逐深度覆盖；`NINFER_PERPLEXITY` 环境变量可覆盖二进制路径（自动探测 build-ninja → build，
  Windows 下补 .exe 后缀）。
- 本地 dry-run 已验证（无 GPU）：全矩阵 5 条命令、`--skip-bf16` 4 条、缺参/未知选项各走
  usage/报错路径；`bash -n` 语法通过。报告摘要行从 report.json 的 overall.perplexity /
  overall.scored_tokens 提取（python3 或 python 均可，缺 python 时跳过摘要不失败）。
- 4090 机器上的执行顺序仍是：先 `ctest -R 'i6_roundtrip|kv_cache_append'` 门禁，再本脚本。

## ninfer_kv_cache_append_test 失败定位与修复（2026-03-03，4090 首跑暴露）

- 4090 首跑：`i6_roundtrip` 门绿，plain `ninfer_kv_cache_append_test` 红，failures=4——
  四个 rk6v4-e8 用例各 1 处 `k codes` 不符，首个 mismatch 恒在 `row*192 + 3`
  （Hkv=2 index 24579，Hkv=4 index 49155）；k scales / v 平面 / guards 全过。
- 定位：每个 48B 组行的 6 字节块 byte 3-5（第 2 个 24 位 quad，维度 4-7）不符，
  byte 0-2 正确 → 宿主 oracle `encode_full_row_i6_keys` 的 quad 循环写死
  `point[j]`（j=0..3），两个 quad 都取了维度 0-3 的 E8 坐标，维度 4-7 的码被维度
  0-3 的码顶替。内核侧布局/打包/蝶形归并/E8 平手规则经逐位审计正确，`i6_roundtrip`
  自洽门（不经过该 oracle）也印证。rk4v4-e8/rk2v4-e8 的 oracle 无 24 位 quad 结构，
  故未受牵连。
- 修复：`tests/ops/test_kv_cache_append.cpp` oracle 改为 `point[4*quad + j]`。内核代码
  零改动。待 4090 重跑 plain `ninfer_kv_cache_append_test` 确认转绿。

## Needle 检索门脚本与解析修复（2026-03-03）

- 交付 `tools/bench/make_needle_probes.py`（自 4090 仓库移植，探针名去掉假尺寸后缀
  `needle_single/five/code`，新增 `--single-tokens` 使单针 haystack 深度可调——
  4090 原版固定 260K，超过本 fork 200000 ctx 的容量目标）与
  `tools/run_needle_gate.sh`（驱动：ctest 往返门禁 → 生成探针 → 每 kv-dtype × 每门
  一次 `ninfer --messages --greedy` 运行 → 对 expected_answers.json 做子串/顺序判分 →
  verdicts.tsv + A/C 门结论；默认 ctx 200000、formats "rk6v4-e8 rk8v4"、gates
  "single code"，`--dry-run`/`--skip-gate`/`--formats`/`--gates` 可调）。
- 判分路径本地验证：生成器 `--scale 0.01` 冒烟 + 假 log 判分（PASS 路径 exit 0、
  MISSING 路径 exit 1）均通过；bf16 深 ctx 的显存告警按 63,928 B/token（27B 模型
  全头/全层 bf16 KV）计算。
- 连带修复：needle 驱动的选项解析 `shift 2`（原实现单次 shift 会把选项值再当位置
  参数——本次 --formats 双值实测暴露）。两个 PPL 脚本的循环本身带无条件 shift，
  case 内 shift 已含两次位移，属原有正确结构；期间误改为 shift 2 造成吞参，已还原。
- `tools/README.md` 任务索引补两行（PPL 深度矩阵、needle 门）；`tools/bench/README.md`
  新增探针烘焙器小节。
- 4090 执行口径：`git pull` 后 `bash tools/run_needle_gate.sh <27B.ninfer>`（默认
  即 A+C 门 @200000 ctx）；判据见 RK6V4E8.md §7.3（A 门单针 ctx ≥ 200K 档、C 门
  code-detail 判分全过）。

## Needle 门首跑全 FAIL 的诊断与修复（2026-03-03，4090 首跑暴露）

- 4090 首跑四档全 FAIL（rk6v4-e8/rk8v4 × single/code），但日志显示引擎运行完全
  正常（prefill 94,703 / 145,918 token，KV capacity 200000，`rotated-k6e8g64-v4g32`
  与 rk8v4 两档都完成解码）。两个独立原因：
  1. **思考模式吃掉答案预算**：artifact `qwen3.8-27b-coder390-Q3LynnStyle-Mtp` 的
     chat 模板默认开思考，32 个 --max-new token 全部耗在 "We need answer user's
     request..." 推理前缀上，答案（passphrase / canary 数字）根本没开始输出。
     修复：ninfer 运行加 `--no-thinking`（4090 原版门的既定口径："disable thinking
     so the observable answer is the retrieved needle"）。
  2. **token 估算偏差**：生成器按 2.5 chars/token 估出 180K，真实 tokenizer 只算
     94,703 token（SVO 填充文本实测 cpl≈4.75）——"200K 档"实际是 95K 深度的 prompt。
     code 探针估算 151K/实际 146K，基本到位。修复：驱动给生成器传 `--cpl-text 4.75`，
     单针 haystack 实际深度回到 ~162K token（含 10% safety），逼近 200K 容量档。
- KV 质量未被此次首跑证伪也未被证实：四份日志的思考前缀都在正确复述题目
  （说明长 prompt 被完整读取），但答案判定要等修复后重跑。
