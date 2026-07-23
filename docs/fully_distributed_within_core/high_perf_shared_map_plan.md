# High-Performance Shared Map Implementation Plan

本文给出 fully_distributed_within_core shared map 的最终性能导向实现计划。
目标不是在 private map 上做小改，而是通过编译期宏隔离出两条 submit
pipeline，使 shared 模式的 loser 热路径只剩 presubmit claim 和返回符号输出。

## 目标与边界

- private map 和 shared map 是两条完全不同的 submit 路径，通过宏隔离。
- 不使用 `SubmitBuilder`、lambda、thunk，也不做运行时 map-mode 分支。
- submit 前先判断是否 winner，再决定是否构建参数。
- shared loser 不构建 `L0TaskArgs`，不做 `view`，不构建 `TensorCreateInfo`，
  不碰 heap / map / publish cursor。
- shared fresh-output 主路径走 O(1) symbol resolve，不被全局
  `published_up_to >= task_id - 1` 阻塞。
- region overlap 作为独立慢路径处理，必要时可先用 conservative frontier
  兜底，但不能拖累 symbol fast path。

建议宏名：

```cpp
#if PTO_FDWIC_SHARED_MAP
// shared submit path
#else
// private submit path
#endif
```

该宏是编译期行为选择。不要引入环境变量或运行时 flag 选择 shared/private。

## 分阶段实现与用例规则

- 每个 phase 必须先完成代码闭环，再补充该 phase 对应的 shared 路径用例。
- phase 不能只用 shared clean build 或 private/default 回归作为完成标准。
- 每个 phase 的 shared 用例必须验证该阶段新增的 shared 功能，而不是通过
  private ABI、用例侧 `resolve`、或临时兼容 wrapper 绕过。
- 每个 phase 合入前必须同时满足：
  - shared 该阶段用例 sim 功能验证通过。普通功能验证不能带
    `--use-example-exec-time`，因为它会跳过 golden 比对；只有 PA sim 性能判断
    才允许单独带该参数。
  - shared 该阶段用例上板通过，上板固定 `task-submit --device 6`。
  - default/private smoke 全量和 PA Case1 保持通过。
- 如果某个 phase 还没有对应 shared 用例，不能声明该 phase 完成，只能声明
  代码骨架或部分 ABI 已落地。

## 当前实现中的关键问题

当前 [TaskOutputTensors](../../src/a5/runtime/fully_distributed_within_core/runtime/pto_types.h)
只保存真实 `Tensor *`，`get_ref()` 返回已物化 descriptor。这会强迫所有核在
submit 返回前物化 output，直接破坏 shared loser do-nothing 的目标。

当前 [dist_submit_impl](../../src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/submit_runtime.h)
在 claim 前已经执行参数物化与 map 准备。最高性能路径必须把 claim 提前到参数构建之前。

当前 [DistCore](../../src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/common/state.h)
内嵌 private `DistTensorMap` 和每核 `heap_next`。这些是 private 路径的正确状态，
但 shared 路径不应继承这两个负担。

## 最终 Submit 形态

codegen 生成普通 C++ 分支，不生成 lambda：

```cpp
SubmitToken tok = rt_presubmit_aic_task(FUNC_QK_MATMUL);

#if PTO_FDWIC_SHARED_MAP
SharedTaskOutputs qk_outs;
if (tok.won) {
    L0TaskArgs args;
    args.add_input(qi);
    args.add_input(key_cache);
    args.add_input(block_table);
    args.add_output(sij_buf_ci);
    args.add_scalar(n_blocks, base_block);
    qk_outs = rt_submit_winner(tok, args);
} else {
    qk_outs = rt_submit_loser(tok, 1);
}
#else
TaskOutputTensors qk_outs;
if (tok.won) {
    L0TaskArgs args;
    args.add_input(qi);
    args.add_input(key_cache);
    args.add_input(block_table);
    args.add_output(sij_buf_ci);
    args.add_scalar(n_blocks, base_block);
    qk_outs = rt_submit_winner(tok, args);
} else if (tok.replay_outputs) {
    L0TaskArgs outputs;
    outputs.add_output(sij_buf_ci);
    qk_outs = rt_submit_loser(tok, outputs);
}
#endif
```

控制流读取必须保留在 `rt_presubmit_*` 前的公共路径。例如读取
`context_lens` / `block_table` 决定循环边界的代码必须所有核执行。

## 公共基础设施

### SubmitToken

新增公共 token：

```cpp
struct SubmitToken {
    int32_t task_id;
    int32_t kernel_id;
    uint8_t active_mask;
    uint8_t anchor_lane;
    bool candidate;
    bool won;
    bool replay_outputs;
    bool joint;
    int32_t joint_block;
    int32_t joint_count;
};
```

`rt_presubmit_task(mixed)` 职责：

1. `local_index++` 得到确定性 `task_id`。
2. execute-first drain：drain follower launch / block-won，再 drain phase B。
3. 根据 `MixedKernels` 和当前 lane 判断是否 candidate。
4. candidate 走 cursor claim，填 `won`。
5. 不构建参数，不分配 heap，不写 TensorMap，不查 fanin。

winner submit 必须消费 token，不能再次 claim。

### 文件拆分

建议结构：

```text
runtime/dist_engine/aicore/submit_token.h
runtime/dist_engine/aicore/submit_private.h
runtime/dist_engine/aicore/submit_shared.h
runtime/dist_engine/aicore/shared_symbol_map.h
runtime/dist_engine/aicore/shared_region_map.h
runtime/dist_engine/common/shared_map_state.h
```

`submit_runtime.h` 最终只做宏选择与薄 wrapper。

### Public API Signatures

orchestration 侧新增 API 直接放在 `pto_orchestration_api.h`，实现落到
`dist_engine_api.h` 暴露的 device-callable symbols。签名固定如下，避免后续阶段
反复改调用点：

```cpp
PTO_DEVICE_FUNC SubmitToken rt_presubmit_task(const MixedKernels &mixed);
PTO_DEVICE_FUNC SubmitToken rt_presubmit_aic_task(int32_t kernel_id);
PTO_DEVICE_FUNC SubmitToken rt_presubmit_aiv_task(int32_t kernel_id);

#if PTO_FDWIC_SHARED_MAP
PTO_DEVICE_FUNC SharedTaskOutputs rt_submit_winner(const SubmitToken &tok, const L0TaskArgs &args);
PTO_DEVICE_FUNC SharedTaskOutputs rt_submit_loser(const SubmitToken &tok, uint32_t output_count);
#else
PTO_DEVICE_FUNC TaskOutputTensors rt_submit_winner(const SubmitToken &tok, const L0TaskArgs &args);
PTO_DEVICE_FUNC TaskOutputTensors rt_submit_loser(const SubmitToken &tok, const L0TaskArgs &outputs);
#endif
```

`rt_submit_winner()` 必须断言 `tok.won == true`。shared build 下的
`rt_submit_loser()` 必须断言 `tok.won == false`，且只使用 `tok.task_id` 和
`output_count`。private build 下的 `rt_submit_loser()` 接收的 `outputs` 只允许包含 `OUTPUT` /
`INOUT` / `OUTPUT_EXISTING` produce 侧需要的信息，不允许包含 input /
scalar / explicit dep。

旧的 `rt_submit_task(mixed, args)` / `rt_submit_aic_task(kernel, args)` 保留为
兼容入口：

- private build：继续走 private 兼容路径，保持语义。
- shared build：不提供性能路径兼容入口。若短期必须保留旧 API 以降低迁移风险，
  只能在 `PTO_FDWIC_SHARED_MAP` 下编译期报错或显式落到 private-compatible 测试
  harness，正式 shared codegen 和 benchmark 不得使用它。

### SubmitToken Construction Details

`rt_presubmit_task()` 内部可复用现有 `dist_submit_begin()` 与
`dist_submit_claim_kernel()` 的逻辑，但必须拆成无参数版本：

```text
dist_presubmit_begin(tok):
    self = g_self
    tok.task_id = self->local_index++
    tok.candidate = false
    tok.won = false
    tok.replay_outputs = true
    tok.kernel_id = INVALID_KERNEL_ID

dist_presubmit_drain(self):
    drain_block_won(self)
    drain_phase_b(self)

dist_presubmit_claim(mixed, tok):
    classify active_mask / anchor_lane / joint fields
    if this lane is not eligible:
        tok.candidate = false
        tok.replay_outputs = true   # private needs producer replay
        return
    tok.candidate = true
    tok.won = claim(cursor[task_id % kCursorShards], task_id)
    tok.kernel_id = selected lane kernel
```

shared build 下 `tok.replay_outputs` 对 loser 无意义，始终不做 output replay。
private build 下 `tok.replay_outputs` 默认为 true，因为每核 private map 必须看到
完整 producer 流。后续若能证明某类 submit 不产生 producer，可由 codegen 传
`output_count=0` 或跳过 loser submit。

## Shared Map 设计

### 输出句柄 ABI

private 和 shared 的输出返回类型必须拆开，不能为了少改调用点强行复用
`TaskOutputTensors`。

private 路径继续使用 `TaskOutputTensors`，语义保持不变：

- `TaskOutputTensors` 只保存已经物化的真实 `Tensor` descriptor 引用。
- `get_ref()` 返回本核 payload 里的 `__gm__ const Tensor&`。
- private loser 仍按 private map 语义回放 output produce 侧。

shared 路径使用独立的 symbolic 返回类型，例如：

```cpp
struct FdwicOutputRef {
    int32_t producer_task_id;
    int16_t output_slot;
};

class SharedTaskOutputs {
public:
    uint32_t size() const;
    FdwicOutputRef output_ref(uint32_t index) const;
};
```

shared 模式只返回 `{task_id, slot}` 这样的 symbolic handle，不返回 `Tensor`，
不暴露 `get_ref()`，也不把 symbolic handle 塞进 `TaskOutputTensors`。

shared 下游 winner 构建参数时，如果输入来自 symbolic output，直接把
`FdwicOutputRef` 作为 shared 参数写进 `L0TaskArgs`：

```cpp
args.add_input(qk_outs.output_ref(0));
```

`L0TaskArgs` / `TensorRef` 在 shared 宏下必须支持 symbolic input slot。winner
submit 在 runtime 内部 materialize 参数时解析 symbolic input 到真实 descriptor
并写入 payload；shared loser 不构建 args，因此不会解析、不会拷贝 descriptor。

`TaskOutputTensors` 的 ABI 不参与 shared 输出路径：

```cpp
class TaskOutputTensors {
public:
    uint32_t size() const;
    void materialize_output(__gm__ const Tensor &tensor);
    __gm__ const Tensor &get_ref(uint32_t index) const &;
};
```

shared 的返回与输入 ABI：

```cpp
class SharedTaskOutputs {
public:
    void add_output_ref(int32_t producer_task_id, int16_t output_slot);
    FdwicOutputRef output_ref(uint32_t index) const;
};

class TensorRef {
#if PTO_FDWIC_SHARED_MAP
    FdwicOutputRef output_ref_;
    bool tensor_from_shared_output() const;
    FdwicOutputRef shared_output_ref() const;
#endif
};
```

winner materialize 时遇到 `tensor_from_shared_output()`，从 shared symbol table /
published descriptor cell 解析并复制到当前 task payload 的 tensor slot；fanin
直接使用 resolved descriptor 的 `owner_task_id`。这个 descriptor copy 是
winner build 参数 payload 的必要持久化，不是 orchestration 调用侧额外 copy。

### 已纠正的错误设计，禁止回退

以下是开发过程中已经确认的错误方向，后续实现不得重复：

1. **不要让 shared 返回 `TaskOutputTensors`。** `TaskOutputTensors` 是 private
   路径的 materialized descriptor 容器，带有 `get_ref()` 和 payload 引用
   生命周期语义。shared loser 不物化 descriptor，强行复用该类型会把 shared
   又拉回 private ABI。
2. **不要在用例 / codegen 侧用 `rt_resolve_output()` 返回 `Tensor` 来补洞。**
   这会在调用侧引入额外 descriptor copy，也会诱导 private 路径从零拷贝
   `get_ref()` 退化成 by-value copy。shared 的 symbolic input 应作为
   `FdwicOutputRef` 进入 `L0TaskArgs`，只在 shared winner runtime 内部解析。
3. **不要为了让 shared 编译通过而改 smoke/PA orchestration 里的 private
   写法。** 用例不是 shared ABI 的适配层；shared 需要自己的返回类型和参数
   slot。private 用例中的 `outs.get_ref(0)` 必须继续表达 private 零拷贝借用。
4. **不要把“宏隔离”理解成同名函数里塞两套半兼容 ABI。** 宏隔离的目标是
   两条 submit pipeline、两套返回/参数 ABI、两个独立热路径。公共的只能是
   `SubmitToken` 和真正共享的底层 primitive。
5. **不要把 descriptor cell resolve 当成最高性能 shared map。** per-task
   descriptor cell 可以作为阶段性 fresh-output 发布机制，但最终 shared map
   应该让 symbolic handle 在 winner materialize 阶段 O(1) 解析，不在
   orchestration 侧显式生成 `Tensor` 临时值。

### Symbol Index

fresh output 主路径只走 symbol index：

```cpp
symbol_desc[task_id & mask][slot] = TensorDescriptor;
store_release(symbol_published[task_id & mask][slot], task_id);
```

resolve 时只等待具体 producer slot：

```cpp
wait_until(load_acquire(symbol_published[p & mask][slot]) == p);
Tensor t = symbol_desc[p & mask][slot].tensor;
```

这比 `wait_until(global_P >= N - 1)` 更并行：task `N` 只等自己实际消费的
symbol producer，不等无关历史 task。

第一版状态放入 host/AICPU 分配的 `DistGlobal` shared state，不放 device 侧
全局对象：

```cpp
constexpr int kSharedSymbolWindow = 4096;
constexpr int kMaxSymbolOutputs = MAX_TENSOR_ARGS;
constexpr int kSharedTensorDescBytes = 128;  // pick by static_assert after Tensor layout audit

struct SharedTensorDesc {
    Tensor tensor;
    int64_t producer_task_id;
    int64_t heap_epoch;
    uint8_t reserved[kSharedTensorDescBytes - sizeof(Tensor) - 16];
};

struct SharedPublishedCell {
    int64_t value;
    uint8_t pad[56];
};

struct SharedSymbolCell {
    SharedTensorDesc desc[kMaxSymbolOutputs];
    SharedPublishedCell published[kMaxSymbolOutputs];
};

struct SharedSymbolIndex {
    SharedSymbolCell cells[kSharedSymbolWindow];
};

static_assert(sizeof(SharedPublishedCell) == 64);
static_assert(sizeof(SharedTensorDesc) == kSharedTensorDescBytes);
static_assert((sizeof(SharedTensorDesc) % 64) == 0);
```

布局原则：

- `published.value` 存 producer task id，而不是 bool，避免 ring/window 复用
  后 ABA。
- 跨 core 独立更新或需要 flush 的同步字段独占 64B cacheline。
- 上板路径优先显式 padding + `sizeof/offsetof` static_assert，不依赖
  `alignas(64)`。
- `kSharedSymbolWindow` 必须大于最大 outstanding task window；如果后续加入
  recycle，需要用 `published.value < task_id - window` 或 completion frontier
  做复用保护。

winner publish 顺序：

```text
write desc tensor fields
store_barrier()
ccec_flush_region(desc, sizeof(desc))
atomic_store_release(published.value, task_id)
ccec_flush_region(published, sizeof(published))
```

resolve 顺序：

```text
cell = cells[producer_task_id & (kSharedSymbolWindow - 1)]
loop:
    ccec_invalidate_region(&cell.published[slot], sizeof(SharedPublishedCell))
    seen = atomic_load_acquire(cell.published[slot].value)
    break if seen == producer_task_id
ccec_invalidate_region(&cell.desc[slot], sizeof(SharedTensorDesc))
return cell.desc[slot].tensor
```

sim 侧用同一 API，但 cache maintenance 是 no-op 或 host fence。CCEC 侧不要用
通用 `__atomic` GM primitive；走已验证的 `common/atomic.h` / 64-bit 原子封装。

### Region Index

region index 只服务：

- `INOUT`
- `OUTPUT_EXISTING`
- external tensor 的 write-after-read / overlap 场景
- view / alias overlap 依赖

第一版 region path 可以使用 conservative frontier 兜底，但必须局限在 region
lookup 内：

```text
symbol input      -> wait exact symbol slot
external read-only -> no wait
no_dep            -> no map wait
explicit dep      -> no map wait, record dep id
region overlap    -> wait region frontier, then region lookup
```

后续如果 region path 成为瓶颈，再把 conservative frontier 替换为 per-buffer /
per-shard publish frontier。这个优化不影响 symbol fast path。

第一版 region state 也放入 `DistGlobal` shared state：

```cpp
constexpr int kRegionShardCount = 64;
constexpr int kRegionEntriesPerShard = 2048;

struct SharedRegionEntry {
    uint64_t base_addr;
    uint64_t begin;
    uint64_t end;
    int64_t producer_task_id;
    int16_t output_slot;
    int16_t kind;
    int32_t next;
};

struct SharedRegionShard {
    SharedPublishedCell published_frontier;
    SharedPublishedCell entry_count;
    int32_t bucket_heads[kRegionBucketCount];
    SharedRegionEntry entries[kRegionEntriesPerShard];
};

struct SharedRegionIndex {
    SharedRegionShard shards[kRegionShardCount];
};
```

region insert 只由 shared winner 执行：

1. 根据 buffer base / address range 选择 shard。
2. 用 64-bit fetch-add 或 shard-local bump 申请 entry。
3. 写 entry，flush entry。
4. 发布 bucket head / frontier。若 bucket head 需要 CAS，CCEC 侧必须用已验证
   的 64-bit 原子表达，不能引入 32-bit GM CAS。

region lookup 只在输入需要 overlap 语义时执行：

- symbol input 不进入 region lookup。
- external read-only 不进入 region lookup。
- `no_dep` 不进入 region lookup。
- explicit dep 只记录 dep task id，不等 region frontier。
- overlap lookup 扫描对应 shard/bucket，过滤 `producer_task_id < current_task_id`
  且 range overlap 的 entry，选最大 producer 作为 fanin edge。

conservative fallback 的正确边界是 region lookup 内部：

```text
wait_until(region_shard.published_frontier >= min_required_task)
scan region entries
```

不能在 shared winner 入口统一等待 `published_up_to >= task_id - 1`。统一等待会让
互不相关的 fresh-output task 串行化，违背 shared map 的性能目标。

### Shared Winner 顺序

shared winner 顺序必须让 descriptor 尽早可见：

1. 构建完整 `L0TaskArgs`。
2. 从全局 / 分片 heap 分配 fresh outputs。
3. 发布 fresh output descriptors 到 symbol index。
4. 发布 `INOUT` / `OUTPUT_EXISTING` descriptors 到 region index。
5. release 对应 published bits。
6. resolve consumed symbolic inputs。
7. 对 region inputs 做 region lookup。
8. collect fanin。
9. build private ring slot / joint anchor slot。

注意：发布自己的 outputs 不需要等待自己的 fanin。这样后续只需要 descriptor
的 task 不会被当前 winner 的输入等待卡住。

建议拆成以下 helper，避免把 shared 路径重新揉回 private submit：

```text
shared_materialize_outputs(tok, args, out_descs)
shared_publish_symbol_outputs(tok, out_descs)
shared_publish_region_outputs(tok, args, out_descs)
shared_resolve_symbol_inputs(args)
shared_collect_region_fanin(tok, args, fanin)
shared_build_winner_slot(tok, args, fanin)
```

`shared_materialize_outputs()` 只处理 fresh output descriptor 和 shared heap
allocation，不插入 private `DistTensorMap`。`shared_collect_region_fanin()` 不准
反查 symbol producer，symbol producer 的依赖来自 `FdwicOutputRef.producer_task_id`
或 explicit dep。

### Shared Heap

shared heap 一开始就按分片设计，即使第一版只打开一个 shard：

```cpp
constexpr int kSharedHeapShardCount = 64;

struct SharedHeapShard {
    SharedPublishedCell next;
    uint64_t base;
    uint64_t limit;
};

struct SharedHeapState {
    SharedHeapShard shards[kSharedHeapShardCount];
};
```

分配规则：

- fresh output 只由 winner 分配。
- shard 可按 `task_id`、`block_id` 或 output size class 选择；第一版优先
  `task_id & (kSharedHeapShardCount - 1)`，减少跨 core 热点。
- `next` 用 64-bit atomic fetch-add；CCEC 不引入 32-bit CAS。
- descriptor 中记录 heap epoch / addr / size，后续 reclaim 用 task completion
  frontier 做安全复用。
- private build 继续使用 per-core `heap_next`，不要让 shared heap 逻辑进入
  private 路径。

### Shared Loser

shared loser 只做：

```text
TaskOutputTensors result;
result.set_task_id(token.task_id);
result.add_symbolic_outputs(token.task_id, output_count);
return result;
```

禁止 shared loser 执行：

- `TensorCreateInfo::buffer_size_bytes`
- `init_tensor_from_create_info`
- heap bump
- TensorMap insert / lookup
- producer publish cursor update
- `L0TaskArgs` construction

## Private Path 设计

private 路径保持每核复制语义：

- 每核 `heap_next` 仍是确定性输出布局。
- 每核 `DistTensorMap` 必须完整。
- private loser 不能全跳过。

private winner：

1. materialize outputs。
2. prepare private map / retire。
3. collect fanin。
4. register outputs / inouts。
5. build private ring slot。

private loser：

1. 只构建 output / inout produce 侧 args。
2. materialize outputs，推进每核 `heap_next`。
3. prepare private map / retire。
4. register outputs / inouts。
5. 不 collect fanin，不 build，不 execute。

## Codegen Rules

codegen 需要生成两套 submit 模板，通过宏隔离，不在 runtime 做 mode 分支。

shared submit 生成规则：

- `rt_presubmit_*()` 前只保留控制流必须全员执行的代码。
- `if (tok.won)` 内构建完整 `L0TaskArgs`。
- symbolic input 在 `tok.won` 分支内以 `FdwicOutputRef` 加入 args。
- loser 分支只调用 shared 宏下的 `rt_submit_loser(tok, output_count)`。
- loser 分支不生成 output `TensorCreateInfo`、view、scalar packing、fanin dep
  计算或 heap size 计算。
- `output_count` 来自编译期已知 kernel schema；动态 output 暂不进入 shared
  fast path。

private submit 生成规则：

- winner 分支构建完整 args。
- loser 分支只构建 produce 侧 output args，保持 private map 完整。
- private loser 可跳过 input / scalar / explicit dep 构建。
- 旧 `rt_submit_*(kernel, args)` 只作为兼容入口，性能用例必须迁到 presubmit
  形态。

需要修改或新增的主要文件：

- `src/a5/runtime/fully_distributed_within_core/runtime/pto_types.h`
- `src/a5/runtime/fully_distributed_within_core/orchestration/pto_orchestration_api.h`
- `src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/dist_engine_api.h`
- `src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/submit_token.h`
- `src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/submit_private.h`
- `src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/submit_shared.h`
- `src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/shared_symbol_map.h`
- `src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/shared_region_map.h`
- `src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/common/shared_map_state.h`
- 相关 codegen 模板与 fdwic examples。

## File-Level Implementation Checklist

### `runtime/pto_types.h`

- 新增 `SharedTaskOutputs`，只存 `{producer_task_id, slot}`。
- `TaskOutputTensors` 保持 private/alloc 的 materialized descriptor 语义。
- private build 下保持现有真实 `Tensor *` 行为，不改变旧用例语义。
- 给 `SharedTaskOutputs::size()` / `empty()` 和 `TaskOutputTensors::size()` /
  `empty()` 保持一致的轻量查询语义，避免 codegen 因宏分支产生额外模板差异。
- 增加 debug-only guard：shared submit 不返回 `TaskOutputTensors`，private
  `TaskOutputTensors` 不能混入 symbolic ref。

### `orchestration/pto_orchestration_api.h`

- 暴露 `rt_presubmit_task` / `rt_presubmit_aic_task` /
  `rt_presubmit_aiv_task`。
- shared build 暴露返回 `SharedTaskOutputs` 的 winner/loser submit。
- private build 暴露返回 `TaskOutputTensors` 的 winner/loser submit。
- 旧 `rt_submit_*` wrapper 保留，但内部必须走 private/shared 对应实现，不再在
  wrapper 内做 winner claim。

### `dist_engine/dist_engine_api.h`

- 定义 device-callable API 的统一入口，不让 orchestration include aicore
  内部文件。
- `PTO_FDWIC_SHARED_MAP` 的宏选择只在 API wrapper 和 submit implementation
  边界出现；底层 helper 文件分别编译各自路径。
- 保证 shared/private 的 `SubmitToken` 类型一致，避免 codegen 为两条路径生成
  不同 task id 逻辑。

### `aicore/submit_token.h`

- 从现有 `dist_submit_begin()` 拆出 `dist_presubmit_begin()`。
- 从现有 drain/claim 逻辑拆出无参数版本：
  `dist_presubmit_drain()`、`dist_presubmit_claim()`。
- `SubmitToken.task_id` 必须在 candidate 判断前分配，所有 lane 对同一
  orchestration submit 得到相同 replay-local 序列。
- `SubmitToken.won` 只能由 cursor/joint ownership 决定，winner submit 不允许
  再次 claim。
- `SubmitToken.replay_outputs` 只服务 private loser；shared loser 忽略它。

### `aicore/submit_private.h`

- 把当前 `dist_submit_impl` 迁入 private winner/loser 拆分。
- private winner 复用现有 materialize、prepare map、fanin、register、build。
- private loser 只接收 produce-side args，不接收 input/scalar/explicit dep。
- private loser 仍执行 materialize/register，保证每核 private `DistTensorMap`
  和 `heap_next` 完整。
- Phase 1 完成时 `PTO_FDWIC_SHARED_MAP=0` 的 swimlane 和结果必须与当前主线
  等价。

### `aicore/submit_shared.h`

- 实现 shared 宏下的 `rt_submit_winner(tok, args)`，入口断言 `tok.won`。
- 实现 shared 宏下的 `rt_submit_loser(tok, output_count)`，入口断言 `!tok.won`。
- shared loser 只构造 symbolic `SharedTaskOutputs`，函数体不得调用任何
  materialize/map/heap/fanin/build helper。
- shared winner 顺序固定为 output descriptor publish -> input resolve ->
  region lookup -> fanin -> build。
- shared winner 只 build 当前 task 的 private ring slot / joint slot，不扫描整条
  private ring。

### `aicore/shared_symbol_map.h`

- 实现 `shared_publish_symbol_output(task_id, slot, Tensor)`。
- 实现 `shared_resolve_symbol_output(producer_task_id, slot)`。
- `published` 值写 producer task id，不写 bool。
- publish 使用 write desc -> store barrier -> flush desc -> release store
  published -> flush published。
- resolve 使用 invalidate published -> acquire load -> invalidate desc -> by-value
  copy Tensor。
- window 复用必须检查 `published == producer_task_id`，不能只看非零。

### `aicore/shared_region_map.h`

- 实现 `shared_publish_region_output()`，只由 winner 调用。
- 实现 `shared_collect_region_fanin()`，只处理 overlap 类输入。
- region fallback 等待只在 lookup 内部出现，不能放在 submit 入口。
- 第一版可以 conservative frontier，但必须保留替换为 per-shard/per-buffer
  frontier 的接口边界。
- 不使用 32-bit GM CAS；bucket head 如果需要原子 claim，改为 64-bit state 或
  使用 append-only entry 后离线扫描。

### `common/shared_map_state.h`

- 新增 `SharedMapState`，包含 symbol index、region index、shared heap。
- 该 state 归属 `DistGlobal` shared memory layout，由 host/AICPU 初始化。
- 所有跨 core 同步字段使用显式 64B cell 和 static_assert。
- 不使用 C++ `alignas(64)` 作为唯一布局保证。
- run-boundary attach/invalidate 必须覆盖 shared map 的 host-initialized fields。

### Host/AICPU Runtime Layout

- 扩展 `dist_engine_global_state_size/align`，把 `SharedMapState` 纳入 layout。
- register/init 时初始化 symbol published、region frontier、heap shard base/limit。
- AICPU 发布 layout 后 flush shared map init 区域。
- AICore replay 开始前 invalidate shared map layout 和 per-run counters。
- dirty build 下确认 runtime content fingerprint 能触发重建；关键 A/B 前明确
  清理或确认 build cache。

### Codegen/Templates

- 为 submit 生成 shared/private 两套模板，不生成 lambda 或 submit builder。
- shared 模板在 `tok.won` 前不构建 args。
- shared loser 分支不生成 output create-info/view/heap-size 相关表达式。
- private loser 分支只生成 produce-side output args。
- shared winner 分支把 `SharedTaskOutputs::output_ref(slot)` 直接加入 args；
  不生成 by-value `Tensor` 临时变量。
- 对动态 output count、动态 alias 关系、未知 overlap 的 submit，第一版降级到
  private 或 region slow path，不能污染 shared symbol fast path。

## Mandatory Invariants

- private map 和 shared map 是编译期隔离的两条 submit pipeline。
- `PTO_FDWIC_SHARED_MAP` 不允许变成运行时 flag。
- 不引入 `SubmitBuilder`、lambda、thunk 作为 submit hot path 抽象。
- 每个 orchestration submit 的 `task_id` 在 private/shared 下必须一致。
- `task_id` 分配发生在 candidate/winner 判断之前。
- `rt_presubmit_*()` 是唯一 claim 点；winner submit 不允许二次 claim。
- shared loser 必须是 O(output_count) symbolic return，不执行 args/materialize/map/
  heap/fanin/build。
- shared winner 发布自己的 output descriptor 不等待自己的 fanin。
- symbol input 只等 exact `{producer_task_id, slot}` published，不等 global prefix。
- region fallback 只能服务 overlap 类输入，不能拖累 fresh symbol path。
- external read-only、`no_dep`、explicit dep 不等待 region frontier。
- shared `published` 存 producer task id，禁止 bool published。
- 跨 core 同步字段默认 64B cacheline 隔离，禁止 byte-packed flag。
- CCEC 上板不使用 `__gm__` 32-bit CAS 或通用 GM `__atomic` 作为新同步 primitive。
- cache maintenance 只处理 descriptor/payload 可见性，不替代 atomic ownership。
- shared submit ABI 使用 `SharedTaskOutputs`，不暴露 `TaskOutputTensors` /
  `__gm__ Tensor&`。
- shared public API 不提供 `rt_resolve_output()`；symbolic input 只在 winner
  runtime materialize 阶段解析。
- private loser 仍维护 private `heap_next` 和 `DistTensorMap` 完整性。
- `dist_alloc_tensors(const L0TaskArgs&)` 按 submit 语义处理；不恢复
  `TensorCreateInfo...` convenience overload。
- 每阶段必须 sim 和 单卡 device 6 上板同时通过，失败阶段不继续堆后续功能。

## Debug Assertions and Instrumentation

第一版开发建议打开 debug-only 断言，避免性能路径悄悄退化：

- shared 宏下 `rt_submit_winner()`：assert `tok.won`，assert args 已构建且 output
  count 与 schema 一致。
- shared 宏下 `rt_submit_loser()`：assert `!tok.won`，assert 不访问 `g_self->map`、
  `heap_next`、shared heap cursor。
- `shared_publish_symbol_output()`：assert `published != task_id`，assert
  descriptor addr/size 非空，assert slot 小于 schema output count。
- `shared_resolve_symbol_output()`：assert observed published 只能是目标
  `task_id` 或旧 window 值；超时日志打印 producer task、slot、cell index。
- region lookup：记录是否走 conservative frontier，swimlane 中单独标注
  `RegionWait`，避免把它混入 `Fanin`。
- private path：assert `PTO_FDWIC_SHARED_MAP=0` 时仍不使用 shared state。
- shared path：assert fresh-output submit 不调用 `DistTensorMap::insert/lookup`。
- codegen smoke：通过 counter 或 swimlane 验证 shared loser 没有
  `Materialize`、`Register`、`Fanin`、`Build`。

需要新增或复用的 swimlane event：

- `Presubmit`
- `Claim`
- `SharedLoser`
- `SharedMaterialize`
- `SymbolPublish`
- `SymbolResolve`
- `RegionWait`
- `RegionLookup`
- `SharedBuild`

性能分析时 `SharedLoser` 应接近 claim 后常数开销；如果出现
`Materialize/Register/Fanin`，说明 codegen 或 shared loser helper 被污染。

## Test Matrix

每阶段命令之外，还必须覆盖以下语义组合。

### Submit ownership

- single AIC submit winner/loser。
- single AIV submit winner/loser。
- mixed AIC+AIV joint submit。
- 2V / multi-lane follower launch。
- repeated many tasks，覆盖 cursor 连续推进和 `block.won` slot 复用。

### Producer/consumer role

- AIC producer -> AIC consumer。
- AIC producer -> AIV consumer。
- AIV producer -> AIC consumer。
- AIV producer -> AIV consumer。
- joint producer -> single consumer。
- single producer -> joint consumer。

### Output dependency class

- fresh output chain：必须走 symbol fast path。
- output consumed by multiple later tasks：多个 consumer resolve 同一个 symbol。
- `INOUT`：走 region path。
- `OUTPUT_EXISTING`：走 region path。
- view/subview/alias overlap：走 region path。
- external read-only tensor：不等 map/frontier。
- `no_dep`：不等 region frontier。
- explicit dep：只记录 dep id，不做隐式 region wait。

### Alloc/reclaim

- `alloc_tensors(const L0TaskArgs&)` fresh output。
- alloc output 被 AIC consumer 消费。
- alloc output 被 AIV consumer 消费。
- alloc loser private replay。
- shared alloc loser do-nothing。
- heap shard wrap/reuse 前的 outstanding window 保护。

### Onboard stress

- repeated dual AIV / mixed smoke，覆盖 slot back-pressure。
- 多 case 同一 pytest worker 连续运行，覆盖 run-boundary attach/invalidate。
- descriptor field-wise copy 与 payload descriptor 来源覆盖。
- dirty tree rebuild 后重新跑，确认不是 stale CMake cache。
- 单卡 device 6 task-submit 上板，不使用 precheck。

## Phase Exit Criteria

每个 phase 完成前必须记录：

- 本阶段改动文件列表。
- 是否修改 runtime C++；若修改，是否已执行 editable rebuild。
- sim 命令和结果。
- 单卡 device 6 上板命令和结果。
- swimlane 对比结论。
- 是否新增 shared state；若新增，cacheline layout static_assert 是否覆盖。
- 是否新增跨 core 同步字段；若新增，atomic primitive 和 flush/invalidate 边界。
- 是否有临时 fallback；fallback 是否局限在 region slow path 或兼容入口。

## Cross-Phase Design Contracts

下面是防止后续 phase 走偏的设计合同。任何 phase 为了通过测试临时违反这些合同，
都必须停下来重审设计，不能把临时绕路带进下一阶段。

### Contract A: Submit ownership

`rt_presubmit_*()` 是唯一 ownership claim 点。后续所有 phase 都只能消费
`SubmitToken`，不能在 winner submit、shared map publish、region lookup、
joint follower launch 里重新 claim。

判定标准：

- `SubmitToken.task_id` 在 private/shared、winner/loser 下序列一致。
- `rt_submit_*_winner()` 入口只 assert `tok.won`，不调用 cursor claim。
- `rt_submit_*_loser()` 入口只 assert `!tok.won`，不调用 cursor claim。

### Contract B: Shared loser

shared loser 的最终形态从 Phase 3 开始就固定为 symbolic return。Phase 4-7 不能
因为 symbol/region/heap/joint 的实现需要，把 materialize、register、fanin 或
heap bump 加回 shared loser。

判定标准：

- shared loser 函数体只写 local `SharedTaskOutputs`。
- shared loser 不读写 `DistTensorMap`、shared symbol index、shared region index、
  shared heap、private heap。
- shared loser swimlane 只允许 `Presubmit/Claim/SharedLoser`。

### Contract C: Symbol fast path

fresh output 依赖的最终性能来自 exact symbol wait。Phase 4 之后，fresh-output
consumer 不能等待 global prefix、region frontier 或 private map。

判定标准：

- fresh output publish 只写 `{task_id, slot}` descriptor 和 exact published cell。
- fresh output resolve 只等待目标 producer task id 的目标 slot。
- `published_up_to >= task_id - 1` 只能出现在 region fallback 文档和 region
  lookup helper 内部，不允许出现在 shared submit 入口。

### Contract D: Region slow path

region index 是 overlap slow path，不是 shared map 的通用依赖机制。Phase 5 之后，
region fallback 不能扩大到 symbol input、external read-only、`no_dep` 或
explicit dep。

判定标准：

- region lookup 的调用点必须能说明输入属于 `INOUT` / `OUTPUT_EXISTING` /
  alias overlap。
- PA Case1 / fresh-output smoke 主路径的 swimlane 不出现 `RegionWait`。
- PA 或复杂 overlap 需要 fallback 时，fallback 只包在 region lookup 内。

### Contract E: Shared state ownership

shared symbol/region/heap state 都属于 host/AICPU 分配的 `DistGlobal` layout。
任何 phase 不得为了快速跑通，在 AICore device global、临时 handoff struct 或
per-core duplicate map 中建立第二份 shared state。

判定标准：

- shared map state size/align 纳入 runtime global layout。
- AICPU init 后 flush，AICore attach/run-boundary 首读前 invalidate。
- 新增同步字段有 64B cell/static_assert/atomic primitive 说明。

### Contract F: Joint/follower boundary

joint/follower 集成不能让 follower 触碰 shared map。shared winner 负责 publish
descriptor、resolve input、构建 payload；follower 只消费 winner 发布的
`BuiltSubtask` / payload。

判定标准：

- follower launch 不调用 `shared_resolve_symbol_output()`。
- follower launch 不调用 region lookup。
- joint output descriptor 只 publish 一次。
- joint completion 仍发布一个 task flag。

## Phase Design Gates

### Phase 1 Gate: Token split 不能改变语义

进入 Phase 2 前必须确认：

- old submit wrapper 与新 private presubmit path 的 task id、winner 分布一致。
- private path 仍完整执行 materialize/map/fanin/register/build。
- 没有引入 shared map state。
- 没有为通过 mixed/joint 临时引入 per-task claim。

### Phase 2 Gate: Codegen 先服务 private correctness

进入 Phase 3 前必须确认：

- generated shape 已经是 presubmit 后普通 `if (tok.won)` 分支。
- private loser 只构建 produce-side args，但仍维护 private map。
- old `rt_submit_*(kernel, args)` 兼容入口不进入性能测试路径。
- swimlane 证明 private loser 无 fanin/build，但仍有 private 必需的
  materialize/register。

### Phase 3 Gate: ABI 先锁死 loser do-nothing

进入 Phase 4 前必须确认：

- shared submit 不返回 `TaskOutputTensors`，而是返回 `SharedTaskOutputs`。
- `TaskOutputTensors/get_ref()` 只属于 private/alloc materialized descriptor
  路径，不作为 shared submit ABI。
- shared public API 不提供 `rt_resolve_output()`。
- shared symbolic input 以 `FdwicOutputRef` 进入 `L0TaskArgs`，只在 winner
  runtime materialize 阶段解析。
- shared loser 只用 `{task_id, output_count}` 构造 `{task_id, slot}`，不依赖
  args、heap、map、descriptor publish 或后续 Phase 4 的 shared symbol index。

### Phase 4 Gate: Symbol + heap 是性能主路径

进入 Phase 5 前必须确认：

- shared winner 可以发布 fresh output descriptor。
- shared consumer 可以 exact-slot resolve。
- shared heap 已按 shard ABI 落地，哪怕初始只启用部分 shard。
- fresh-output benchmark 不走 region path、不等 global prefix。
- shared loser swimlane 没有 materialize/register/fanin。

### Phase 5 Gate: Region 只能处理 overlap

进入 Phase 6 前必须确认：

- `INOUT` / `OUTPUT_EXISTING` / view overlap 通过 region path。
- symbol input、external read-only、`no_dep`、explicit dep 不进入 region wait。
- conservative frontier 没有出现在 shared submit 入口。
- region path 的 swimlane event 独立可见，不能被混入 symbol/fanin 热路径。

### Phase 6 Gate: Joint 不污染 shared map

进入 Phase 7 前必须确认：

- joint winner publish descriptor 一次。
- follower 不 resolve symbol、不查 region、不写 shared map。
- follower 只消费 payload / `BuiltSubtask`。
- mixed/2V 上板通过，且没有 immediate execute 或无边界 private ring scan 回归。

### Phase 7 Gate: 性能目标验收

完成 Phase 7 必须确认：

- shared map 功能覆盖完整 fully_distributed_within_core examples。
- shared loser critical span 明显低于 private loser。
- fresh symbol path 的 `SubmitExclusive` 低于 private map 路径。
- region path 若成为热点，下一步优化限定为 per-shard/per-buffer frontier，不改变
  symbol fast path。
- private build 仍可独立通过，证明宏隔离没有把 shared state 泄漏到 private。

## 阶段计划

每个阶段都必须同时通过 sim 和 a5 上板。上板固定使用单卡 device 6，通过 `task-submit`
持有设备锁；a5 不走 onboard precheck。

当前 CI pin：

```text
PTO_ISA_COMMIT=ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8
PTO_SESSION_TIMEOUT=90
```

运行 runtime/platform C++ 改动后需要重建：

```bash
source .venv/bin/activate
pip install --no-build-isolation -e .
```

上板通用命令形态：

```bash
task-submit --timeout 90 --max-time 90 --device 6 \
  --run "source .venv/bin/activate && python -m pytest <TARGET> \
    --platform a5 --device \$TASK_DEVICE -v --clone-protocol ssh \
    --require-pto-isa --pto-session-timeout 90 \
    --pto-isa-commit ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8"
```

### Phase 1: Split submit infrastructure

实现内容：

- 新增 `SubmitToken`。
- 新增 `rt_presubmit_task` / `rt_presubmit_aic_task` /
  `rt_presubmit_aiv_task`。
- 从现有 submit 中拆出 claim 前逻辑，确保 winner submit 不二次 claim。
- 新建 `submit_private.h`，把现有 `dist_submit_impl` 迁入 private path。
- `PTO_FDWIC_SHARED_MAP=0` 下行为与当前完全一致。

验收：

- task id 序列不变。
- winner 分布不变。
- full smoke / PA Case1 结果不变。

sim：

```bash
python -m pytest examples/a5/fully_distributed_within_core/simple_orch_smoke \
  examples/a5/fully_distributed_within_core/submit_dependency_smoke \
  --platform a5sim --device 0-15 -p no:xdist -v -rs --manual include \
  --use-example-exec-time --clone-protocol https \
  --require-pto-isa --pto-session-timeout 90 \
  --pto-isa-commit ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8
```

```bash
python -m pytest examples/a5/fully_distributed_within_core/paged_attention_unroll \
  --platform a5sim --device 0-15 -p no:xdist -v -rs --case Case1 \
  --use-example-exec-time --clone-protocol https \
  --require-pto-isa --pto-session-timeout 90 \
  --pto-isa-commit ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8
```

上板：

```bash
task-submit --timeout 90 --max-time 90 --device 6 \
  --run "source .venv/bin/activate && python -m pytest \
    examples/a5/fully_distributed_within_core/simple_orch_smoke \
    examples/a5/fully_distributed_within_core/submit_dependency_smoke \
    --platform a5 --device \$TASK_DEVICE -v -rs --manual include --clone-protocol ssh \
    --require-pto-isa --pto-session-timeout 90 \
    --pto-isa-commit ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8"
```

```bash
task-submit --timeout 90 --max-time 90 --device 6 \
  --run "source .venv/bin/activate && python -m pytest \
    examples/a5/fully_distributed_within_core/paged_attention_unroll \
    --platform a5 --device \$TASK_DEVICE -v -rs --case Case1 --clone-protocol ssh \
    --require-pto-isa --pto-session-timeout 90 \
    --pto-isa-commit ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8"
```

### Phase 2: Private presubmit codegen shape

实现内容：

- codegen / hand-written high-value examples 改为 presubmit 后分支。
- private winner 构建 full args。
- private loser 只构建 output / inout produce args。
- 保留旧 `rt_submit_*(kernel, L0TaskArgs)` 兼容入口，但性能测试使用新形态。

验收：

- private 功能与 Phase 1 一致。
- loser swimlane 不再出现 fanin / build。
- loser materialize/register 仍存在，这是 private 正确性要求。

sim / 上板同 Phase 1，额外打开 swimlane：

```bash
python -m pytest examples/a5/fully_distributed_within_core/simple_orch_smoke \
  examples/a5/fully_distributed_within_core/submit_dependency_smoke \
  --platform a5sim --device 0-15 -p no:xdist -v -rs --manual include \
  --use-example-exec-time --enable-l2-swimlane \
  --clone-protocol https --require-pto-isa --pto-session-timeout 90 \
  --pto-isa-commit ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8
```

### Phase 3: Shared symbolic output ABI

实现内容：

- 新增独立 `SharedTaskOutputs`，shared submit 返回它，不返回
  `TaskOutputTensors`。
- shared 模式下 `rt_submit_loser(token, output_count)` 返回符号输出。
- `L0TaskArgs` / `TensorRef` 在 shared 宏下支持 `FdwicOutputRef` input slot。
- codegen 在 shared winner 分支把 `outs.output_ref(slot)` 加入 args，不在
  orchestration 侧 resolve 成 `Tensor`。
- shared winner runtime 内部 materialize 参数时解析 symbolic input，并把真实
  descriptor 写入当前 task payload。
- private 模式保持真实 tensor pointer 行为。

验收：

- shared loser 不构建 args 也能把输出交给下游 winner。
- 下游 winner runtime 内部解析 symbolic input 后 descriptor 完整：
  addr / size / shape / dtype / owner_task_id 正确。
- shared public API 中不存在 `rt_resolve_output()`。
- shared public submit 返回类型不是 `TaskOutputTensors`。
- shared 模式尚可只跑 symbol-only 用例，不要求 region overlap 完整。

sim：

```bash
python -m pytest examples/a5/fully_distributed_within_core/paged_attention_unroll \
  --platform a5sim --device 0-15 -p no:xdist -v -rs --case Case1 \
  --use-example-exec-time --clone-protocol https \
  --require-pto-isa --pto-session-timeout 90 \
  --pto-isa-commit ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8
```

上板：

```bash
task-submit --timeout 90 --max-time 90 --device 6 \
  --run "source .venv/bin/activate && python -m pytest \
    examples/a5/fully_distributed_within_core/paged_attention_unroll \
    --platform a5 --device \$TASK_DEVICE -v -rs --case Case1 --clone-protocol ssh \
    --require-pto-isa --pto-session-timeout 90 \
    --pto-isa-commit ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8"
```

### Phase 4: Shared symbol index and sharded heap

实现内容：

- 新增 shared symbol descriptor table。
- 新增 per-slot published bits。
- 新增 shared heap allocator。初始版本可以全局 bump；同阶段内完成
  per-shard heap，避免后续重写 descriptor ABI。
- shared winner 分配 fresh outputs 并发布 descriptor。
- symbolic input materialize 等 exact symbol published bit，不等 global `P`。

验收：

- fresh-output 链路完全不走 region map。
- shared loser 没有 materialize / register / replay swimlane 成本。
- symbol resolve 只等待实际 producer slot。

sim / 上板：

```bash
python -m pytest examples/a5/fully_distributed_within_core/paged_attention_unroll \
  --platform a5sim --device 0-15 -p no:xdist -v -rs --case Case1 \
  --use-example-exec-time --enable-l2-swimlane \
  --clone-protocol https --require-pto-isa --pto-session-timeout 90 \
  --pto-isa-commit ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8
```

```bash
task-submit --timeout 90 --max-time 90 --device 6 \
  --run "source .venv/bin/activate && python -m pytest \
    examples/a5/fully_distributed_within_core/paged_attention_unroll \
    --platform a5 --device \$TASK_DEVICE -v -rs --case Case1 --enable-l2-swimlane \
    --clone-protocol ssh --require-pto-isa --pto-session-timeout 90 \
    --pto-isa-commit ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8"
```

#### Phase 4 实现闭环记录

实现文件：

- `runtime/dist_engine/common/state.h`
- `runtime/dist_engine/aicpu/control_plane.h`
- `runtime/dist_engine/aicore/onboard_entry.h`
- `runtime/dist_engine/aicore/submit_core.h`
- `runtime/dist_engine/aicore/submit_runtime.h`
- `examples/a5/fully_distributed_within_core/shared_symbol_smoke/`
- `examples/a5/fully_distributed_within_core/paged_attention_unroll/`

已落地内容：

- shared descriptor table 为 `shared_outputs[task_id].tensors[slot]`。
- published bit 为 per-task/per-slot，
  `shared_outputs[task_id].published[slot]`。
- shared heap 使用 `kSharedHeapShards == kCursorShards`，active shard 全打开。
- shared winner fresh output materialize 后 publish descriptor，再 release publish
  对应 slot。
- shared symbolic input materialize 等待
  `published[producer_slot] == producer_task_id`，不等 global prefix。
- shared loser 只根据 `{task_id, output_count}` 返回 symbol handle，不构建
  `L0TaskArgs`，不 materialize，不 register，不 fanin，不 replay。

关键坑点与修复：

- shared winner 构建 slot 时，解析 symbolic input 可能重入
  `drain_phase_b()`。因此 `RingSlot::built` 不能在 descriptor、scalar、args、
  fanin 全部填完前发布。
- 正确做法是在 `build_ring_slot()` /
  `build_ring_slot_from_submit()` 开始先写 `built=false`，所有字段完整后最后写
  `built=true`。否则 PA 的 QK->SF 链路会执行到半初始化 slot，表现为真实 sim
  崩溃或上板 507000。
- 这个问题不能通过单 shard、跳过 golden、跳过后续 submit、或只跑
  `--use-example-exec-time` 规避；必须修 publication ordering。

Phase 4 验证记录：

- shared clean rebuild：
  `rm -rf build/cache/a5/sim build/cache/a5/onboard &&
  CXXFLAGS='-DPTO_FDWIC_SHARED_MAP=1' python simpler_setup/build_runtimes.py
  --platforms a5sim a5`，通过。
- shared PA Case1 真实 sim，不带 `--use-example-exec-time`，通过。该项用于确认
  真实 kernel 执行不再触发半初始化 slot 问题。
- shared PA Case1 sim gate：
  `--use-example-exec-time --enable-l2-swimlane`，通过。
- shared PA Case1 onboard gate：
  `task-submit --timeout 90 --max-time 90 --device 6 ... --case Case1
  --enable-l2-swimlane`，通过。
- shared symbol smoke sim：
  `--manual include --use-example-exec-time`，通过。
- shared symbol smoke onboard：
  `task-submit --timeout 90 --max-time 90 --device 6 ... --manual include`，
  通过。
- default/private clean rebuild：
  `rm -rf build/cache/a5/sim build/cache/a5/onboard &&
  python simpler_setup/build_runtimes.py --platforms a5sim a5`，通过。
- default/private full `submit_dependency_smoke` sim：
  `--manual include --use-example-exec-time`，通过。
- default/private PA Case1 sim：
  `--case Case1 --use-example-exec-time`，通过。
- default/private full `submit_dependency_smoke` onboard：
  `task-submit --timeout 90 --max-time 90 --device 6 ... --manual include`，
  通过。
- default/private PA Case1 onboard：
  `task-submit --timeout 90 --max-time 90 --device 6 ... --case Case1`，
  通过。

swimlane 结论：

- 最新 shared PA Case1 swimlane 中，kernel winners 有
  `Materialize/Fanin/Build/Kernel/Commit`。
- kernel losers 只有 presubmit claim，不出现 loser-side
  `Materialize/Register/Fanin/Build/Kernel`。
- `Register/PrepareMap` 事件来自 `alloc_tensors`，不是 shared fresh-output
  submit path。
- fresh symbol resolve 代码只等待 producer 的 exact slot published bit；不存在
  `published_up_to >= task_id - 1` 或 global prefix wait。

提交状态：

- Phase 4 代码已提交为 `d8fc96a7 Add: fdwic shared symbol heap phase`。
- 该 commit 包含 runtime shared descriptor/published-slot/sharded-heap 改动、
  shared symbol smoke multi-output/shard 用例、PA shared winner-first submit
  改造。
- 注意：该 commit 内的 plan 文档曾被错误改成英文压缩版。当前工作区已从
  git blob `6991213dff0e2dd7a21b35f19028ab21d1c39c89` 恢复原始 1519 行中文
  plan，并在此基础上继续增量更新。

### Phase 5: Shared region index

实现内容：

- 新增 shared region index，仅用于 `INOUT` / `OUTPUT_EXISTING` / alias overlap。
- region lookup 可先用 conservative publish frontier 兜底。
- fresh symbol path 不依赖 region frontier。
- 记录 region path swimlane，确认它不是主热点。

具体实现路径：

1. **新增 shared region state。**

   不复用 `DistCore::map`。`DistCore::map` 是 private per-core producer map，
   shared path 必须有独立的 global region index，放在 `DistGlobal` 的
   `#if PTO_FDWIC_SHARED_MAP` 分支下。

   建议第一版数据结构：

   ```cpp
   constexpr int32_t kSharedRegionBuckets = ...;
   constexpr int32_t kSharedRegionCap = ...;

   struct SharedRegionEntry {
       uint64_t buf_addr;
       uint64_t lo;
       uint64_t hi;
       int32_t producer;
       int32_t bucket;
       int32_t next_in_bucket;
       int32_t next_in_task;
   };

   struct SharedRegionMap {
       volatile int32_t high_water;
       volatile int32_t buckets[kSharedRegionBuckets];
       volatile int32_t task_heads[kTaskWindow];
       SharedRegionEntry entries[kSharedRegionCap];
   };
   ```

   如果担心 `DistGlobal` reserved arena 尺寸，先用较小 cap，并加
   `static_assert(sizeof(DistGlobal) <= kDistEngineGlobalStateSize)` 保护。

2. **AICPU 初始化。**

   在 `dist_engine_register()` 中 reset shared region map：

   - `high_water = 0`
   - 所有 buckets 置 `-1`
   - 所有 task_heads 置 `-1`

   CCE attach 时如果 AICore 会读这些 global 控制字段，需要按已有 pattern
   invalidate 对应 cacheline。不要新增环境变量或 runtime flag。

3. **实现 shared region helper。**

   可以先放在 `submit_core.h` 附近，稳定后再拆成
   `aicore/shared_region_map.h`。

   最小 helper 集合：

   ```text
   dist_shared_region_byte_range(tensor, addr, lo, hi)
   dist_shared_region_hash(addr)
   dist_shared_region_lookup(tensor) -> max overlapping producer
   dist_shared_region_insert(tensor, producer)
   dist_shared_region_relevant(tag, tensor_ref) -> bool
   ```

   byte range 计算可以复用 private `dist_tensor_map_byte_range()` 的规则：

   - `buf_addr = tensor.buffer.addr`
   - `lo = start_offset * element_size`
   - contiguous tensor 用 shape product 计算 extent
   - non-contiguous tensor 使用 `extent_elem_cache`

4. **严格限定进入 region path 的 tag。**

   Phase 5 第一版只允许这些路径查 region：

   - `TensorArgType::INOUT`
   - `TensorArgType::OUTPUT_EXISTING`
   - 能证明会读写 existing region 的 overlap/view case

   这些路径不能查 region：

   - `TensorArgType::OUTPUT` fresh output
   - `FdwicOutputRef` symbolic input
   - read-only external input
   - `manual_dep` / `no_dep` / explicit dep 已表达的路径

   判断不了的 case 不能悄悄走全局等待；要先做最小正确分类，缺口补测试后再扩。

5. **拆 shared fanin 收集。**

   当前 shared winner 使用 `dist_submit_collect_symbol_fanin()`。Phase 5 应改成
   更明确的 shared 版本，例如：

   ```text
   dist_submit_collect_shared_fanin(args, ctx, fanin):
       for each tensor arg:
           if tag == OUTPUT:
               continue

           if tensor_from_shared_output:
               add shared_output_ref.producer_task_id
               continue

           owner_task_id != UINT64_MAX:
               add owner_task_id

           if dist_shared_region_relevant(tag, tensor):
               p = dist_shared_region_lookup(tensor)
               add p
   ```

   这里 symbolic producer fanin 和 region producer fanin 可以共享最终
   `ctx.fanin[]`，但 swimlane flags 必须能区分 symbol 和 region 成本。

6. **新增 shared region register。**

   shared winner 在 fanin 收集后、build slot 前注册它产生的 existing-region
   producer：

   ```text
   dist_submit_register_shared_regions(ctx, args):
       for each tensor arg:
           if tag == INOUT or tag == OUTPUT_EXISTING:
               insert resolved/current descriptor with ctx.task_id
   ```

   fresh `OUTPUT` 不注册到 region map。fresh output 已经通过 Phase 4 的
   descriptor table 表达 producer，不能重复进入 region slow path。

7. **submit runtime 接入点。**

   shared `dist_submit_winner_impl()` 的 Phase 5 目标顺序：

   ```text
   check task cap
   materialize args / fresh outputs
   publish shared fresh output descriptors
   collect shared fanin:
       symbol exact producer
       owner_task_id
       region overlap producer
   register shared regions for INOUT / OUTPUT_EXISTING
   build winner task
   ```

   不要调用 private 的 `dist_submit_materialize_and_prepare_map()`，因为它会走
   private map 的 `PrepareMap/Register` 语义。

8. **swimlane 标记。**

   Phase 5 不一定要新增 enum，但必须可区分：

   - symbol fanin
   - region lookup
   - region register

   如果复用 `TracePhase::Fanin` / `TracePhase::Register`，用 flags/aux 标记
   shared region，避免后续误判 PA fresh path 也付了 region 成本。

验收：

- `submit_dependency_smoke` 的 INOUT / view / overlap 场景通过。
- no-dep / explicit-dep 不等待 region frontier。
- region fallback 不影响 PA Case1 / fresh-output smoke 主路径。
- shared PA Case1 swimlane 中不得出现 region lookup/register。
- shared symbol smoke swimlane 中不得出现 region lookup/register。
- shared loser 仍然只有 presubmit claim 和 symbolic return，不新增
  materialize/register/fanin/build/kernel。

Phase 5 开发顺序：

1. 先实现 shared region state/reset/helper，不接 submit，clean build。
2. 接入 shared winner fanin lookup 和 register，保持 shared loser 不变。
3. 先跑 shared `submit_dependency_smoke` sim，定位第一个 overlap/INOUT 缺口。
4. 根据失败 case 补最小 shared 用例或打开既有 manual case，不先大改 PA。
5. shared `submit_dependency_smoke` sim/onboard 全量通过后，再回归 shared PA
   Case1 和 shared symbol smoke，确认 fresh path 没被污染。
6. 最后跑 default/private clean build、full smoke、PA Case1 sim/onboard 回归。

sim / 上板：

```bash
python -m pytest examples/a5/fully_distributed_within_core/submit_dependency_smoke \
  --platform a5sim --device 0-15 -p no:xdist -v --manual include \
  --clone-protocol https --require-pto-isa --pto-session-timeout 90 \
  --pto-isa-commit ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8
```

```bash
task-submit --timeout 90 --max-time 90 --device 6 \
  --run "source .venv/bin/activate && python -m pytest \
    examples/a5/fully_distributed_within_core/submit_dependency_smoke \
    --platform a5 --device \$TASK_DEVICE -v --manual include \
    --clone-protocol ssh --require-pto-isa --pto-session-timeout 90 \
    --pto-isa-commit ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8"
```

注意：

- shared map 当前不支持用 `--enable-dep-gen` 作为验证手段，private map 也没有
  支撑该参数。不要把 dep-gen 失败归因到 shared map 功能正确性。
- `--enable-l2-swimlane` 只用于性能与泳道分析，不用于普通 golden 功能验证。
- 上板固定使用单卡 `--device 6`，不是 `--device-num 6`。

#### Phase 5 当前闭环记录

截至 2026-07-23，当前代码已经完成 Phase 5 的 shared region 主功能闭环：

- shared runtime clean build：`a5sim` / `a5` 通过，编译时使用
  `CXXFLAGS='-DPTO_FDWIC_SHARED_MAP=1'`，并清理 a5 FDWIC build cache 后重建。
- shared smoke sim 全量通过：
  `simple_orch_smoke`、`shared_symbol_smoke`、`submit_dependency_smoke`，
  使用 `--manual include`，不带 `--use-example-exec-time`，不带
  `--enable-dep-gen`。
- shared PA Case1 sim 通过 golden 验证，不带 `--use-example-exec-time`。
- shared smoke 上板全量通过，固定 `task-submit --device 6`。
- shared PA Case1 上板通过，`--enable-l2-swimlane` 生成
  `outputs/TestPagedAttentionUnroll_Case1_20260723_150651/merged_swimlane.json`，
  端到端 span 为 `2285.352us`。
- `merged_swimlane.json` 已保留 FDWIC runtime 事件的 `flags` / `aux`，可以直接
  区分 fanin/register/resolve 的来源。该次 PA 中 `register flags=0`、
  `aux_sum=0`，说明 PA fresh-output 主路径没有 region register。
- default/private clean build：`a5sim` / `a5` 通过，不带
  `PTO_FDWIC_SHARED_MAP`。
- default/private smoke sim/onboard 通过：
  `simple_orch_smoke`、`submit_dependency_smoke`，使用 `--manual include`。
- default/private PA Case1 sim/onboard 通过。

仍需保留为后续阶段或限制说明的事项：

- 当前 runtime 仍以 `kFlagCap` 为任务上限，不是完整 task ring generation
  复用能力。
- shared region intent 目前依赖 orchestration/codegen 显式选择
  `*_with_region_intent` API；后续 codegen 需要系统接入，不能靠用例手写兜底。
- mixed/joint task 的完整 shared map 支持属于 Phase 6，Phase 5 不声明完成。

### Phase 6: Joint task and follower launch integration

实现内容：

- shared winner 对 joint task 发布所有 output descriptors。
- joint follower launch 使用 winner 已 resolve 的 args，不自行查 fanin。
- follower 不触碰 shared map。
- joint completion 仍只发布一个 task flag。

验收：

- MIX / 2V co-owner 用例通过。
- follower swimlane 不出现 shared map lookup。
- joint output descriptor 只发布一次。

#### Phase 6 设计草案：winner-gated lane inbox

以下内容原本被误写入 `docs/fully_distributed_within_core.md`，现作为未验证的
phase 6 草案保留在本 plan。它不是主设计文档的替代方案；只有在实现、sim、
上板和泳道数据都闭环后，才能再讨论是否同步到主文档。

目标是把 joint task 的 follower 依赖解析从 follower 本地 fanin 轮询中移出：

- joint task 仍由 anchor/winner 负责 claim 和完整 fanin resolve。
- anchor/winner 先把自己的子任务构建进私有环，槽内保留 `fanin[]`。
- Phase B 中 anchor/winner 观察到 fanin 全部 ready 后，向同 block 的目标
  follower lane 发布一条 launch。
- follower 不参与该 joint task 的 claim，不查 shared map，不解析 fanin；
  只 drain 自己 lane 的 inbox。
- follower 收到 launch 后构建本核私有环槽，`fanin_count = 0`，表示依赖已经
  由 winner 收敛完成。
- joint task 仍只有一个 task completion flag。各 co-owner 执行完自己的子任务后
  递减同一个 `task_cell[N].remaining`，最后一个完成者发布 `flag(N)`。

建议状态模型：

```text
lane_inbox[block][lane]:
    单 writer：该 block 的 anchor/winner
    单 reader：对应 follower lane
    entry：kernel/args/payload 引用，不携带 fanin 列表
    push：release
    pop：acquire
```

关键语义：

- launch 只表示 winner 已经确认 joint task 的 fanin ready，不表示 winner
  自己的 kernel 已完成。
- follower 进入 kernel 前，在非一致缓存平台上仍需对 launch 中输入地址做必要的
  invalidate / 旁路读；launch 的 acquire 不能替代数据面可见性维护。
- follower 不按编排走位等待 anchor 决定，也不通过 task id 查询
  `block.won[N]`；没有 launch 就继续执行本 lane 其它就绪工作。
- anchor 超前时，launch 在 inbox 中累积；inbox 满时 anchor 在 claim/build
  前反压，先执行 Phase B / drain，不能无限超前。

这个草案和 shared map 的关系：

- shared map 只服务 submit 依赖与 output descriptor 发布。
- joint follower launch 是多核任务内部的 owner handoff，不应让 follower 再走
  shared region lookup/register。
- shared winner 只发布一次 output descriptor；follower 不重复发布 output
  descriptor。
- 如果后续 phase 6 发现 follower swimlane 仍出现 shared map lookup，说明 joint
  handoff 没隔离好，不能进入 phase 7。

落地风险：

- 当前 a5 代码仍可能使用 `BlockWon` / `WonSlot` 这类 block 内多方共享结构；
  迁移到 `lane_inbox` 必须先审计现有 joint completion 和反压语义。
- `remaining` 不应放在 inbox entry 中，否则多个 follower / anchor 的完成路径
  会重新引入 block-local 多方共享状态；应放在 per-task completion cell。
- inbox entry 不能按 task id 简单取模覆盖；必须有容量反压和 release/acquire
  可见性。
- 不能用该草案规避 PA shared map 的真实性能问题。PA 的 `INOUT`/alias 路径必须
  由 shared map runtime 正向优化，不能在用例侧手动建立依赖。

sim / 上板目标：

```bash
python -m pytest examples/a5/fully_distributed_within_core/simple_orch_smoke \
  --platform a5sim --device 0-15 -p no:xdist -v --use-example-exec-time --clone-protocol https \
  --require-pto-isa --pto-session-timeout 90 \
  --pto-isa-commit ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8
```

```bash
task-submit --timeout 90 --max-time 90 --device 6 \
  --run "source .venv/bin/activate && python -m pytest \
    examples/a5/fully_distributed_within_core/simple_orch_smoke \
    --platform a5 --device \$TASK_DEVICE -v --clone-protocol ssh \
    --require-pto-isa --pto-session-timeout 90 \
    --pto-isa-commit ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8"
```

### Phase 7: Full a5 shared-map qualification

实现内容：

- 打开 shared macro 后跑完整 a5 fully_distributed_within_core examples。
- 对比 private vs shared swimlane。
- 如果 region path 成为热点，再实现 per-buffer/per-shard frontier。

sim：

```bash
python -m pytest examples/a5/fully_distributed_within_core \
  --platform a5sim --device 0-15 -p no:xdist -v --use-example-exec-time --enable-l2-swimlane \
  --clone-protocol https --require-pto-isa --pto-session-timeout 90 \
  --pto-isa-commit ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8
```

上板单卡 device 6：

```bash
task-submit --timeout 90 --max-time 90 --device 6 \
  --run "source .venv/bin/activate && python -m pytest \
    examples/a5/fully_distributed_within_core \
    --platform a5 --device \$TASK_DEVICE -v --enable-l2-swimlane \
    --clone-protocol ssh --require-pto-isa --pto-session-timeout 90 \
    --pto-isa-commit ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8"
```

## 性能验收标准

shared map 达标必须满足：

- shared loser 无 `Materialize`。
- shared loser 无 `Register`。
- shared loser 无 fanin lookup。
- shared loser 不写 heap / map / publish frontier。
- fresh-output resolve 不等待 global publish prefix。
- symbol resolve 为 O(1) exact-slot wait。
- shared `SubmitExclusive` 和 critical span 明显低于 private。

使用：

```bash
python scripts/analyze_fdwic_swimlane_critical_path.py <l2_swimlane_records.json>
```

重点比较：

- `SubmitExclusive`
- `Materialize`
- `Register`
- `Fanin`
- `Replay`
- critical core span

## 风险与决策点

- `TaskOutputTensors` ABI 是最大重构点，必须优先完成，否则 shared loser
  无法 do-nothing。
- symbol index 必须先于 region index 落地，避免 fresh-output 主路径被 region
  fallback 污染。
- global publish prefix 只能作为 region fallback，不能用于所有输入。
- shared heap 一开始就按可分片 descriptor 设计，即使第一版先用单 bump，也不要让
  descriptor ABI 依赖单 bump。
- 如果某阶段 sim 过但上板单卡 device 6失败，不进入下一阶段；先修复当前阶段的 onboard
  可见性、cache flush/invalidate、或 atomic ordering 问题。

## Onboard Pitfalls Checklist

本节复制整理自
`/home/pyptouser/chenpeng/.claude/plans/mellow-munching-cocoa.md` 末尾坑点区，
用于 shared map 设计和开发时随时核对。若原坑点文档后续更新，需要同步刷新本节。

### 全局状态内存归属

1. `DistGlobal` 不能在 device 侧分配。shared symbol / region / heap state
   必须由 host 按 size/align 分配并通过 `runtime->dist.shared_addr` 发布，
   AICore 只 attach。
2. AICPU/AICore 不能各自初始化一份 global/per-core 状态。shared map layout、
   heap base/size、worker 数量由 host/AICPU 初始化，AICore 只做 per-core
   attach/reset。
3. `DistCore` 不能挂在临时 handoff 状态里。shared map 新状态必须进入正式
   `DistGlobal` / `DistCore` layout，不建立第二套 submit。

### Cache 一致性和 cacheline 粒度

1. AICPU publish 的 GM state 不会自动对 AICore 可见。AICPU 唤醒前 flush；
   AICore 首读 shared map layout / run-global 字段前 invalidate。
2. byte-packed completion flag 上板会被 cacheline 写回覆盖。shared map 的
   `published` / frontier / claim 字段不能 byte-pack。
3. 相邻 `uint64_t` 仍共享 cacheline。跨 core 独立写并需要 flush 的同步字段
   默认独占 64B cell。
4. `__builtin_memcpy` / `aicore_memcpy` 不是可见性协议。descriptor publish 后
   必须明确 flush，resolve 前必须明确 invalidate。
5. 写完 GM 数据是否需要 dcci 要按数据交接边界判断。shared descriptor、region
   entry、winner payload 是跨 core 交接点；private ring slot 不是。

### CCEC 编译和地址空间

1. AICore 栈物理落点和 CCEC 地址空间类型不同。local `Tensor` 保持默认地址
   空间，只有真实 GM 指针/引用标注 `__gm__`。
2. `__gm__` 指针不能随意 reinterpret 成普通指针。shared map descriptor 搬运
    不要做 GM pointer 到 non-GM pointer 的地址空间转换。
3. `__gm__ this` member 调用受限。shared resolve 返回 by-value local
    `Tensor`，避免对 GM tensor 做 member call。
4. CCEC GM atomic 支持面窄。共享 cursor/frontier/published/claim 字段优先
    使用已验证的 64-bit atomic primitive，不用通用 GM `__atomic` helper 或
    32-bit CAS。
5. `aicore_memcpy` 不能随意缓存 GM base 指针到局部再复用。复杂 GM pointer
    复用需要单独验证。
6. `Tensor::init_from_line1` 不能直接替换成 64B `aicore_memcpy`。line1 复制
    保持 field-wise。
7. `TensorCreateInfo` field copy 和 descriptor line copy 不能混为一谈。
    create-info 到 Tensor descriptor 的已验证 64B materialize 可保留，
    create-info 自身 field copy 继续 field-wise。
8. `fill_tensor_initial_value` 首段不能直接按 tmr 形态换成 `aicore_memcpy`。
    首段保持逐 byte GM store，已验证的 GM->GM doubling 才可使用。

### 真实 submit 语义迁移

1. 临时 CCEC handoff 字段容易变成另一套 submit。shared map 必须围绕真实
    submit 的 token、payload、ring、completion 阶段扩展。
2. `dist_alloc_tensors` 也是 submit 路径。shared heap / symbol publish 设计
    需要覆盖 alloc submit，不能只改 kernel submit。
3. mixed submit 需要 joint completion，不是单 owner flag。shared winner 只
    发布一次 output descriptor，joint completion 仍聚合成一个 task flag。
4. output-only 持久化不够，input/view/scalar 也要持久化。winner 构建 args
    后必须进入 `DistTaskPayload`，后续 ring/follower 只读 payload。
5. payload ring 太小会覆盖仍在使用的 task descriptor。shared symbol window
    和 payload window 都要大于 outstanding task，并设计复用保护。
6. alloc loser 不能在语义不完整时提前返回。private alloc loser 仍需维护
    private producer 流；shared alloc loser 只能在 symbol/heap publish 语义由
    winner 完整承担后 do-nothing。
7. PA 太复杂，不适合作为第一验收。先用 smoke 覆盖 dependency、many tasks、
    mixed submit、alloc/reclaim、overlap/fanin，再推进 PA。
8. `block.won` slot 不能按 task id 取模复用。joint/shared follower 交接需要
    扫描空闲 slot 并 reserve，slot 满在 build 阶段 back-pressure。
9. GM cacheline flush 必须避免写回邻接控制字段。shared map 同步 cell 与
    descriptor/payload 分开，follower drain 只 flush 自己拥有的 cell。
10. 后续跨 core 原子/同步字段默认按 cacheline 隔离。只有只读、单 writer、
    整体 publish record 或纯硬件原子且不会 flush 邻接脏数据的字段可以例外。
11. 每轮 run 的 GM 初始化必须配套 AICore attach/invalidate。shared map
    AICPU 初始化字段必须接入 run-boundary attach/invalidate。
12. winner build 后不能无边界扫描整条 private ring。shared winner 只 build
    当前 slot，ready slots 由统一 drain 执行。
13. 去掉 immediate execute 必须同时补 final drain。shared path 不得让 replay
    末尾 task 留在 ring 里未执行。
14. 同 lane 顺序执行会掩盖跨 role TensorMap 注册缺陷。shared map 验证要覆盖
    producer role x consumer role x descriptor 来源。
15. C++ `alignas(64)` 在当前 CCEC/onboard 路径上不是可靠布局手段。shared
    descriptor/control cell 使用显式 padding 加 `sizeof/offsetof` static_assert。
16. 便捷 `alloc_tensors(TensorCreateInfo...)` 会绕开真实 submit 语义。fdwic
    只保留 `alloc_tensors(const L0TaskArgs&)`，alloc 必须经过 submit 阶段。
17. payload 里已有 producer metadata 时，不应再依赖栈上中转做 fan-in。shared
    fanin 优先读 payload/descriptor 中的 `owner_task_id`。
18. dirty worktree 下 CMake cache stale 会伪造/掩盖上板根因。关键 A/B 前确认
    runtime 已重建，必要时清理 build cache。
19. lint/clang-tidy compile DB 要匹配实际编译器环境。不要为了 lint include
    失败污染业务 include。
20. 不要用临时 per-task claim 绕开真实 cursor ownership。presubmit winner
    ownership 仍基于真实 cursor/fetch-max 证明。
21. 不要把整段 orchestration replay 误记成单个 submit 的 Replay。shared
    swimlane 采集点要落在每个 submit 阶段。
22. raw collector schema 可以不同，但 converted swimlane 不能变形。性能比较
    继续输出既有 `merged_swimlane.json` 展示约定。
23. 通过 smoke/PA 不等于真实 submit 语义完整。shared map 新增 atomic、
    progress、back-pressure 字段要逐项审计不变量。
24. CCEC 上板不支持 `__gm__` 32-bit CAS 作为通用原子 primitive。shared
    bucket head、heap cursor、published/frontier 不引入 32-bit GM CAS。
25. 不能用 cache maintenance 替代真实 submit 的原子字段语义。cache flush /
    invalidate 只处理 payload/descriptor 可见性；ownership、frontier、
    cursor、published 等同步语义走 atomic primitive 和 store barrier。
26. cache 同步边界要放在共享数据交接处，不能散落到 private ring。shared map
    的同步边界是 symbol/region/heap publish 和 block.won 交接；private ring
    drain 不做 shared GM invalidate。
