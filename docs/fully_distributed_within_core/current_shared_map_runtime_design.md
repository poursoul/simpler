# FDWIC Shared Map Runtime 当前设计

本文档描述当前 `PTO_FDWIC_SHARED_MAP=1` 下 fully distributed within core runtime 的实际代码设计，用于后续开发、评审和性能分析。本文档只记录当前实现的机制和正确性约束，不替代阶段 plan，也不修改 `docs/fully_distributed_within_core.md`。

本文档的写法刻意区分三类内容：

1. **当前代码已经实现的机制**：以代码中实际存在的结构和函数为准。
2. **这些机制成立所依赖的不变量**：后续重构必须保持。
3. **已经证明不应采用的设计**：避免后续性能优化时重新踩坑。

当前 shared map 已覆盖 single AIC、single AIV、alloc、shared output ref、普通 Tensor 区域依赖、view ref 合并和 shared heap 分片。mix 路径仍不是最终高性能闭环。

---

# 第一部分 - shared map 要解决的问题

## 1. private map 模型的限制

原始 FDWIC 设计中，每个 AICore 都 replay 同一份 orchestration。第 N 次 submit 在所有核上都对应同一个 task id `N`，所有权由 claim race 决定。private map 模型为了让每个核都能本地推导依赖，要求每个核维护一份完整的 TensorMap 副本：

```text
每个核 replay task N：
  1. 对 INPUT / INOUT 查询本核 TensorMap，得到 fanin producer
  2. 对 OUTPUT / INOUT 插入本核 TensorMap，记录 producer = N
  3. 只有 claim winner 构建并执行任务
  4. loser 虽然不执行任务，但仍要维护本核 TensorMap
```

这条路径的正确性来自“每个核的 submit 序列完全相同，因此每个核的 TensorMap 内容最终相同”。但它的性能代价也很明确：

- loser 也要执行 map lookup / insert。
- 每个核都持有一份完整 map，空间和 cache 压力随核数放大。
- 输出 Tensor descriptor 通过 `TaskOutputTensors` 返回，天然倾向于“当前 submit 调用构造真实 Tensor”。
- 对 shared replay 来说，很多 loser 上的 Tensor 构造和 map 操作只是为了维持 private map 副本，不是实际执行所需。

shared map 的目标就是去掉这部分 loser 重逻辑，把依赖和输出身份提升成全局共享协议。

## 2. shared map 的基本选择

shared map 没有沿用 private map 的“每核完整复制”思路，而是做了两层拆分：

1. **输出身份共享化**：runtime-created output 用 `FdwicOutputRef` 表示。ref 只描述“哪个 task 的第几个 output”，不携带真实 Tensor descriptor。
2. **依赖状态共享化**：shared output 的最新 writer 放在 `SharedOutputCell::last_writer`；普通 Tensor 的写入区域放在全局 `SharedRegionMap`。

因此 shared map 下 loser 不需要维护一份本核 map，也不需要构造真实 output Tensor。loser 只要能够返回一个和 winner 一致的符号输出身份即可：

```text
task N 有 k 个 output：
  winner:
    materialize output Tensor
    publish SharedOutputCell[N].tensors[0..k)
    return refs (N, 0..k-1)

  loser:
    不 materialize
    不 register
    不 build slot
    return refs (N, 0..k-1)
```

consumer 使用这些 ref 构建后续 Arg。真正需要 Tensor descriptor 的时间点是 kernel 执行前，而不是 loser replay 到 submit 点时。

## 3. 为什么不能把 shared 强行合到 TaskOutputTensors

`TaskOutputTensors` 的语义是“submit 返回一组已经 materialized 的 Tensor descriptor 引用”。private map 中这些 descriptor 位于当前核的 task payload 里，生命周期和 private replay 的 slot 绑定。

shared map 中这个语义不成立：

- loser 没有 materialize output，所以 loser 没有 Tensor descriptor 可以返回。
- winner 和 loser 返回值必须在用户代码里表现为同一个 task 的输出。
- 真正 descriptor 的全局位置是 `SharedOutputCell`，它由 winner 发布，并由 consumer resolve。

如果 shared 仍返回 `TaskOutputTensors`，只有两个选择：

1. loser 伪造或复制 Tensor descriptor。这会重新引入 loser 重逻辑，也容易制造 descriptor 生命周期错误。
2. loser 等 winner 发布后再返回 Tensor。这会把 replay 串行化，破坏 submit 并行性。

所以 shared map 必须返回 `SharedTaskOutputs` / `FdwicOutputRef`。这不是类型包装差异，而是 runtime ownership 模型不同。

## 4. shared map 的总流程

完整流程如下：

```text
所有核 replay orchestration：

  tok = rt_presubmit_task(mixed)
    ├─ 生成 task id
    ├─ shared 按需 drain 本核已有工作
    └─ claim 判断当前核是否 winner

  if tok.won:
      outs = rt_submit_winner(tok, args)
        ├─ materialize OUTPUT
        ├─ publish SharedOutputCell
        ├─ register ordinary Tensor write regions
        ├─ collect fanin
        └─ build RingSlot
  else:
      outs = rt_submit_loser(tok, output_count)
        └─ 构造同 task id 的 FdwicOutputRef

后续 task 使用 outs.output_ref(i)：

  Arg.add_input(ref) / add_inout(ref) / add_output(ref)
    └─ Arg 中保存 symbolic ref

winner build consumer slot：
  ├─ shared ref 已 published：立即 resolve 成 Tensor
  └─ shared ref 未 published：保存到 RingSlot.shared_refs

execute_slot 前：
  ├─ 等 fanin producer task flag ready
  ├─ resolve 未完成 shared refs
  └─ 调用真实 kernel
```

这条路径的核心原则是：

- **claim 越早越好**：先判断 winner，再决定是否执行 submit 重逻辑。
- **resolve 越晚越好**：不在 submit 阶段等待 producer，执行 kernel 前满足即可。
- **依赖必须显式**：shared ref 依赖靠 `last_writer`，普通 Tensor 依赖靠 `SharedRegionMap`。

---

# 第二部分 - 代码结构与数据布局

## 5. 宏隔离边界

shared/private 的核心隔离点是 `PTO_FDWIC_SHARED_MAP`。

在 `state.h` 中：

- private build 保留 `DistCore::map`。
- shared build 去掉 `DistCore::map`，增加 `DistGlobal::shared_outputs` 和 `DistGlobal::shared_region`。

在 `submit_runtime.h` 中：

- private build 支持 `dist_submit_impl()` 内部自行分 winner/loser。
- shared build 禁用 `dist_submit_impl()`，要求显式 `presubmit -> winner/loser wrapper`。

在 `pto_orchestration_api.h` 中：

- private build 的 `rt_submit_task()` 返回 `TaskOutputTensors`。
- shared build 不暴露同名合一路径，而是暴露 `rt_presubmit_task()`、`rt_submit_winner()`、`rt_submit_loser()`，返回 `SharedTaskOutputs`。

这个边界非常重要。后续不能为了减少代码重复，把 shared submit 重新塞回 private 的 `TaskOutputTensors` 路径。

## 6. DistGlobal 的 shared 状态

shared build 中新增的全局状态可以分为四类：

| 状态 | 作用 | 热度 | 正确性要求 |
| ---- | ---- | ---- | ---------- |
| `shared_outputs[kFlagCap]` | task output descriptor 发布表 | 高 | published/last_writer 独占 cacheline |
| `shared_region` | 普通 Tensor 区域 writer map | 中 | bucket 发布必须 release/acquire |
| `shared_heap_cursor[kSharedHeapShards]` | output heap 分片分配 cursor | 高 | shard 内线性分配，复用受 frontier 约束 |
| `shared_heap_vend` | 全局 vend 高水位 | 中 | task complete 时可用于窗口推进 |

`kFlagCap` 是 task id 环容量。`shared_output_cell(task_id)` 使用：

```cpp
g_dist.shared_outputs[task_id & (kFlagCap - 1)]
```

因此复用正确性依赖 task id 不超过当前 flag/window 语义可以承载的范围。当前代码对 `task_id >= kFlagCap` 会 fatal，而不是 silent wrap。

## 7. Cacheline 布局约束

shared map 增加了多个跨核共享热字段。如果这些字段落在同一个 cacheline，会出现严重 false sharing，甚至导致性能波动被误判为协议问题。

当前代码用三类手段保证布局：

1. `PaddedCursor`：一个 int64 atomic 独占 64B。
2. 显式 tail pad：保证 struct size 是 64B 的整数倍。
3. `static_assert(offsetof(...) % 64 == 0)`：防止后续字段调整破坏布局。

典型例子：

```cpp
static_assert(sizeof(SharedOutputCell::published[0]) == kCacheLine);
static_assert(sizeof(SharedOutputCell::last_writer[0]) == kCacheLine);
static_assert(sizeof(SharedRegionEntry) == kCacheLine);
static_assert(sizeof(RingSlot) % kCacheLine == 0);
```

这里没有依赖 `alignas(64)` 作为 CCEC ABI 保证。padding + static_assert 才是当前代码采用的稳定约束。

---

# 第三部分 - Submit 与执行协议

## 8. task id 的含义

每个核 replay 同一份 orchestration。第 N 个 submit 点在每个核上都生成 task id `N`：

```cpp
ctx.task_id = ctx.self->local_index++;
```

task id 不由 winner 分配，也不因 loser 跳过而改变。它同时用于：

- claim cursor 的目标值。
- `SharedOutputCell` 索引。
- fanin producer id。
- task completion flag。
- shared heap/window 复用判断。

因此 shared map 的 loser 虽然不构建任务，但必须返回同一个 `task_id` 的输出 ref，否则后续 consumer 的依赖和 resolve 都会指向错误 producer。

## 9. presubmit 的职责

`dist_presubmit_task_impl()` 是 shared submit 的第一阶段。

它不能 materialize output，也不能 collect fanin。它只负责：

1. 构造最小 `DistSubmitCtx`。
2. 对明显 wrong-role 的核提前返回。
3. 按需 drain 本核已有工作。
4. 调用 `dist_submit_claim()`。
5. 返回 `SubmitToken`。

为什么 presubmit 里不能执行 args 相关逻辑：

- loser 不需要 args。
- loser 不应该读取 output create info 并构造 Tensor。
- loser 不应该写 shared region map。
- loser 不应该因为 producer 未 ready 而等待。

这就是“不走 submitbuilder/lambda”的实际落点：用户代码在拿到 `tok.won` 后，自己决定是否构造 loser 不需要的 Arg 或跳过重逻辑。

## 10. winner submit 的职责边界

`dist_submit_winner_impl()` 是唯一执行 submit 重逻辑的 kernel submit 路径。

顺序不能随意打乱：

```text
check task cap
  ↓
materialize args
  ↓
register shared regions
  ↓
collect shared fanin
  ↓
build winner task
```

这里有两个细节：

1. `materialize args` 必须早于 fanin，因为 shared output ref 的 `last_writer` 可能需要在 materialize 时初始化。
2. `register shared regions` 在 `collect shared fanin` 之前执行当前代码也能成立，因为 lookup 会要求 `entry.producer < before_task_id`，当前 task 自己插入的 entry 不会成为自己的 fanin。

如果后续改动这个顺序，需要重新检查：

- `last_writer` 初始值是否可见。
- ordinary Tensor 的 current writer 是否会被自己错误依赖。
- output descriptor 是否在 published 前已经 flush。

## 11. loser submit 的职责边界

shared build 下 loser wrapper 只做：

```text
rt_output_refs(tok.task_id, output_count)
```

它不调用 runtime submit heavy path。

这条限制非常硬：

- loser 不能 materialize。
- loser 不能 register region。
- loser 不能 collect fanin。
- loser 不能 build slot。
- loser 不能 resolve shared refs。

如果 loser 做了这些事，shared map 就会退化回 private map 的成本模型，甚至可能引入错误 producer。

## 12. alloc submit

shared alloc 也是 submit，但它不是 kernel submit，也不应该返回 `TaskOutputTensors`。

当前 shared alloc 入口是：

```cpp
int32_t dist_alloc_outputs_impl(PTO2Runtime *, const L0TaskArgs &args)
```

用户侧 wrapper：

```cpp
SharedTaskOutputs alloc_tensors(const L0TaskArgs &args)
```

流程：

1. 生成 alloc task id。
2. `dist_submit_is_alloc_candidate()` 过滤候选核。
3. 候选核执行 EfDrain。
4. claim alloc。
5. winner materialize outputs。
6. winner complete alloc task。
7. wrapper 根据 task id 和 output count 返回 refs。

alloc 没有真实 kernel 执行，所以它可以更激进地限制候选集合以降低 claim 竞争。当前实现按 lane 和 block 过滤：

```text
target_lane = task_id % kLaneCount
target_block = alloc_shard(task_id) % num_blocks
```

这不会影响普通 kernel task 的调度范围，因为它只作用于 alloc submit。

---

# 第四部分 - 依赖协议

## 13. 依赖协议总览

shared map 下 fanin 不是由 private TensorMap 推导，而是组合多种来源：

```text
fanin = explicit deps
      + shared output ref writer
      + ordinary Tensor owner_task_id
      + ordinary Tensor region overlap writer
```

`dist_submit_add_fanin()` 会去重，并限制在 `kMaxFanin` 内。

## 14. shared output ref 的生命周期

一个 runtime-created shared output 的生命周期可以写成：

```text
producer winner materialize:
  SharedOutputCell[N].tensors[slot] = Tensor descriptor
  SharedOutputCell[N].last_writer[slot] = N
  release publish SharedOutputCell[N].published[slot] = N

consumer submit:
  INPUT:
    dep = last_writer if valid else N
  INOUT / OUTPUT_EXISTING:
    dep = atomic_exchange(last_writer, current_task)
    if dep invalid, dep = N

consumer execute:
  wait fanin task flag
  wait published if descriptor still unresolved
  copy or view-copy Tensor descriptor into local RingSlot tensor
```

`last_writer` 和 `published` 分开是必要的：

- `published` 表示 producer output descriptor 是否已经可读。
- `last_writer` 表示当前逻辑最新 writer，用于依赖排序。

一个 writer 可以在 producer descriptor 发布之后改变 `last_writer`，但不能改变 producer descriptor 本身。对于 `INOUT`，writer 链保证后续读写依赖最新 writer，而底层 Tensor descriptor 仍来自最早 producer 的 output slot。

## 15. INOUT 为什么必须 exchange last_writer

假设 task A 产生 output ref `R=(A,0)`，task B 对 R 做 INOUT，task C 读取 R：

```text
A: OUTPUT R
B: INOUT R
C: INPUT R
```

正确依赖应为：

```text
B depends on A
C depends on B
```

如果 B 只读取 `last_writer` 而不更新它，那么 C 仍会依赖 A，可能与 B 并行执行，破坏 INOUT 写入顺序。

当前 `dist_shared_output_claim_writer()` 使用 atomic exchange：

```cpp
old = atomic_exchange(last_writer, current_task_id)
```

因此 B 得到旧 writer A，同时把 latest writer 改成 B。C 再读取 `last_writer` 时会依赖 B。

## 16. ordinary Tensor 的 region map

不是所有 Tensor 都来自 shared output ref。例如外部输入、用户传入的已有 Tensor、普通 output existing Tensor 都可能只表现为 Tensor descriptor。

这些 Tensor 需要通过 byte range overlap 建立依赖：

```text
lookup(tensor, before_task_id):
  bucket = hash(buffer address)
  遍历 bucket entries
  找 entry.producer < before_task_id
  且 byte range overlap
  且 producer 最大的 entry
```

注册规则：

- `INOUT` 注册。
- `OUTPUT_EXISTING` 注册。
- shared output ref 不注册。
- `manual_dep` 不注册。

查询规则：

- `INOUT` / `OUTPUT_EXISTING` 查询。
- `INPUT` 如果来自有 owner 的 Tensor，也会查询。
- `manual_dep` 不查询。

为什么 `OUTPUT` 不进入 region map：

- `OUTPUT` 是 runtime 新分配 buffer。
- 它不会与已有 Tensor 重叠。
- 它的输出身份通过 `SharedOutputCell` 发布。

为什么 `OUTPUT_EXISTING` 要进入 region map：

- 它写入已有 Tensor。
- 后续普通 Tensor consumer 需要知道这个区域的最新 producer。

## 17. manual_dep 的含义

`manual_dep` 表示用户明确要求不走自动 overlap map 依赖。shared region lookup/register 都会跳过它。

这不是性能特判，而是 API 语义：用户选择手动依赖时，runtime 不应该再从地址重叠自动补边。

---

# 第五部分 - 发布、cache 与内存序

## 18. SharedOutputCell 发布顺序

winner materialize output 时必须遵守：

```text
write Tensor descriptor
flush Tensor descriptor on CCEC
store_barrier
atomic_release published = task_id
```

consumer 必须：

```text
atomic_acquire load published
if ready:
  invalidate Tensor descriptor on CCEC
  copy Tensor descriptor
```

这个协议把 descriptor 数据和 ready flag 分开：

- descriptor 是普通 GM 数据，需要 CCEC cache flush/invalidate。
- `published` 是 atomic 同步字段，不再额外 dcci。

如果对 atomic 字段再做粗粒度 dcci，可能破坏性能，也没有必要。atomic 的 acquire/release 负责同步 flag；descriptor 的 flush/invalidate 负责普通数据可见性。

## 19. SharedRegionMap 插入顺序

region map bucket 使用 `-2` 作为插入锁 sentinel：

```text
old_head = atomic_exchange(bucket, -2, acquire)
write entry fields
store_barrier
flush entry on CCEC
atomic_exchange(bucket, new_slot, release)
```

lookup 看到 `bucket == -2` 时自旋等待。看到正常 head 后，遍历 entry 前在 CCEC 下 invalidate entry。

正确性来自：

- bucket release publish 保证 entry 字段先于 bucket head 可见。
- lookup acquire bucket 后再读取 entry。
- 插入锁只保护同 bucket 链表，不是全局锁。

当前代码里 `insert_lock` 字段存在于 `SharedRegionMap`，但实际插入路径使用 bucket sentinel，不走全局 `insert_lock`。后续可以评估是否清理这个字段，但不能在没有 ABI/初始化检查的情况下随意删除。

## 20. task completion flag

执行完成后：

```cpp
store_task_vend(task_id, self->heap_next);
store_barrier();
publish_task_flag(task_id);
```

consumer 在 `drain_phase_b()` 中 acquire 读取 producer flag。flag ready 后才能执行当前 slot。

shared build 下 `complete_executed_task()` 不调用 `advance_frontier()`，避免每个任务完成时都做全局 frontier 推进。frontier 在需要 heap 复用时由等待路径推进。

---

# 第六部分 - 性能设计

## 21. 性能目标的含义

shared map 的性能目标不是“和 private map 打平”，而是利用 winner-first 和 shared ref 减少 replay 控制开销，使 shared 在高并发场景下比 private 更快。

主要希望减少：

- loser materialize 成本。
- loser map lookup/insert 成本。
- output Tensor descriptor 的本地复制成本。
- heap 单 cursor 竞争。
- wrong-role claim 竞争。
- 空 EfDrain。

## 22. 当前 PA Case1 的性能观测口径

性能判断以上板 `merged_swimlane.json` 为准。`l2_swimlane_records.json` 可用于原始 phase 统计，但可视化上的端到端窗口应看 `merged_swimlane.json`。

最近观测：

```text
baseline:
  global span: ~2.540 ms
  EfDrain events: 73466
  Claim sum: ~58.30 ms

按需 EfDrain 后:
  global span: ~2.318 ms
  EfDrain events: 32862
  Claim sum: ~53.16 ms
```

这说明按需 EfDrain 有效，但它不是全部瓶颈。后续如果继续优化，需要优先看单核内 task 之间空白、claim 单次开销、SubmitExclusive 和 resolve/build/materialize 的局部成本。

## 23. 为什么 wrong-role skip 是通用优化

single AIC task 的合法 winner 只能是 AIC；single AIV task 的合法 winner 只能是 AIV。wrong-role 核参与 claim 不可能改变正确 winner，只会增加：

- atomic 竞争。
- trace 事件。
- replay 控制分支成本。

因此 shared presubmit 中先判断 single-lane wrong-role：

```text
if task only needs AIC and self is AIV:
  skip claim
if task only needs AIV and self is AIC:
  skip claim
```

但 wrong-role 核可能已经持有之前任务的 ring slot，或者有 block-won 投递待 drain。因此 skip 前仍要按需 drain。

这不是 PA 特判，而是由 task active mask 和 core role 推导出的通用规则。

## 24. 为什么按需 EfDrain 是通用优化

EfDrain 做两类工作：

1. `drain_block_won_if_enabled()`：把 block-won 投递转成本核 ring slot。
2. `drain_phase_b()`：执行本核 ready slot。

如果本核：

- `occupied_count == 0`
- 且没有 pending won

则 EfDrain 不可能推进任何状态。此时跳过 EfDrain 不改变：

- task flag
- ring slot 状态
- shared output published
- fanin ready
- heap frontier

所以 shared 路径可以按需执行。private 路径保持原逻辑，是因为 private 的历史语义没有在这个优化中重新审查，不应越界修改。

## 25. 为什么 alloc 可以单独限制候选

alloc submit 没有真实 kernel，不涉及 AIC/AIV 计算资源调度。它的任务是 materialize output 并发布 completion。

因此 alloc 的 winner 不需要在所有核里自由选举。当前 shared alloc 使用：

```text
target_lane = task_id % 3
target_block = alloc_shard(task_id) % num_blocks
```

只有这个 lane/block 的核参与 alloc claim。

这样减少 alloc 的原子竞争，同时不影响 kernel task 的调度范围。它是 alloc 专属优化，不应该推广到 AIC/AIV kernel claim。

## 26. 为什么没有采用 claim cell

claim cell 的方向是把每个 task 的 claim 独立成：

```text
claim_cells[task_id & mask]
```

理论上可以减少 fetch-max cursor 的竞争。但它引入 ring 复用问题：复用 cell 前必须证明上一代 task 已完成，否则新 generation 覆盖旧 generation 会丢 winner 状态。

当前最终代码没有 claim cell。原因不是它一定错误，而是尚未形成稳定收益和复用协议。当前保留 flat cursor shard，先保证正确性和已验证性能。

## 27. 为什么没有采用 tree/group cursor

tree/group cursor 曾用于降低 claim 竞争，但它破坏原 claim 语义。

错误路径：

```text
task N:
  A1 赢 leaf，但还没访问 root
  A2 输 leaf，直接继续 replay

task N+G:
  A2 赢 leaf
  A2 访问 root，把 root 推到 N+G

A1 恢复:
  A1 访问 root(N)，发现 root 已经是 N+G，于是失败

结果:
  task N 没有 root winner
```

这个问题不是最终 cursor 数值能发现的。root 最终值正确，但中间 task owner 丢失。

加 release 可以阻止 leaf loser 越序，但会改变 loser 独立推进语义，引入 per-task group lockstep。当前不采用。

---

# 第七部分 - 端到端状态机

## 28. Producer task 状态机

以一个产生 shared output 的 task N 为例：

```text
Unseen
  ↓ presubmit claim winner
ClaimedByWinner
  ↓ materialize output descriptors
OutputsMaterialized
  ↓ published[slot] = N
OutputsPublished
  ↓ build ring slot
Built
  ↓ fanin ready, execute kernel
Executed
  ↓ task flag = ready
Completed
```

loser 对这个 task 只经历：

```text
Unseen
  ↓ presubmit claim loser
NotOwner
  ↓ rt_submit_loser
ReturnedSymbolicRefs
```

loser 不参与 producer 状态推进。

## 29. Consumer task 状态机

consumer task M 使用 producer N 的 ref：

```text
ArgCreatedWithRef(N, slot)
  ↓ winner submit
CollectFanin
  ├─ INPUT: dep = last_writer(N, slot)
  └─ INOUT: dep = exchange(last_writer(N, slot), M)
  ↓ build slot
TryResolve
  ├─ published ready: Tensor copied into slot.tensors[i]
  └─ not ready: ref copied into slot.shared_refs[i]
  ↓ drain_phase_b
WaitFaninFlags
  ↓ execute_slot
ResolveRemainingRefs
  ↓ call kernel
CompleteTask
```

这里 fanin 和 resolve 是两个不同条件：

- fanin 保证 producer task 的执行顺序。
- published 保证 Tensor descriptor 可见。

一般情况下 producer complete 意味着 output 已 published，但代码仍以 `published` 作为 descriptor ready 的直接判断，避免隐式依赖。

## 30. INOUT 链状态机

对同一个 output ref 连续 INOUT：

```text
A produces R
  last_writer = A

B INOUT R
  old = exchange(last_writer, B)
  fanin += old  # A

C INOUT R
  old = exchange(last_writer, C)
  fanin += old  # B

D INPUT R
  old = load(last_writer)
  fanin += old  # C
```

因此依赖链为：

```text
A -> B -> C -> D
```

这个机制保证了即使底层 Tensor descriptor 仍是 A 发布的 output slot，逻辑 writer 也会随着 INOUT 更新。

---

# 第八部分 - 开发注意事项

## 31. 不能修改 private 语义

shared 和 private 是两条路径。除非明确是在做通用 bugfix，否则 shared 优化不应修改 private submit 的行为。

判断标准：

- 是否在 `#if PTO_FDWIC_SHARED_MAP` 内？
- 是否会改变 private 的 materialize/map/fanin/register 顺序？
- 是否会改变 private loser 行为？
- 是否会改变 `TaskOutputTensors` 生命周期？

如果答案不清楚，应先停下来重新分析。

## 32. 不要把函数名当语义隔离

shared/private 隔离应靠宏和 API contract，而不是在 runtime 内制造 `private_xxx` / `shared_xxx` 两套混杂函数名。最终代码应该让同一概念的函数在宏下有不同实现，而不是同时出现两套路径互相调用。

当前仍有一些函数名包含 shared，例如 `dist_shared_region_lookup`、`dist_resolve_shared_output_ref`。这些名字表示 shared-only 数据结构或操作，不是 submit 主路径上的 private/shared 双分支。

## 33. 不要给 shared loser 加工作

任何“为了方便返回值”而让 loser 做 materialize/resolve/copy 的修改都应视为错误方向。

shared loser 的唯一合理输出是：

```text
task id + output slot count -> FdwicOutputRef
```

## 34. 不要在 shared ref resolve 中引入等待 submit

resolve 等待只能发生在执行前或明确需要 Tensor descriptor 的位置。submit 阶段应尽量记录 ref 并继续。

如果在 winner submit 的 build 阶段发现 producer 未 published，正确行为是保存 ref 到 slot，而不是等待 producer。

## 35. 不要把普通 Tensor 和 shared ref 混成一套 map

shared ref 的 producer identity 是 task id + output slot，普通 Tensor 的 producer identity 是 byte range writer。两者的依赖来源不同：

- shared ref：`SharedOutputCell.last_writer`
- ordinary Tensor：`SharedRegionMap`

把两者强行合一会导致：

- shared output 重复 register region。
- INOUT writer 链和 byte overlap writer 链互相覆盖。
- view ref 无法保持符号语义。

## 36. 上板验证要求

后续修改 shared map 时，至少需要：

1. 清理 a5 sim/onboard 编译缓存后重编。
2. PA Case1 sim 功能通过，平时不带 `--use-example-exec-time`。
3. PA Case1 上板 `--device 6` 通过。
4. shared smoke 全量上板通过，使用 `--manual include`。
5. 性能分析使用上板 `merged_swimlane.json`。

只有在做 PA sim 性能判断时才使用 `--use-example-exec-time`。上板不使用该参数。

---

# 第九部分 - 按 commit 梳理当前设计的形成过程

本节按分支上的 commit 演进解释当前 shared map 为什么形成现在的结构。这里不是流水账，而是把每个阶段“解决了什么问题、留下了什么机制、修正了什么错误方向”对应到最终代码。

## 37. 基础 FDWIC runtime 稳定阶段

在 shared map 之前，分支先完成了一组 FDWIC 基础 runtime commit：

```text
4b039d33 Refactor: unify fdwic aicore core entry
50946ffa Update: converge fdwic slot completion into real submit
bf5e3285 Fix: pass PTO ISA includes to aicore-extra builds
62c19d6a Support: add fdwic AICore swimlane tracing
2679fb3d Fix: align fdwic block.won drain semantics
1867fb22 Fix: align fdwic atomic submit primitives
3b530606 Refactor: align fdwic submit runtime naming
12fde60b Add: cover fdwic submit ring reuse stress
b8dc1520 Add: extend fdwic submit stress coverage
3ef1fff0 Support: expand fdwic submit swimlane phases
1387a89b Fix: isolate fdwic atomic cachelines
824aff65 Support: reduce fdwic swimlane collection overhead
35921a85 Support: reduce fdwic payload flush range
73fe27b0 Support: persist only fdwic output descriptors
5189c886 Support: reduce fdwic output layout initialization
fc2cf814 Support: simplify fdwic output layout offsets
94712c38 Support: skip fdwic output tensor map registration
bd620630 Support: reduce fdwic private slot cache flushes
```

这些 commit 的目标不是 shared map 本身，而是让 FDWIC 的基础 submit/execution 模型具备可扩展前提：

1. AICore entry、slot completion、block-won drain、atomic primitive 先收敛。
2. ring reuse stress、submit stress 和 swimlane phase 先建立可验证性。
3. payload flush、output layout、private slot cache flush 做了基础性能瘦身。
4. cacheline 隔离在 shared map 前先修掉，否则后续 shared atomic 热点会被 false sharing 放大。

对当前 shared map 来说，这一阶段留下的关键基础是：

- `DistCore::slots` 作为每核私有执行 ring。
- `DistTaskCell` 作为全局 task completion flag/vend。
- `BlockWon` / `WonSlot` 作为 mix 的 anchor-to-follower 投递机制。
- `TracePhase` 和 swimlane 记录作为性能分析基础。
- `PaddedCursor` 和 64B padding 的布局风格。

也就是说，shared map 后续是在这个 FDWIC runtime 骨架上替换“输出身份和依赖发现”机制，而不是重新写执行循环。

## 38. `ee605410` - winner-first shared submit path

commit：

```text
ee605410 Add: fdwic winner-first shared submit path
```

commit 描述：

```text
- Introduce presubmit winner/loser submit APIs with macro-isolated paths
- Implement shared symbolic output publication and resolution
- Keep orchestration call sites path-neutral while shared losers skip arg construction
- Thread CXXFLAGS into sim/direct and onboard AICore compile paths
```

这一阶段解决的是 shared map 最关键的路径选择问题：**submit 必须拆成 presubmit 和 winner/loser 两段**。

引入的机制：

1. `SubmitToken`：
   - 保存 task id、kernel id、active mask、candidate、won 等 presubmit 结果。
   - 让用户侧可以在 submit 前判断 winner。

2. `dist_presubmit_task_impl()`：
   - 只做 task id 分配、drain、claim。
   - 不读取完整 args，不 materialize output。

3. `rt_submit_winner()` / `rt_submit_loser()`：
   - shared build 下用户代码显式区分 winner 和 loser。
   - loser 不进入 submit heavy path。

4. `SharedTaskOutputs` / `FdwicOutputRef` 的雏形：
   - loser 返回符号输出，而不是 Tensor descriptor。

5. shared 输出 publication / resolution 的初始版本：
   - 让 producer winner 可以发布 output descriptor。
   - consumer 可以通过 symbolic ref 解析。

这个 commit 确立了后续所有 shared map 优化的前提：**先选 winner，再决定是否执行重逻辑**。如果没有这一步，后续 heap shard、region map、resolve 延迟都只能变成 private path 上的小修小补。

这一阶段也修正了一个设计方向：shared submit 不应该靠 lambda 或 submit builder 延迟 loser 逻辑。正确做法是在 submit 前通过 `SubmitToken` 暴露 winner 结果，让用户代码直接跳过 loser 不需要构造的 Arg。

## 39. `a25d5bec` - shared symbol submit phase

commit：

```text
a25d5bec Add: fdwic shared symbol submit phase
```

commit 描述：

```text
- Add shared-map output refs and macro-isolated winner/loser submit APIs.
- Resolve shared symbolic inputs inside winner materialization without exposing a public tensor resolve API.
- Add shared symbol smoke coverage and restore submit dependency smoke to its private-map role.
```

这一阶段把“symbolic output”从初始能力变成可测试的 submit ABI。

主要变化：

1. `FdwicOutputRef` 成为 `Arg` 可接收的 tensor ref 类型。
2. `Arg::add_input(FdwicOutputRef)`、`add_output(FdwicOutputRef)`、`add_inout(FdwicOutputRef)` 进入 shared build。
3. shared submit API 在 orchestration wrapper 层宏隔离。
4. shared symbol smoke 用例覆盖 producer/consumer 基本链路。
5. 历史 submit dependency smoke 回到 private-map 角色，避免在功能还没完成时把旧用例改成不符合 phase 设计的 shared 适配。

这一阶段形成的设计原则：

- shared ref 是用户侧可传递的符号句柄。
- public API 不暴露“手动 resolve Tensor”的接口。
- resolve 是 runtime 内部职责，不能让用户代码拿到半成品 Tensor descriptor。

这也是后续“shared 不返回 TaskOutputTensors”的直接来源。

## 40. `69575a4a` - shared symbol heap phase

commit：

```text
69575a4a Add: fdwic shared symbol heap phase
```

commit 描述：

```text
- Add shared descriptor publication with per-slot ready bits and sharded heap allocation
- Keep shared losers on symbolic outputs while private submit remains separate
- Cover shared symbol multi-output and shard cases, plus PA shared submit shape
- Record Phase 4 closure, validation, and next gates in the implementation plan
```

这一阶段把 output ref 和真实 output buffer 生命周期接起来。

保留下来的机制：

1. `SharedOutputCell::published[]`
   - 每个 output slot 一个 ready flag。
   - 值等于 producer task id 才表示该 generation 的 output ready。

2. `SharedOutputCell::tensors[]`
   - winner 发布 Tensor descriptor 的全局位置。

3. shared heap shard：
   - `shared_heap_cursor[kSharedHeapShards]`
   - `shared_heap_vend`
   - task 按 shard 分配 output buffer。

4. 多 output 支持：
   - `SharedTaskOutputs` 可按 output ordinal 生成 ref。
   - `kSharedOutputMaxPerTask` 定义每 task shared output 上限。

正确性闭环：

```text
winner materialize descriptor
  ↓
flush descriptor
  ↓
release publish published[slot] = task_id
  ↓
consumer acquire published
  ↓
invalidate descriptor and copy
```

这一阶段还明确 private submit 保持独立，不因 shared heap 引入修改。

## 41. `1bcb7169` - shared map phase updates

commit：

```text
1bcb7169 Add: fdwic shared map phase updates
```

commit 描述：

```text
- Add shared output refs, descriptor publication, shared heap and region state for the macro-isolated shared submit path
- Update smoke and PA orchestration to use the shared output ABI without changing private map semantics
- Keep the shared map plan in Chinese and record the phase 6 joint follower handoff draft
```

这一阶段把 shared map 从“shared output symbol”扩展为“shared dependency map”。

主要机制：

1. `SharedRegionMap`
   - 用于普通 Tensor 的全局 byte range writer 记录。
   - 替代 shared build 下的 per-core `DistTensorMap`。

2. `SharedRegionEntry`
   - 记录 `buf_addr`、`lo`、`hi`、`producer`、bucket next。

3. `dist_shared_region_lookup()`
   - 根据 ordinary Tensor 的 byte range 找最近 overlap producer。

4. `dist_shared_region_insert()`
   - 对写入类 ordinary Tensor 注册 writer。

5. PA orchestration 开始使用 shared output ABI。

这一阶段确立了当前 shared map 的两条依赖路径：

```text
shared output ref:
  SharedOutputCell.last_writer

ordinary Tensor:
  SharedRegionMap byte-range overlap
```

这两条路径不能合并，因为 shared ref 的身份是 task/output slot，ordinary Tensor 的身份是地址区间。

## 42. `f0366b08` - optimize shared submit refs

commit：

```text
f0366b08 WIP: optimize fdwic shared submit refs
```

这一阶段主要围绕 shared ref 表示和 view 场景做瘦身。

最终保留下来的设计点：

1. `FdwicOutputRef` 携带轻量 view 信息：
   - `flags`
   - `view_ndims`
   - `view_shape0`
   - `view_offset0`

2. 连续 view 在 `FdwicOutputRef::view()` 中合并：
   - 已经是 view 时累加 offset。
   - 更新 shape。

3. resolve 时直接根据 ref 生成 Tensor view：

```cpp
Tensor::copy(resolved, Tensor::view(cell.tensors[ref.output_slot], view_shapes, view_offsets));
```

为什么这样对：

- view 是用户侧操作，用户执行 view 时应该立刻得到新的 ref，而不是在 runtime 内记录一条 view 操作链。
- resolve 阶段只需要处理最终合并后的 view 参数。
- 避免连续 view 造成 runtime 递归追踪。

这一阶段也明确了一个边界：shared resolve 不应该通过额外本地临时 Tensor 强行转成 private 风格返回；应该直接把解析结果写入目标 slot Tensor。

## 43. `ec02a7cd` - shared resolve trace phases

commit：

```text
ec02a7cd Support: trace fdwic shared resolve phases
```

commit 描述：

```text
- Add swimlane phases for shared output resolve wait, invalidate, and copy
- Split submit claim/build attribution so PA traces show per-core bottlenecks
- Teach host export and converter to name the new fdwic trace phases
```

这一阶段不是功能语义变化，而是让性能问题可观察。

新增/细化的 phase：

- `ResolveWait`
- `ResolveInvalidate`
- `ResolveCopy`
- 更清晰的 `Claim`
- 更清晰的 `Build`

为什么重要：

shared map 的性能瓶颈经常不是某个大 kernel，而是单核内 submit/replay 控制路径中的微小空白累积。没有这些 phase，会把 resolve wait、dcci/invalidate、copy 和 claim 混在一起，无法判断优化方向。

后续多次性能分析都依赖这组 trace：

- 判断 resolve 是否过早等待。
- 判断 AIC/AIV 单核空白是否来自 claim 还是 efdrain。
- 判断 `merged_swimlane.json` 中真实端到端窗口。

## 44. `ad37df57` - optimize shared submit resolution

commit：

```text
ad37df57 Update: optimize fdwic shared submit resolution
```

commit 描述：

```text
- Keep shared output refs in submit slots until execution when the producer is not yet published
- Resolve ready shared refs directly into ring slot tensors to avoid extra local Tensor copies
- Avoid block-won draining on paths that have not enabled joint submit work
- Remove the claim hot-path fatal load that inflated PA submit latency
```

这是 shared map 性能路径中非常关键的一次修正。

保留下来的机制：

1. 未 ready 的 shared ref 保存在 `RingSlot.shared_refs`。
2. `RingSlot.shared_ref_mask` 标记哪些 tensor 参数还需要执行前 resolve。
3. `execute_slot()` 前调用 `dist_resolve_slot_shared_refs()`。
4. ready ref 直接 resolve 到 ring slot tensor，避免额外本地 Tensor。
5. 非 joint 路径避免无意义 block-won drain。
6. claim hot path 去掉额外 fatal load。

设计意义：

```text
submit 阶段:
  producer ready -> resolve
  producer not ready -> record ref and continue

execute 阶段:
  fanin ready 后 resolve remaining refs
```

这比“submit 阶段 wait producer published”更好，因为 submit/replay 可以继续并行推进。只有 kernel 执行前才必须拿到真实 Tensor descriptor。

这一阶段也修正了一个错误方向：不要为了统一路径在 resolve 中制造额外 copy 或返回 `TaskOutputTensors` 风格对象。

## 45. `ae0a2e19` - reduce shared submit contention

commit：

```text
ae0a2e19 Update: reduce fdwic shared submit contention
```

commit 描述：

```text
- Split submit claim cursor shards by lane type to reduce AIV contention
- Balance alloc claims across lanes with per-lane cursor shards
- Keep shared commit off the global frontier hot path; wait paths still advance frontier for reuse
```

这一阶段开始针对 PA 的 submit 控制开销做通用优化。

保留下来的机制：

1. `cube_cursor[kCubeCursorShards]`
2. `vector_cursor[kVectorCursorShards]`
3. `alloc_cursor[kLaneCount][kAllocCursorShards]`
4. shared complete 不在每次 task complete 时推进 global frontier。

为什么正确：

- cursor shard 只把不同 task 分散到不同 atomic cell。
- 同一个 task 仍由 `task_id & shard_mask` 定位到唯一权威 cursor。
- 没有二级 claim，没有 leaf/root，不会漏 owner。
- frontier 推进从 complete hot path 移到 wait/reuse 路径，减少每 task commit 的全局 atomic 压力；需要 heap 复用时仍会推进 frontier。

这一阶段需要特别区分“cursor sharding”和“tree/group cursor”：

- cursor sharding 是当前保留设计。
- tree/group cursor 后来证明有漏 owner 风险，已经移除。

## 46. `edbf7f6a` - reduce shared alloc submit contention

commit：

```text
edbf7f6a Update: reduce shared alloc submit contention
```

commit 描述：

```text
- Filter shared alloc submits before efdrain and claim so only one lane/block candidate enters the hot path per task.
- Keep alloc candidate selection aligned with the alloc cursor shard and preserve small block_dim correctness.
- Use fetch_max for shared output last_writer initialization so runahead writers are not overwritten by late producer materialization.
```

这一阶段有两个独立但都很重要的修正。

### alloc candidate 过滤

alloc 是轻量 submit，没有真实 kernel 调度需求。如果所有核都参与 alloc claim，原子竞争成本会超过 alloc 本身。

当前保留：

```text
target_lane = task_id % kLaneCount
target_block = alloc_shard(task_id) % g_dist.num_blocks
```

只有匹配 lane/block 的核进入 alloc hot path。

正确性理由：

- alloc 只需要一个 winner 发布 output 和完成 task。
- alloc 不涉及 AIC/AIV kernel 执行资源选择。
- 对 small block_dim，通过 `target_block % num_blocks` 保持候选存在。

### last_writer 初始化改 fetch_max

producer materialize output 时会初始化：

```cpp
atomic_fetch_max(last_writer, producer_task_id)
```

而不是无条件 exchange。

原因是 runahead 场景中可能出现：

```text
consumer INOUT 先到:
  exchange(last_writer, consumer_task_id)

producer materialize 后到:
  如果 exchange(last_writer, producer_task_id)
    会把 last_writer 从 consumer 覆盖回 producer
    后续读者错误依赖 producer，而不是 consumer
```

用 `fetch_max` 后，producer 的初始化不会覆盖更大的 writer task id。这是 shared INOUT 正确性的关键点之一。

## 47. `22e37869` - skip shared wrong-role kernel claims

commit：

```text
22e37869 Update: skip shared wrong-role kernel claims
```

commit 描述：

```text
- Keep single-lane shared submit from claiming on impossible core roles
- Preserve flat cursor claim ordering for all eligible workers
- Remove unsafe grouped cursor election that could skip task owners
```

这一阶段有一个正向优化和一个重要回撤。

### 正向优化：wrong-role skip

single AIC task 上 AIV 不可能赢，single AIV task 上 AIC 不可能赢。因此这些核可以不进入 claim。

但 skip 前必须检查 drain work：

```text
if wrong_role:
  if has drain work:
    drain
  return token
```

这样不会阻塞本核之前已经持有的工作。

### 回撤：unsafe grouped cursor election

曾经尝试 group/tree cursor 降低 claim 竞争。但它被证明可能让 task 没有 owner。

最终 commit 明确移除了该方向，并保留 flat cursor claim ordering。

这一点必须记录在设计文档中，因为后续性能优化很容易再次想到 tree claim。除非引入完整 release/barrier 协议并重新证明语义，否则不能恢复。

## 48. `f0805edf` - shared submit 按需 EfDrain

commit：

```text
f0805edf Optimize shared submit efdrain
```

这一阶段优化 shared submit 中大量空 EfDrain。

新增：

```cpp
bool dist_submit_has_drain_work(DistCore *self) {
    if (self == nullptr) return false;
    if (self->occupied_count != 0) return true;
    return has_pending_won(self);
}
```

shared presubmit 中：

```text
if has drain work:
  drain_block_won_if_enabled
  drain_phase_b
```

private submit 保持原无条件 drain。

性能结果：

```text
PA Case1 baseline:
  merged_swimlane global span ~2.540 ms
  EfDrain events 73466

按需 EfDrain 后:
  merged_swimlane global span ~2.318 ms
  EfDrain events 32862
```

正确性理由：

- 没有 occupied slot 且没有 pending won 时，drain 不可能推进状态。
- 有 pending won 或 occupied slot 时仍 drain。
- 该优化只在 shared 宏下启用，不改 private 语义。

## 49. 当前最终机制与 commit 的对应关系

| 最终机制 | 主要来源 commit | 后续修正 |
| -------- | --------------- | -------- |
| presubmit / winner / loser 两段 submit | `ee605410` | `a25d5bec` 强化 shared ABI |
| shared symbolic output ref | `ee605410`, `a25d5bec` | `f0366b08` 增加 view ref |
| `SharedOutputCell` descriptor publish | `69575a4a` | `edbf7f6a` 修正 last_writer 初始化 |
| shared heap shard | `69575a4a` | `ae0a2e19` 避免 complete hot path frontier |
| `SharedRegionMap` | `1bcb7169` | 当前仍保留 bucket sentinel 插入 |
| 延迟 resolve 到 execute 前 | `ad37df57` | `ec02a7cd` 增加 trace 可观测性 |
| cursor shard 降 claim 竞争 | `ae0a2e19` | `22e37869` 移除 unsafe grouped cursor |
| alloc candidate 过滤 | `edbf7f6a` | 当前仅用于 shared alloc |
| wrong-role skip | `22e37869` | `f0805edf` 复用按需 drain helper |
| 按需 EfDrain | `f0805edf` | 当前只作用 shared presubmit |

## 50. 当前代码中不应再回退的结论

从这些 commit 的演进看，当前设计有几个已经被反复验证的方向：

1. shared 和 private 必须是宏隔离的两条 submit 路径。
2. shared submit 必须 winner-first。
3. shared output 必须是 symbolic ref，而不是 `TaskOutputTensors`。
4. loser 不应执行 materialize/register/fanin/build/resolve。
5. resolve 未 ready 时应记录 ref，而不是 submit 阶段等待。
6. ordinary Tensor overlap 和 shared output ref writer 是两套依赖机制。
7. cursor shard 可以保留，tree/group cursor 不能无 release 协议地恢复。
8. shared alloc 可以限制候选，kernel task 不能为了性能随意限制调度范围。
9. atomic 字段不要再叠加不必要 dcci；descriptor 数据才需要 flush/invalidate。
10. 每个阶段完成后要补功能验证，不能先加不可运行用例再说未来实现。


## 设计目标

shared map 的目标不是把 private map 小改成共享版本，而是走一条独立 submit 路径：

1. submit 前先选 winner。
2. 只有 winner 执行 materialize、register、fanin、build 等重逻辑。
3. loser 不执行 loser 不需要的逻辑，只返回同 task id 的符号输出。
4. shared 输出使用 `FdwicOutputRef` 表示，不使用 `TaskOutputTensors` 暴露跨核 Tensor descriptor。
5. private/shared 通过 `PTO_FDWIC_SHARED_MAP` 宏隔离，避免两条路径在语义上强行合一。

## 核心数据结构

### DistCore

shared build 下 `DistCore` 不再包含 private `DistTensorMap`：

- private build：每个 core 内有 `DistCore::map`，用于本核 producer map。
- shared build：map 从 `DistCore` 移除，依赖关系由全局 shared output 表和 shared region map 维护。

每个 core 仍保留：

- `slots[kPrivateSlots]`：本核待执行 ring slot。
- `task_payloads[kTaskPayloadSlots]`：submit 期间 materialize 输出和参数的 payload。
- `occupied_count`：本核 ring slot 占用数量。
- `local_index`：本核 replay orchestration 时生成的 task id。

shared build 下 `kPrivateSlots` 增大到 14，用于承载更多 in-core replay 并发和 block-won drain 压力。

### SharedOutputCell

每个 task id 对应一个 shared output cell：

```cpp
struct SharedOutputCell {
    PaddedCursor published[kSharedOutputMaxPerTask];
    PaddedCursor last_writer[kSharedOutputMaxPerTask];
    Tensor tensors[kSharedOutputMaxPerTask];
};
```

含义：

- `tensors[slot]`：winner materialize 后发布的 Tensor descriptor。
- `published[slot]`：该 output slot 是否已经由 producer task 发布。值等于 `producer_task_id` 才表示 ready。
- `last_writer[slot]`：该 shared output 当前最新 writer task，用于 `INOUT` / `OUTPUT_EXISTING` 建立写后写依赖链。

`published` 和 `last_writer` 的每个元素都是独占 cacheline 的 `PaddedCursor`。这是必要的，因为这些字段是多核高频 atomic 读写字段，不能和其他热字段共享 cacheline。

### SharedTaskOutputs 和 FdwicOutputRef

shared submit 返回 `SharedTaskOutputs`，其中只包含 task id 和 output count。用户通过 `output_ref(index)` 得到：

```cpp
struct FdwicOutputRef {
    int32_t producer_task_id;
    int16_t output_slot;
    uint8_t flags;
    uint8_t view_ndims;
    uint32_t view_shape0;
    uint32_t view_offset0;
};
```

`FdwicOutputRef` 是符号输出引用，不是 Tensor descriptor。它可以被后续 `Arg::add_input/add_output/add_inout` 接收。这样 loser 可以构造输出引用，但不需要持有或复制真实 Tensor。

### SharedRegionMap

普通 Tensor 的跨核重叠依赖由全局 `SharedRegionMap` 维护：

```cpp
struct SharedRegionMap {
    PaddedCursor high_water;
    PaddedCursor insert_lock;
    PaddedCursor buckets[kSharedRegionBuckets];
    SharedRegionEntry entries[kSharedRegionCap];
};
```

它记录普通 Tensor 写入区域：

- `buf_addr`
- byte range `[lo, hi)`
- `producer`
- bucket 链表 next

shared output ref 不进入 region map，因为它有专门的 `SharedOutputCell` 和 `last_writer` 机制。region map 用于外部 Tensor 或普通 Tensor 引用的重叠检查。

## Submit 路径

### presubmit

用户侧通过 `rt_presubmit_task()` / `rt_presubmit_aic_task()` / `rt_presubmit_aiv_task()` 进入：

```cpp
SubmitToken tok = dist_presubmit_task_impl(rt, mixed);
```

`dist_presubmit_task_impl()` 做三件事：

1. 生成当前 replay task id。
2. shared 路径按需执行 EfDrain。
3. 执行 `dist_submit_claim()` 判断当前核是否 winner。

`SubmitToken` 记录：

- `task_id`
- `kernel_id`
- `active_mask`
- `candidate`
- `won`
- `joint`
- `joint_block`
- `joint_count`
- `mixed`

shared build 下，`dist_submit_impl()` 被禁止：

```cpp
always_assert(false && "shared-map submit requires explicit winner/loser submit");
```

这保证 shared 不能退回 private 的“一次 submit 内部自己分 winner/loser”路径。

### winner submit

shared build 下 `rt_submit_winner(tok, args)` 调：

```cpp
void dist_submit_winner_impl(PTO2Runtime *, const SubmitToken &tok, const L0TaskArgs &args);
```

winner 路径顺序：

1. `dist_submit_check_task_cap()`
2. `dist_submit_materialize_args()`
3. `dist_submit_register_shared_regions()`
4. `dist_submit_collect_shared_fanin()`
5. `dist_submit_build_winner_task()`

只有 winner 执行这些逻辑。loser 不 materialize，不 register，不 collect fanin，不 build slot。

### loser submit

shared build 下没有 runtime 内部的 `dist_submit_loser_impl()`。用户侧 wrapper：

```cpp
SharedTaskOutputs rt_submit_loser(const SubmitToken &tok, uint32_t output_count)
```

只通过 `task_id + output_count` 构造 `SharedTaskOutputs`。这是正确的，因为 shared output 的真实 Tensor descriptor 由 winner 发布到 `SharedOutputCell`，loser 不应该构造任何本地 Tensor。

## Claim 设计

当前 claim 使用 flat authoritative cursor shard，不使用树形 claim。

### Kernel claim

single AIC：

- 只有 AIC role 可以参与。
- 使用 `g_dist.cube_cursor[task_id & kCubeCursorShardMask]`。

single AIV：

- 只有 AIV role 可以参与。
- 使用 `g_dist.vector_cursor[task_id & kVectorCursorShardMask]`。

mix：

- 当前仍使用 anchor lane 进入 claim。
- `g_fdwic_block_won_enabled = true`。
- winner 负责向同 block 其他 lane deposit built subtask。
- mix 不是本文档认为的最终高性能闭环。

alloc：

- 目标 lane 为 `task_id % kLaneCount`。
- shard 为 `task_id & kAllocCursorShardMask`。
- shared alloc 额外通过 `dist_submit_is_alloc_candidate()` 限定具体 block，避免所有同 lane core 都参与轻量 alloc 的 claim。

### 为什么 cursor shard 是正确的

每个 task 仍只有一个权威 claim 点：

```cpp
claim(cursors[task_id & shard_mask].v, task_id)
```

shard 只减少不同 task 之间对同一 atomic cacheline 的竞争，不改变同一个 task 的 winner 选举语义。所有合法候选核对同一个 task 仍竞争同一个 cursor cell。

当前实现没有使用树形/group cursor。树形/group cursor 的问题是 leaf loser 可能跳过 root claim，使后续 task 推高 root cursor，导致中间 task 没有 owner。这个设计不在最终代码内。

## Shared 输出发布与解析

### 发布

`dist_submit_materialize_args()` 对 `OUTPUT` 执行：

1. 从 shared heap 分配 output buffer。
2. 构造 materialized Tensor。
3. 写入本 task payload。
4. 写入 `SharedOutputCell::tensors[output_ordinal]`。
5. CCEC 下 flush Tensor descriptor。
6. 更新 `last_writer[output_ordinal]`。
7. 在所有 output descriptor 写完后 `store_barrier()`。
8. release 发布 `published[output_ordinal] = task_id`。

发布顺序保证 consumer acquire 看到 `published == producer_task_id` 后，能够读取到对应 Tensor descriptor。

### 解析

consumer 对 shared ref 有两种解析时机：

1. build ring slot 时尝试 `dist_try_resolve_shared_output_ref()`。
2. 如果 producer 尚未 published，则把 `FdwicOutputRef` 存入 `RingSlot.shared_refs`，在 `execute_slot()` 前调用 `dist_resolve_slot_shared_refs()` 解析。

这种设计避免 submit 阶段因为 producer 未发布而过早等待。只有 kernel 真正执行前必须拿到 Tensor descriptor，这样可以让 replay 和其他可并行 submit 继续推进。

### view 合并

`FdwicOutputRef::view()` 在用户侧执行 view 时立即合并符号 view 信息。当前支持一维 view：

- 如果 ref 已是 view，则 offset 累加。
- shape 更新为新的 view shape。
- resolve 时对 published Tensor 执行 `Tensor::view()` 后 copy 到目标 Tensor descriptor。

这避免在 runtime 中记录 view 链，也避免连续 view 导致 resolve 阶段递归追溯。

## 依赖收集

shared build 使用 `dist_submit_collect_shared_fanin()`。

依赖来源包括：

1. 显式依赖 `args.explicit_dep()`。
2. shared output ref。
3. 普通 Tensor 的 `owner_task_id`。
4. 普通 Tensor 与 shared region map 中已有 writer 的 byte-range overlap。

### Shared output ref 依赖

如果参数来自 `FdwicOutputRef`：

- `INPUT`：读取 `last_writer`。如果没有 writer，则依赖原 producer。
- `INOUT` / `OUTPUT_EXISTING`：`atomic_exchange(last_writer, current_task_id)`，返回旧 writer，并依赖旧 writer。

正确性理由：

- `INPUT` 必须读到最新 writer，因此依赖当前 `last_writer`。
- `INOUT` / `OUTPUT_EXISTING` 会成为新 writer，必须依赖旧 writer，避免写后写或读写乱序。
- 如果 `last_writer` 还未有效，则退回 producer task，保证生命周期至少依赖最早创建者。

### 普通 Tensor 区域依赖

对普通 Tensor：

- `INPUT` 如果有 `owner_task_id`，先依赖 owner。
- `INPUT` 且 owner 有效时，也可能通过 shared region map 找到最近 overlap writer。
- `INOUT` / `OUTPUT_EXISTING` 会查 shared region map，并在 register 阶段插入自己的写入区域。
- `manual_dep` 跳过自动 region map 依赖。

正确性理由：

- 普通 Tensor 不是 shared ref，不能用 `SharedOutputCell` 判断 writer。
- 跨核 replay 时 private per-core map 不完整，所以必须使用全局 shared region map。
- 只注册写入类参数，避免把纯读误认为 producer。

## Shared heap

shared build 下 output heap 使用 shard cursor：

```cpp
shared_heap_cursor[kSharedHeapShards]
shared_heap_vend
```

分配流程：

1. 根据 `task_id % kSharedHeapActiveShards` 选择 heap shard。
2. 在 shard 内 `atomic_fetch_add` 预留空间。
3. 如果 shard wrap 或复用窗口压力过大，等待 `frontier` 推进。
4. 用 `shared_heap_vend` 维护全局 vend，用于完成路径和窗口推进。

正确性理由：

- shard 减少 output allocation 对单一 cursor 的争用。
- 每个 task 的输出仍来自唯一 winner 的 materialize。
- heap 复用仍受 `frontier` / `H` 窗口约束，不会覆盖仍可能被引用的数据。

## Ring slot 与执行

winner build 后把任务放入本核 `RingSlot`：

1. copy tensor/scalar args。
2. 对 shared ref 尝试立即 resolve。
3. 未 ready 的 shared ref 保存到 `shared_refs`。
4. 写 fanin。
5. flush slot payload。
6. publish `built = true`。

`drain_phase_b()` 执行时：

1. 遍历本核 occupied + built slots。
2. 检查所有 fanin task flag 是否 ready。
3. shared build 下先 resolve slot 中未完成的 shared refs。
4. 调用真实 kernel。
5. `complete_executed_task()` 发布 task flag。

这保证 kernel 入口拿到的是完整 Tensor descriptor，同时 fanin producer 已完成。

## Block-won 机制

mix task 需要一个 winner 构建多个 lane 的 subtask，并让其他 lane drain：

- winner 申请 `WonSlot`。
- winner 填充 `BuiltSubtask lane[]`。
- flush metadata 和 lane payload。
- 发布 `remaining` 和 `state`。
- 设置 block 的 `any_pub`。
- 其他 lane 通过 `drain_block_won()` 拿到自己的 lane subtask 并转换成本地 ring slot。

当前 shared map 的 single-lane 路径不依赖 mix block-won。mix 后续还需要单独优化和闭环。

## 已做性能优化及正确性理由

### Winner-first submit

优化内容：

- claim 在 submit 重逻辑之前发生。
- loser 不 materialize、不 register、不 build。

正确性：

- task id 在 presubmit 已确定。
- winner 是唯一负责发布真实输出 descriptor 和任务执行 slot 的核。
- loser 只需要返回同 task id 的符号输出，后续 consumer 会等 winner 发布。

### Shared ref 替代 TaskOutputTensors

优化内容：

- shared submit 返回 `SharedTaskOutputs`。
- 输出通过 `FdwicOutputRef` 传递。

正确性：

- `TaskOutputTensors` 的语义是借用某个 payload 中的 Tensor descriptor，适合 private 本核路径。
- shared 路径 descriptor 属于全局 `SharedOutputCell`，并且 winner/loser 不是同一个执行路径。
- 用符号 ref 可以让 loser 也返回正确输出身份，而不伪造 Tensor。

### 延迟 resolve

优化内容：

- producer ready 时 build 阶段直接 resolve。
- producer 未 ready 时记录 ref，执行前再 resolve。

正确性：

- fanin 保证 kernel 执行前 producer task complete。
- published flag 保证 Tensor descriptor 已发布。
- 未 ready 时不阻塞 submit，增加 replay 并行度。

### Shared heap shard

优化内容：

- 多 shard 分配 output buffer。

正确性：

- shard 只影响地址分配位置，不影响 task 所有权和依赖。
- 每个 shard 内仍用 atomic cursor 线性分配。
- 复用窗口等待 `frontier`，不绕过生命周期。

### Cursor shard

优化内容：

- cube/vector/alloc 分别使用多个 cursor shard。
- vector shard 数大于 cube，用于降低 AIV-only claim 压力。

正确性：

- 同一个 task 的所有候选核仍竞争同一个 shard cell。
- shard 不形成多级 claim，也不让 loser 跳过权威 claim。
- 不存在 tree cursor 的漏 owner 问题。

### Wrong-role early skip

优化内容：

- AIC-only 任务上 AIV 不 claim。
- AIV-only 任务上 AIC 不 claim。
- wrong-role 如果本核有 drain work，仍先 drain。

正确性：

- wrong-role 核不可能成为该 single-lane task 的 winner。
- 跳过 claim 不影响 winner 集合。
- 保留按需 drain 避免本核已有 ring/won 工作无法推进。

### 按需 EfDrain

优化内容：

- shared submit 只有本核存在 `occupied_count != 0` 或 pending won 时执行 EfDrain。
- private submit 不改，仍保持原有无条件 drain。

正确性：

- 没有 occupied slot 且没有 pending won 时，drain 没有可推进对象。
- 跳过空 drain 不改变 task flag、fanin、slot 执行顺序。
- shared/private 宏隔离，不修改 private 语义。

### Atomic cacheline 隔离

优化内容：

- 高频 atomic 字段使用 `PaddedCursor` 或显式 pad 到 64B。
- `SharedOutputCell`、`SharedRegionEntry`、`WonSlot`、`RingSlot`、`BuiltSubtask` 都有 static_assert 检查 cacheline 布局。

正确性：

- padding 不改变协议语义。
- static_assert 防止后续字段调整破坏 cacheline 隔离。
- 避免依赖 CCEC 上不稳定的 `alignas(64)` ABI 行为。

## 当前没有采用的设计

### SubmitBuilder / lambda

当前代码没有使用 submit builder 或 lambda 延迟执行 loser 逻辑。shared 路径要求 submit 前先判断 winner，再由用户侧显式调用 winner/loser wrapper。

### Tree/group cursor claim

当前代码没有 tree/group cursor claim。该设计有漏 owner 风险：

1. leaf winner 尚未执行 root claim。
2. leaf loser 继续 replay 到后续 task。
3. 后续 task 推高 root cursor。
4. 前一个 task 的 root claim 全部失败，导致没有最终 winner。

当前 cursor shard 不是 tree claim；每个 task 仍只有一个权威 cursor。

### fanout 通知式 ready check

当前代码没有采用 task 完成时主动更新 consumer pending count 的 fanout 方案。`drain_phase_b()` 仍按 slot fanin 扫描 producer task flag。

## 当前限制

1. mix task 不是最终高性能闭环。
2. `FdwicOutputRef` view 当前只支持一维 view。
3. shared region map 插入使用 bucket lock sentinel，后续如果普通 Tensor 写入冲突很高，需要继续优化。
4. `drain_phase_b()` 仍是本地 slot 扫描 fanin flags，没有 fanout 反向通知。
5. shared output 每 task output 上限是 `kSharedOutputMaxPerTask = 8`。

## 验证记录

最近一次验证：

- PA Case1 sim 通过。
- PA Case1 上板通过，单卡 device 6，启用 L2 swimlane。
- 按 `merged_swimlane.json` 看，按需 EfDrain 后 PA Case1 global span 从约 2.540 ms 降到约 2.318 ms。

后续每个 phase 完成后仍需要：

1. shared smoke 全量用例通过。
2. PA Case1 sim 功能通过。
3. PA Case1 上板通过。
4. 性能分析时以上板 `merged_swimlane.json` 为准。

---

# 第十部分 - 与 `fully_distributed_within_core.md` 原设计形态的差异

`docs/fully_distributed_within_core.md` 描述的是原始 FDWIC 模型：每核全量 replay、每核复制 TensorMap、owner=builder=executor、block.won 支持 MIX follower 投递。当前 shared map 是在这个模型上为了性能做出的 shared-output/shared-dependency 变体，因此并不是逐字实现原文的每个机制。本节逐项记录差异和原因。

## 51. 原文：每核全量复制 TensorMap；当前：shared build 移除 DistCore::map

原文核心设定：

```text
TensorMap 是每核全量 DUPLICATE。
胜者和败者都做 TensorMap lookup / insert。
只有 build + execute 受所有权门控。
```

当前 shared 实现：

```text
#if !PTO_FDWIC_SHARED_MAP
  DistCore::map
#endif

#if PTO_FDWIC_SHARED_MAP
  DistGlobal::shared_outputs
  DistGlobal::shared_region
#endif
```

也就是说，private build 仍保留原文的 per-core map 路线；shared build 则不再让每个 loser 维护完整 TensorMap。

为什么改：

1. shared map 的目标就是避免 loser 执行 map lookup/insert 的重逻辑。
2. per-core duplicate map 的正确性依赖所有核无条件维护同一份 map；这和 winner-first shared submit 的性能目标冲突。
3. shared output ref 已经提供了 producer identity，不需要再把 runtime-created output 重新登记到每个核的 private map。
4. 普通 Tensor 的跨核依赖可以通过全局 `SharedRegionMap` 维护，不需要每核复制。

正确性替代关系：

| 原文 per-core TensorMap 职责 | 当前 shared 替代 |
| ---------------------------- | ---------------- |
| runtime-created output producer 查询 | `FdwicOutputRef.producer_task_id` |
| shared output 最新 writer | `SharedOutputCell.last_writer` |
| 普通 Tensor overlap writer | `SharedRegionMap` |
| producer descriptor 获取 | `SharedOutputCell.tensors + published` |

这个差异是有意设计，不是漏实现。

## 52. 原文：API 表面保持 `rt_submit_task`；当前 shared：显式 presubmit/winner/loser

原文强调面向编排 API 保持不变，通用原语是：

```text
rt_submit_task(MixedKernels, args)
rt_submit_aic_task(...)
rt_submit_aiv_task(...)
```

当前 shared build 下不采用这个形态。shared API 要求：

```cpp
SubmitToken tok = rt_presubmit_task(mixed);
if (tok.won) {
    outputs = rt_submit_winner(tok, args);
} else {
    outputs = rt_submit_loser(tok, output_count);
}
```

并且 `dist_submit_impl()` 在 shared build 下直接 assert。

为什么改：

1. 如果保持 `rt_submit_task(mixed, args)`，用户侧必须先构造完整 `args`。这会让 loser 仍然承担大量不必要逻辑，尤其 PA 中 loser 会构造复杂 Arg。
2. shared map 的关键优化是 submit 前知道 winner，然后 loser 跳过 Arg 构造、materialize、register、fanin、build。
3. 使用 submit builder/lambda 延迟构造曾被讨论过，但会把热路径变成复杂控制流，且 lambda 是文档设计失误；当前代码选择显式 presubmit。
4. shared/private 返回类型不同。private 返回 `TaskOutputTensors`，shared 返回 `SharedTaskOutputs`。强行保留同一 API 会隐藏语义差异。

因此 shared build 牺牲了原文“API 表面完全不变”的目标，换取 winner-first 的性能路径和更清晰的类型语义。

## 53. 原文：winner 和 loser 都维护 map；当前：loser 只返回 symbolic refs

原文为了保证每核 TensorMap 副本一致，要求：

```text
胜者 AND 败者都做：
  lookup INPUT / INOUT
  insert OUTPUT / INOUT
```

当前 shared build 下 loser 行为：

```text
rt_submit_loser(tok, output_count):
  return SharedTaskOutputs(task_id=tok.task_id, output_count)
```

loser 不做：

- materialize
- output descriptor publish
- shared region register
- fanin collect
- slot build
- shared ref resolve

为什么改：

1. shared build 不再依赖 per-core map 一致性。
2. output identity 是 task id + output slot，loser 能直接构造。
3. 真正的 Tensor descriptor 由 winner 发布，consumer 执行前 resolve。
4. loser 做 map/register 会重新引入 private map 的主要成本。

这也是 shared map 能比 private map 更快的核心来源之一。

## 54. 原文：GM heap 地址是确定性 submit 函数；当前：shared heap 使用 atomic shard 分配

原文描述的 heap 方向是：

```text
每个核在确定性 submit 重放中对每个任务推进 heap top。
任务 N 的输出地址是 submit 序列和输出大小的纯函数。
```

当前 shared 实现不是这个模型。当前 winner 在 materialize 时通过 shared heap shard 分配：

```text
shard = task_id % kSharedHeapActiveShards
cursor = atomic_fetch_add(shared_heap_cursor[shard], reserve)
task_base = shard_base + cursor_offset
```

为什么改：

1. loser 不 materialize，也不应该计算或推进 heap top。
2. 如果为了保持确定性地址让 loser 也推进 heap，就会保留大量 loser 输出布局/heap 逻辑。
3. winner-only allocation 必须使用共享分配状态，否则多个 winner 核之间无法协调 output buffer。
4. 单一全局 heap cursor 会变成热点，所以当前使用 shard cursor。

正确性代价：

- 地址不再是“所有核重放同一 heap top”的纯函数。
- 地址由 winner 分配并发布到 `SharedOutputCell.tensors`，consumer 必须通过 ref resolve 获取。

这个差异和 symbolic output ref 是配套的：既然 loser 不知道真实地址，后续 consumer 就不能依赖 loser 返回 Tensor。

## 55. 原文：cursor 主要按类型全局高水位；当前：按类型做 cursor shard

原文早期形态是：

```text
cube_cursor
vector_cursor
```

后文性能章节也讨论了 cursor 分片。当前代码落地的是：

```cpp
cube_cursor[kCubeCursorShards]      // 当前 8
vector_cursor[kVectorCursorShards]  // 当前 16
alloc_cursor[kLaneCount][kAllocCursorShards]
```

为什么改：

1. PA 中 AIV-only submit 很密集，所有 AIV 抢一个 vector cursor 会造成明显原子竞争。
2. shard 按 task id 选择，不改变 task id，也不改变同 task 的候选集合。
3. 同一个 task 仍只有一个权威 cursor cell，因此不会漏 owner。
4. alloc 没有真实 kernel 调度需求，可以进一步按 lane/block 限制候选。

需要特别说明：

- 当前 cursor shard 不是 tree/group cursor。
- 当前没有 leaf/root 两级 election。
- 当前没有让 loser 跳过权威 claim。

这是和后来被撤销的 grouped cursor 实验的根本区别。

## 56. 原文：block.won 以 task id 为键；当前：WonSlot 池 + task_id metadata

原文描述：

```text
block.won[N] = {
  active_mask,
  kernels,
  args,
  fanin,
  remaining
}
```

它强调 block.won 以 task id 为键，避免同 block 多个并发 MIX 任务互相覆盖。

当前代码中 `BlockWon` 是：

```cpp
struct BlockWon {
    WonSlot slots[kPrivateSlots];
    int32_t any_pub;
};

struct WonSlot {
    state;
    meta.task_id;
    remaining;
    drained[lane];
    lane[lane];
};
```

也就是一个 block-local won slot 池，而不是直接用 `block.won[N]` 数组索引。

为什么当前这样：

1. 当前 mix 还不是最终高性能闭环，先用有限 slot 池承载 anchor deposit。
2. `meta.task_id` 仍记录 task id，避免 follower drain 时丢失 task identity。
3. `state/remaining/drained` 保证每个 deposited subtask 被目标 lane drain 一次，最后完成者发布 task flag。

差异和风险：

- 原文的 `block.won[N]` 更接近“按 task id 直接定位”，天然避免查找和复用歧义。
- 当前 `WonSlot` 池需要 slot state、remaining、drained 协议保证复用正确。
- 当前 mix 尚未作为 shared map 的完整高性能目标验收，因此后续仍需决定是否继续沿用 slot 池，还是改成更贴近原文的 task-id keyed window。

## 57. 原文：MIX follower 异步抽取且不等待；当前：基础机制存在，但 shared mix 未完整闭环

原文对 MIX 的目标非常明确：

```text
AIC anchor 赢 MIX。
同 block AIV follower 不竞争 MIX。
follower 不 wait_until(anchor_progress >= N)。
anchor deposit 到 block.won。
follower 异步 drain 属于自己的 subtask。
```

当前代码已有：

- `g_fdwic_block_won_enabled`
- `publish_joint_deposits()`
- `drain_block_won_if_enabled()`
- `drain_block_won()`
- `WonSlot.remaining`
- `drained[lane]`

但当前 shared map 的主要验证和优化集中在 single AIC、single AIV、alloc、shared output ref、ordinary Tensor dependency。mix task 的完整 shared 支持还没完成。

未完成点包括：

1. MIX shared output ref 在 anchor/follower 多 subtask 下的完整 producer 发布语义。
2. MIX 的 follower slot 中 shared refs 延迟 resolve 的端到端验证。
3. MIX 下 `region register / fanin collect` 由 anchor 一次性生成后，follower drain 使用是否覆盖所有 shared 参数场景。
4. MIX 的 PA 之外 smoke/stress 覆盖。
5. MIX block-won 容量、反压、尾部 drain 的性能验证。
6. MIX 与 shared heap shard、task completion flag、remaining 递减之间的稳定性验证。

因此本文档前面说“mix 不是最终高性能闭环”，指的是：基础 block-won 机制存在，但 shared map 的完整 mix 语义、测试和性能还没有达到当前 single-lane 路径的完成度。

## 58. 原文：wait_until(anchor/published) 需要避免阻塞可并行部分；当前：延迟 resolve 替代 submit 阶段等待

原文中明确反对 follower 在 MIX 上按 task 走位等待 anchor：

```text
不要 wait_until(block.anchor_progress >= N)
```

当前 shared output ref 也采用同样思想：不要在 submit 阶段等待 producer output published。

当前实现：

```text
build slot:
  if producer published:
    resolve
  else:
    save FdwicOutputRef into RingSlot.shared_refs

execute slot:
  after fanin ready:
    resolve remaining refs
```

为什么改成这样：

1. submit/replay 阶段等待 producer 会缩小并行窗口。
2. consumer 真正需要 Tensor descriptor 的时间点是 kernel 执行前。
3. fanin flag 已经保证执行顺序，published flag 只保证 descriptor 可见。

这点和原文的“follower 不等待 anchor、由 deposit 异步到达”是一致的设计思想，但应用在 shared output descriptor resolve 上。

## 59. 原文：完成前沿可在完成时推进；当前 shared：frontier 不在每次 complete 热路径推进

原文中的全局完成前沿 `F` / 回收前沿 `R` 是 task flag 环和 heap/window 回收的核心。当前 shared 实现仍保留 `frontier`，但 shared complete 不在每个 task 完成时调用 `advance_frontier()`。

当前：

```cpp
complete_executed_task():
  publish task flag
#if !PTO_FDWIC_SHARED_MAP
  advance_frontier()
#endif
```

shared build 下，frontier 主要在等待 heap reuse window 时推进：

```text
dist_submit_wait_heap_reuse_window()
  advance_frontier_until(target, max_steps)
```

为什么改：

1. PA 中每 task complete 都推进 global frontier 会给 commit 热路径增加全局 atomic/flag 扫描成本。
2. heap 复用不是每个 complete 都必须立即处理的事情。
3. 等待复用时再推进 frontier，可以把成本移动到真正需要回收空间的慢路径。

正确性条件：

- task flag 仍在 complete 时 release 发布。
- consumer ready 判断仍看 task flag。
- heap 复用前必须等待 frontier 到达对应 target。

也就是说，frontier 推进延后只影响回收时机，不影响依赖 ready 语义。

## 60. 原文：task_completed_flag 是唯一 per-task 共享状态；当前 shared 增加 shared output/region 状态

原文基础 FDWIC 模型中，唯一 per-task 全局共享状态是 completion flag。依赖发现靠每核复制 map，所以不需要全局 producer map。

当前 shared map 必须额外增加：

- `SharedOutputCell`
- `SharedRegionMap`
- shared heap cursors

为什么增加：

1. 去掉 per-core duplicate map 后，producer identity 必须有共享来源。
2. loser 不返回 Tensor，descriptor 必须由 winner 发布到全局位置。
3. ordinary Tensor overlap 不能靠每核 map 推导，必须有全局 writer table。

这确实增加了全局共享状态，但换掉的是“每核都做完整 map 维护”的成本。当前性能目标是用少量高频共享 atomic + winner-only submit，替代大规模 loser map work。

## 61. 当前实现相对原文的完成度判断

| 原文能力 | 当前 shared 实现状态 | 说明 |
| -------- | -------------------- | ---- |
| SPMD replay 同一 submit 序列 | 已完成 | task id 仍由 `local_index++` 得到 |
| claim race 决定 owner | 已完成 | flat cursor shard，不使用 unsafe tree claim |
| owner=builder=executor | single-lane 已完成 | mix 仍需完整闭环 |
| 每核复制 TensorMap | shared 下有意替换 | 用 shared output ref + region map 替代 |
| loser 维护 map | shared 下有意删除 | loser 只返回 symbolic refs |
| deterministic heap 地址 | shared 下有意替换 | winner 用 shared heap shard 分配并 publish descriptor |
| block.won follower deposit | 基础机制存在 | mix shared 还未完整验收 |
| follower 不等待 anchor | 设计目标保留 | 当前 single-lane 不涉及；mix 后续必须继续保证 |
| 每 task completion flag | 已完成 | fanin 仍看 task flag |
| run-ahead 执行 | 已完成 | submit 阶段尽量不等待 producer published |
| shared output descriptor 发布 | 已完成 | `published + tensors` |
| shared INOUT writer 链 | 已完成 | `last_writer exchange/fetch_max` |
| ordinary Tensor overlap 依赖 | 已完成 | `SharedRegionMap` |

## 62. 当前还没完成的任务

下面是当前 shared map 后续明确还要做的任务，不应被“single-lane PA 已跑进 3ms”掩盖。

### 62.1 MIX task 完整支持

目标：

- shared build 下 MIX task 的语义、功能测试和性能达到 single AIC/AIV 路径同等完成度。

需要完成：

1. 明确 MIX shared output 的发布者：
   - 如果 MIX 有 output，哪个 subtask 负责 materialize/publish。
   - 多 subtask 是否都可能写 output，如何约束。

2. 明确 MIX fanin 生成方式：
   - anchor collect 的 fanin 如何完整覆盖 follower subtask。
   - follower drain 出来的 `BuiltSubtask` 是否携带所有 shared refs 和 fanin。

3. 验证 follower 延迟 resolve：
   - shared ref 未 ready 时 anchor deposit 到 `BuiltSubtask.shared_refs`。
   - follower drain 到 `RingSlot.shared_refs`。
   - follower execute 前 resolve。

4. 验证 completion：
   - 每个 subtask 完成后递减 `remaining`。
   - 最后一个 subtask 发布 task flag。
   - consumer 只依赖一个 task id。

5. 验证 block-won 复用：
   - 多个 MIX task 连续 deposit。
   - follower 慢于 anchor 时不覆盖未 drain slot。
   - tail drain 不丢任务。

6. 增加 shared MIX smoke/stress：
   - 1C+1V
   - 1C+2V
   - 2V AIV-only
   - MIX producer -> single consumer
   - single producer -> MIX consumer
   - MIX 中 shared ref view
   - MIX 中 INOUT / OUTPUT_EXISTING

### 62.2 SharedRegionMap 性能与容量

当前 region map 是全局 bucket + entry append：

- 插入使用 bucket sentinel。
- lookup 遍历 bucket 链。
- entry 不回收，只受 `kSharedRegionCap` 限制。

后续需要：

1. 测普通 Tensor 写入密集用例中的 bucket 冲突。
2. 明确 region entry 是否需要 window 回收。
3. 评估 `insert_lock` 字段是否废弃并可删除。
4. 验证 view Tensor byte range overlap 的完整性。

### 62.3 fanin ready 扫描优化

当前 `drain_phase_b()` 每次扫描 slot 的 fanin producer flags：

```text
for each occupied slot:
  for each fanin:
    task_flag_ready(fanin)
```

这和原文 pull 模型一致，但性能上可能仍有单核空白。后续如果要改 fanout/pending_count，必须重新设计：

- fanout list 的 owner。
- per-core 还是 global 存储。
- atomic/cacheline 布局。
- producer complete 与 consumer submit 并发建边。

当前没有采用 fanout 方案。

### 62.4 claim 进一步优化

当前已保留安全的 cursor shard，但 claim 仍是 PA 中主要控制开销之一。

后续可分析：

1. flat cursor 下 atomic primitive 的单次成本。
2. trace 对 claim 的扰动。
3. 是否存在不改变语义的候选过滤。
4. claim cell 是否能在完整 ring reuse 协议下重新评估。

不能做：

- 不带 release 的 tree/group cursor。
- 按 task id 固定到某个 AIV lane，限制 kernel task 调度范围。
- 让后续 task 越过当前 task 的权威 claim。

### 62.5 Resolve/view 能力扩展

当前 `FdwicOutputRef` view 只支持一维。

后续需要：

1. 多维 view ref 表示。
2. 连续多维 view 合并。
3. stride/contiguous 语义验证。
4. shared ref view 与 region map byte range 的交互边界。

### 62.6 验证矩阵补齐

当前已验证重点是：

- shared smoke 全量。
- PA Case1 sim/onboard。
- PA 性能。

后续还应补：

1. shared MIX 全量用例。
2. region map overlap stress。
3. shared heap shard wrap stress。
4. 超过 `kTaskPayloadSlots` 的 replay stress。
5. `last_writer` runahead stress：
   - consumer INOUT 先于 producer materialize 到达。
   - producer fetch_max 不覆盖 later writer。
