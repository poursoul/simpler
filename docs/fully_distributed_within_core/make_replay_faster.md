# 让 Replay 更快 —— Compete-First 编排

本文档为 [fully_distributed_within_core.md](fully_distributed_within_core.md) 的 runtime 提出一项
性能增强：**把 kernel 的参数块（param-block）构建移出 replay 关键路径**，让绝大多数“败者”核在
认领失败后**立即跳过**昂贵的参数打包，从而显著降低每个 AICore 花在 “replay” 阶段的 cycle。

本增强同时已并入主设计文档的 [§6.8](fully_distributed_within_core.md)。本文提供更完整的动机、
正确性论证与 API/codegen 落地方案。

---

## 1. 背景：replay 阶段在做什么

全分布式模式下**没有中心调度器**：编排函数被加载并**同时运行在每一个 AICore 上**（SPMD），
每个核完整重放（replay）同一段编排程序，逐个 submit 点竞争任务所有权（[§1、§2](fully_distributed_within_core.md)）。

编排函数由两部分组成：

1. **高层控制/数据流**——描述 kernel 之间关系的循环与分支（batch/head/block 迭代、`PTO2_SCOPE`
   等）。
2. **每个 kernel 的参数块构建**——在每次 `rt_submit_*` 之前，把该 kernel 的输入/输出/标量参数
   打包进一个 `L0TaskArgs`（`add_input` / `add_output` / `add_scalar`），其中还包括构造
   tensor view（`tensor.view(...)`）等。

以 `paged_attention_orch.cpp` 的 SplitK PV matmul 为例（简化）：

```cpp
// === Task 3: SplitK PV matmul (accumulated P @ V) ===
L0TaskArgs params_pv;
params_pv.add_input(pij_f16);          // 打包输入
params_pv.add_input(vj);               // 打包输入（vj 来自 value_cache.view(...)）
params_pv.add_output(tile2d_ci);       // 打包输出 create-info
CYCLE_COUNT_LAP(prof_param_setup);     // ← 这一段就是 param-block 构建
TaskOutputTensors pv_outs = rt_submit_aic_task(FUNC_PV_MATMUL, params_pv);
```

在该文件自带的 profiling（`ENABLE_PROFILING`）里，`prof_param_setup` 与 `prof_tensor_view`
两项**合计占据了片上编排墙钟的大头**。

## 2. 问题：参数块构建被每个核对每个任务无条件执行

关键事实：**上面这段参数块构建，在每个核上、对每个任务都会执行——无论该核是否会赢得该任务。**

而参数块**真正被谁用到**？

| 角色 | 是否需要完整参数块 | 用途 |
| ---- | ------------------ | ---- |
| **winner**（认领成功） | **需要**全部（input + output + scalar） | 更新 TensorMap、构建任务、执行 kernel |
| **loser**，`tensormap == private`（每核复制，即本设计的默认模型 [§4](fully_distributed_within_core.md)） | 只需要 **output** 部分 | 把 output 作为 producer 登记进本核私有 TensorMap，以保持每核副本一致 |
| **loser**，`tensormap == shared`（全局共享 map 变体） | **完全不需要** | 无 —— 全局 map 由 winner 维护 |

由于任一任务只有约 `1 / 参与核数` 的核会成为 winner，**绝大多数核都是 loser**。它们却照样付出了
完整参数块构建（尤其是 input 侧的 tensor view + `add_input`、以及 scalar 打包）的开销——这部分对
loser 而言是**纯浪费**，正是 replay 阶段偏慢的主因。

> 已有的 runtime 内部优化（[§6.4](fully_distributed_within_core.md) 的 winner-only fan-in +
> `built[]` 后置）**无法覆盖这里**：那些优化发生在 `rt_submit_*` 的**内部**；而参数块构建发生在
> `rt_submit_*` 被调用**之前**的编排代码里，等 runtime 拿到 `L0TaskArgs` 时，昂贵的打包早已完成。
> 要省掉它，必须让**竞争（认领）发生在参数块构建之前**，并让编排代码按胜负**条件化**地构建参数。

## 3. 方案：先竞争，后按胜负条件化构建参数

### 3.1 `local_cursor` + `compete_cursor` 原语（**runtime 内部，不暴露给编排**）

1. **`local_cursor` 是每核私有计数器**，带确定性初值——它就是现有的 `local_current_task_index`
   （[§2](fully_distributed_within_core.md)）。每次提交时先 `++`，得到确定性任务 id `N`（各核一致，
   与最终谁执行无关）。**它是 runtime 内部状态，编排层不需要、也不应该看到它**（见 §3.3 对“为何可以
   隐藏”的说明）。
2. runtime 在每次提交内部先做一次竞争：

   ```text
   bool compete_cursor(T, local_cursor):
       # 原子地比较 local_cursor 与该类型的全局 cursor[T]（cube / vector）
       old = atomic_fetch_max(global_cursor[T], local_cursor)
       return local_cursor > old      # TRUE=winner（并已把 global_cursor 推到 local_cursor）；FALSE=loser
   ```

   这**正是**现有 `claim()` / `atomic_fetch_max` 的语义（[§11.1](fully_distributed_within_core.md)），
   区别仅在于：把它**提前到参数块构建之前**执行，使竞争先于（且门控）昂贵的参数打包。分片 cursor
   （[§6.6](fully_distributed_within_core.md)，`N % G`）与两条 cursor（cube/vector）语义不变。

   > **`compete_cursor` 不必作为公开 API 暴露给编排。** 它是 `rt_submit_*` 提交路径的一个内部步骤：
   > runtime 在收到本次提交后，自行 `local_cursor++`、调用 `compete_cursor` 得到 winner/loser，再据此
   > 决定是否/如何回放参数（§3.3）。编排只描述“这个任务要哪些 input/output/inout”，不感知 cursor 与
   > 竞争。

### 3.2 条件化编排（由 codegen 生成）

编排在每个 submit 点改为生成如下结构：

```text
local_cursor++                                   # 任务 id N（确定性、各核一致）
win = compete_cursor(T, local_cursor)            # 先竞争，再决定要不要打包参数

if win:
    # —— winner 路径：与当前 rt_submit_* 完全等价，唯一区别是不再重复 claim ——
    #     （认领已在上面的 compete_cursor 里完成，此处直接进入构建）
    构建【完整】参数块：add_input / add_output / add_scalar（含 input 侧 tensor.view）
    update_tensormap(task)                        # 查 INPUT/INOUT→fanin；插 OUTPUT/INOUT→producer=N
    构建任务进本核私有环（带 fanin）
    ... 任务检查 / 执行（Phase B，多核则 winner-gated launch，§3.1）
else if tensormap == shared:
    # —— loser + 共享 map：什么都不做 ——
    pass
else:  # loser + tensormap == private（本设计默认的每核复制模型）
    构建【仅 output】参数块：output create-info + 确定性分配（§9.3）
    仅把 OUTPUT/INOUT 作为 producer=N 插入本核私有 TensorMap
    # 跳过：input 侧 tensor.view、add_input、add_scalar、fan-in lookup、任务构建/执行
```

- **winner 路径**行为与今天的 `rt_submit_*` 逐字等价（TensorMap → 私有环构建 → 执行），**唯一区别是
  不再重复执行 claim**——认领这一步已经由前面的 `compete_cursor` 完成（`compete_cursor` 本身就是那次
  claim），winner 分支直接从“已认领成功”的状态进入构建，不再碰 cursor。
- **loser + shared**：完全跳过。
- **loser + private**：只构建并登记 output，跳过 input 侧打包与所有 winner 专属工作。

省掉的正是 loser 身上最贵的部分——input 侧的 `tensor.view` + `add_input` + `add_scalar`
（`prof_tensor_view` + `prof_param_setup`），而这些对 loser 本就没有任何用途。

### 3.3 API 形态：单个 builder 回调 + runtime 自行决定

**目标 API（推荐）。** 编排只提供**一份**参数清单——通过 builder 调用 `add_input` / `add_output` /
`add_inout`——由 `rt_submit_aic_task` **在内部**完成 `local_cursor++` → `compete_cursor` →
按 winner/loser × tensormap-mode **自行决定回放哪些项**。`local_cursor` 与竞争都藏在 API 后面，编排
不感知：

```cpp
// 单个 builder 回调；runtime 内部 local_cursor++ → compete → 决定回放策略。
// input / output / inout 全部以惰性 thunk 登记，runtime 按角色 × map-mode 选择性求值。
rt_submit_aic_task(FUNC_PV_MATMUL, [&](SubmitBuilder &b) {
    b.add_input([&] { return pij_f16; });                                 // 仅 winner 求值
    b.add_input([&] { return value_cache.view(kv_shapes, kv_offsets); }); // 仅 winner 求值
    b.add_output([&] { return tile2d_ci; });                              // winner + loser(private) 求值；shared 跳过
    // b.add_inout([&] { return x; });                                    // produce: winner+private；consume: winner；shared 全跳
    // b.add_scalar([&] { return scale_value; });                         // 仅 winner 求值（同 input）
});
```

runtime 侧回放策略（一次回调即可覆盖三种情况）：

| 结果 | `add_input` / `add_scalar`（惰性项） | `add_output` / `add_inout` 的 produce 侧（惰性项） | `add_inout` 的 consume 侧 |
| ---- | ------------------------------------ | ------------------------------------------------- | ------------------------- |
| **winner** | 求值 + 打包（用于 build/exec） | 求值 + 分配 + build | 求值（fan-in lookup） |
| **loser + private** | **跳过（不求值）** | 求值 + 确定性分配 + 插入 map（§9.3、§4） | 跳过 |
| **loser + shared** | **跳过** | **跳过**（连 output/inout thunk 都不求值，零开销；map/堆全局共享，见 §4 第 4 条约束） | 跳过 |

**为什么 input / output / inout 全部要以惰性项（lambda/thunk）登记——这是能否省下开销的关键。** 昂贵
的构建（如 input 侧 `tensor.view`、output 的 create-info 组装）若写成 `b.add_output(tile2d_ci)` 这类
**即时求值**，C++ 会在调用 `add_output` **之前**就把参数求值掉；等 runtime 决定“这是 loser，跳过”时，
构建早已执行——**优化落空**。因此三类参数都必须以 `[&]{ return <表达式>; }` 交给 builder，runtime 才能
对不需要的项**根本不求值**。分角色看：

- **input**：只有 winner 需要（build/exec）；loser 一律跳过。
- **output / inout 的 produce 侧**：winner 与 **loser+private** 需要（确定性分配 + 插入 map，§9.3/§4）；
  但 **loser+shared 不需要**（map/堆全局共享）。**正因为要让 loser+shared 也跳过 output/inout 的构建、
  省下这一档 replay 开销，output/inout 同样必须惰性**——否则即时求值会在 shared 场景下白白付出构建成本。
- **inout 的 consume 侧**（fan-in lookup）：winner-only。`add_inout` 的双重身份（produce + consume）由
  builder 内部按角色区分，编排只写一次。

**这样就同时回答了两个问题：**

- **单一清单 + runtime 决定？可以。** 只要 input 项惰性化，runtime 就能对 winner 全量回放、对
  loser+private 只回放 output/inout-produce、对 loser+shared 全部跳过——**编排只写一份清单**，无需
  `build_full` / `build_outputs` 两个闭包。
- **`local_cursor` 可以隐藏？可以。** 它就是每核私有的 `local_current_task_index`，由 `rt_submit_*`
  内部推进；竞争也在内部完成。编排层完全不需要看到 cursor。

**落地。** runtime 侧新增一个 ops 表项承接这种“单 builder 回调 + 竞争优先 + 条件回放”的提交路径；
`SubmitBuilder` 的 `add_input` / `add_output` / `add_inout` 均接收惰性 thunk，runtime 按角色 × map-mode
选择性求值（input=winner；output/inout-produce=winner+private；inout-consume=winner；shared 全跳）。旧的
`rt_submit_*(kernel_id, L0TaskArgs)` 保留兼容（相当于所有项立即求值、winner 全量、无 loser 优化）。

> **权衡与退化行为。** 若某项图省事仍写成即时求值（如 `b.add_output(tile2d_ci)`），API 仍然**正确**
> ——只是该项对本可跳过它的 loser 也执行了，退回到“省不掉这一项”的旧成本。换言之，惰性化是**逐项可选
> 的性能手段**，不影响正确性；codegen 应默认对 input / output / inout 三类都生成惰性形式。

> **codegen 视角。** 用户所说的 “generating more optimized orchestration function” 即由代码生成器
> 直接产出上述 compete-first 结构：为每个 kernel 生成一份 `SubmitBuilder` 回调，**把 input / output /
> inout 三类参数都自动包成惰性 thunk**（`[&]{ return <表达式>; }`）；竞争与 `local_cursor` 全部交给
> runtime。手写编排也可按同一形态改写。完整的 codegen 改进方案见 §8。

### 3.4 所有参数类型的惰性归类（`add_scalar` / 显式依赖 / launch_spec 等）

`L0TaskArgs`（`Arg`）当前提供的参数登记 API 不止 `add_input/output/inout`。要把 replay 成本压到最低，
**除“kernel 身份”外的所有参数都应以惰性 thunk 登记**，再由 runtime 按“谁需要”分三档求值。归类如下：

| 参数 API | 类别 | 求值时机（惰性档） | 理由 |
| -------- | ---- | ------------------ | ---- |
| kernel 身份：`MixedKernels`（`aic/aiv0/aiv1_kernel_id` + `active_mask`） | **Tier 0：eager，不惰性** | **compete 之前**（所有核） | compete 要靠它判定任务**类型**（cube/vector）以选 `cursor[T]` 与分片 `N%G`（§3.1、§6.6）；必须先于认领可知。它作为 `rt_submit_*` 的直接实参传入，不进 builder |
| `add_output`（`TensorCreateInfo`→分配 / 既有 Tensor 写目标） | **Tier 1：winner + loser(private)** | winner 或 loser+private 求值；loser+shared 跳过 | 需据其大小做确定性分配（§9.3）并把 producer=N 插入本核私有 map（§4）；shared 变体由 winner 维护全局 map，loser 免 |
| `add_inout` 的 **produce 侧** | **Tier 1：winner + loser(private)** | 同上 | INOUT 既产出新版本（登记 producer）又消费旧版本；产出侧同 output |
| `add_input` | **Tier 2：winner-only** | 仅 winner | 只用于 build/exec；loser 从不消费输入 |
| `add_inout` 的 **consume 侧**（fan-in lookup） | **Tier 2：winner-only** | 仅 winner | 消费侧的依赖解析是 winner-only（§6.4） |
| `add_scalar` / `add_scalars` / `add_scalars_i32` / `copy_scalars_from` | **Tier 2：winner-only** | 仅 winner | 标量是 kernel 执行参数，不参与 map/分配；loser 不需要 |
| 显式依赖 `add_dep` / `set_explicit_deps` | **Tier 2：winner-only** | 仅 winner | 显式 fan-in producer id，仅供 winner 依赖轮询 |
| `add_no_dep`（输入角色的张量） | **Tier 2：winner-only** | 仅 winner | 不建依赖的输入，仅 build/exec 用 |
| `launch_spec`（SPMD `block_num` 等）/ `set_allow_early_resolve` | **Tier 2：winner-only**（若纯执行元数据） | 仅 winner | 执行期元数据。**例外**：若某字段会影响任务类型/所有权判定，则须上提到 Tier 0 |

**结论（直接回答“`add_scalar` 等是否也要惰性”）：是。** 除 kernel 身份（Tier 0，必须 eager 以供 compete
选 cursor）外，**其余全部参数——包括 `add_scalar`、显式依赖、`add_no_dep`、`launch_spec`——都应以惰性
thunk 登记**，让 loser+shared 能整个跳过、loser+private 只求值 Tier 1。标量本身打包很廉价，惰性化对它
的直接收益有限，但纳入是为了：(a) 让 loser+shared 真正**零开销**；(b) API 统一，codegen 无需对参数类型
特判。

> **一处必须遵守的约束：输出句柄的数据流。** `add_output` 返回的 Tensor 句柄常被**后续 submit 点**当作
> 输入引用（如 `oi_tmp` 喂给 online_update）。其地址是任务 id 的**纯函数**（§9.3 确定性布局），因此可在
> 任意核上**确定性重建**。为使 loser+shared 跳过 output 后、下游仍能正确引用：**输出句柄只能在惰性参数
> thunk 内被消费，不得进入控制流**；下游对该张量的解析要么走（共享/私有）TensorMap 按区域查找，要么由
> codegen 生成确定性重建（§8.3）。这样，哪个核赢得下游任务，它的 input thunk 就在该核上按确定性地址取到
> 句柄——无需本核曾经 build 过上游 output。

## 4. 正确性论证

1. **任务 id 不变、各核一致。** `local_cursor` 仍是每核私有、单调 `++`，竞争只决定“谁执行”，
   **不改变任务 id**（[§2](fully_distributed_within_core.md)）。所有核仍走完全相同的确定性 submit
   序列——把竞争提前不影响 id 分配。
2. **winner 行为不变（但不重复 claim）。** winner 分支就是今天的 `rt_submit_*`，TensorMap 更新、
   私有环构建、winner-gated launch（[§3.1](fully_distributed_within_core.md)）、执行全部保留；**唯一
   差别是不再执行 claim**——认领已由前面的 `compete_cursor` 完成（它就是那次 `atomic_fetch_max`
   claim，§11.1），winner 分支不再碰 cursor，避免对同一任务做第二次原子认领。
3. **private TensorMap 每核副本一致（关键）。** producer 条目只在处理 `OUTPUT`/`INOUT` 时创建，
   且**必须每个核都创建**，否则本核上的下游消费者会查不到（[§4](fully_distributed_within_core.md) 的
   “为什么部分 map 是错的”）。因此 loser+private **仍求值 output/inout-produce thunk 并登记**——这正是
   本方案对 loser+private 仍回放 output/inout 项的原因。相对地，**input 侧 lookup 只有 winner 需要**（消费者的 fan-in 只
   给 winner 用于依赖轮询），loser 跳过 input 打包与 lookup 安全无误——这与
   [§6.4](fully_distributed_within_core.md) 的 winner-only fan-in 完全同源，只是把它从 runtime 内部
   进一步上推到了编排层。
4. **确定性 heap 布局。** `heap_top` 由每个核对每个任务**无条件**确定性推进
   （[§9.3](fully_distributed_within_core.md)），依赖 output 的大小。loser+private 构建 output
   create-info 后照常做确定性分配，布局不漂移。**loser+shared 若要整体跳过**，前提是该变体的输出堆
   也是**全局共享分配**（而非每核复制 bump）——这是 `tensormap == shared` 变体的配置约束，需与
   [§9](fully_distributed_within_core.md) 的分配模型配套。
5. **多核任务不受影响。** anchor/follower、launch、joint 完成计数
   （[§3.1](fully_distributed_within_core.md)）都在 winner 分支内，loser 不参与，语义不变。

## 5. 一处重要 caveat：区分“控制流读取”与“纯参数打包”

编排里有些量来自 `get_tensor_data(...)`（如 `paged_attention_orch.cpp` 读 `context_lens` /
`block_table` 得到 `cur_seq`、`cur_block_idx`、循环上界等）。这些是**驱动控制流**的读取——决定了
后续会 submit 哪些任务、循环走多少次——因此**必须由所有核执行**，不能移进 winner-only 闭包，否则
各核的 submit 序列会分叉、任务 id 不再一致。

因此 codegen（或手写改写）必须严格区分：

- **控制流读取 / 索引计算**：保留在提交调用**之前**、所有核都执行的公共路径里（在 `SubmitBuilder`
  回调**之外**）。
- **纯参数打包**（`tensor.view` 后 `add_input` 等）：才可放进 builder 回调、以惰性 input 项交给
  runtime 条件求值。

对 loser+private，回调里 output 的形状若依赖某控制流量，该量已在公共路径算好，惰性/直接项只做纯打包，
无副作用；被跳过的 input thunk 同理不含控制流读取。

## 6. 预期收益

- **loser 的 replay 成本**：从“完整参数块”降到 **outputs-only（private）** 或 **零（shared）**，省掉
  input 侧 `tensor.view` + `add_input` + `add_scalar`（profiling 中 `prof_tensor_view` +
  `prof_param_setup` 的主要部分）。
- **摊销随核数放大**：核越多，单核赢得的任务越少、走 loser 快路径的比例越高——省得越多，正好补上
  [§6.2](fully_distributed_within_core.md) 里“SPMD 冗余重放随核数近线性增长”的开销。
- **与 [§6.4](fully_distributed_within_core.md) 正交叠加**：§6.4 省的是 `rt_submit_*` **内部**的
  fan-in lookup 与 `built[]` 拷贝；本方案省的是 `rt_submit_*` **之前**的参数打包（tensor view +
  `add_*`），量级更大。二者相加把 loser 的整条 submit 路径压到接近“只推进 cursor + 登记 output”。

## 7. 落地清单

| 改动 | 位置 | 说明 |
| ---- | ---- | ---- |
| 新增“单 builder 回调 + 竞争优先 + 条件回放”提交路径 | orchestration API（`PTO2RuntimeOps` 加 ops 项）+ dist_engine submit runtime | `rt_submit_*` 内部：`local_cursor++` → `compete_cursor`（= 现有 `claim()`/`atomic_fetch_max`，§11.1）→ 按 winner/loser × map-mode 回放。`local_cursor` 与竞争**不暴露**给编排 |
| `SubmitBuilder`：input / output / inout 全惰性 | orchestration API | 三类均以 thunk 登记；求值策略：input=winner；output/inout-produce=winner+private；inout-consume=winner；loser+shared 全跳 |
| 保留旧 `rt_submit_*(kernel_id, L0TaskArgs)` | orchestration API | 兼容路径：所有项立即求值、winner 全量、无 loser 优化 |
| codegen 生成 builder 回调、input 自动惰性化 | Codegen（`examples/`） | 把 input 参数包成 `[&]{ return <view>; }`；控制流读取留在回调之外的公共路径（§5） |
| `tensormap == shared` 变体的堆分配配套 | dist_engine 内存管理（§9） | shared 变体需全局共享分配，loser 方可整体跳过（§4 第 4 条约束） |

## 8. PYPTO 前端 / orchestration codegen 改进方案

本章给出**完整**的代码生成侧改进方案：PYPTO 前端把用户的图/DSL 降级（lower）为 AICore 上重放的
orchestration function（`aicpu_orchestration_entry`）。要落地 §3 的 compete-first + 单 builder 惰性
提交，codegen 的产物形态必须改变。本章描述改动目标、核心变换、所需分析、边界情况、分阶段落地与验证。

### 8.1 现状：codegen 生成的 orchestration 形态

今天 codegen 为每个任务生成**直线式（straight-line）**代码，且在 `rt_submit_*` **之前内联**完成参数
打包（以 `paged_attention_orch.cpp` 的 PV matmul 为原型）：

```cpp
// 控制流读取（索引/边界，来自 tensor 数据）
uint64_t cur_block_idx = get_tensor_data<int32_t>(block_table, 2, bt_idx);
Tensor vj = value_cache.view(kv_shapes, kv_offsets);   // ← 纯参数打包（view）
// 参数块内联构建（每个核都执行）
L0TaskArgs params_pv;
params_pv.add_input(pij_f16);
params_pv.add_input(vj);
params_pv.add_output(tile2d_ci);
TaskOutputTensors pv_outs = rt_submit_aic_task(FUNC_PV_MATMUL, params_pv);
const Tensor &oi_tmp = pv_outs.get_ref(0);             // ← 输出句柄，喂给下游任务
```

问题（§1–§2）：`view` + `add_*` 这段**纯参数打包**被每个核无条件执行，而它对 loser 是浪费。

### 8.2 目标形态：compete-first + 单 builder 惰性回调

codegen 改为对每个任务生成一次 `rt_submit_*(FUNC, builder_lambda)`，其中：

- **kernel 身份**（`FUNC` / `MixedKernels`）作为直接实参（Tier 0，eager，供内部 compete，§3.4）。
- **参数打包**全部搬进 `builder_lambda`，并按方向以**惰性 thunk** 登记（§3.3、§3.4）。
- **控制流读取 / 索引计算**留在 lambda **之外**的公共路径（所有核都执行，§5）。

```cpp
// 控制流读取仍在公共路径（所有核执行）
uint64_t cur_block_idx = get_tensor_data<int32_t>(block_table, 2, bt_idx);
uint32_t kv_offsets[2] = {static_cast<uint32_t>(cur_block_idx * block_size), 0};
// 参数打包进单 builder 回调，全部惰性
auto pv = rt_submit_aic_task(FUNC_PV_MATMUL, [&](SubmitBuilder &b) {
    b.add_input([&] { return pij_f16; });
    b.add_input([&] { return value_cache.view(kv_shapes, kv_offsets); });  // view 惰性，仅 winner 求值
    b.add_output([&] { return tile2d_ci; });
});
const Tensor &oi_tmp = pv.get_ref(0);   // 确定性句柄（§8.3）：地址 = f(task_id)
```

### 8.3 codegen 的核心变换

| 变换 | 内容 | 依据 |
| ---- | ---- | ---- |
| **T1 参数惰性化** | 把每条 `add_input/output/inout/scalar/dep/no_dep` 连同其实参表达式（尤其 `tensor.view(...)`）包成 `[&]{ return <expr>; }` 交给 `SubmitBuilder`；按 §3.4 归类到 Tier 1/2 | §3.3、§3.4 |
| **T2 控制流/打包分离** | 对每个任务的输入做 def-use 分析：凡是**驱动后续控制流或索引**的值（`get_tensor_data`、循环边界、`view` 的 offset/shape 计算）留在 lambda 外的公共路径；只有**最终打包**（`view` 调用 + `add_*`）进 thunk | §5 |
| **T3 输出句柄确定性化** | 上游 `add_output` 返回的句柄若被下游任务引用，改为**确定性重建**（地址 = `f(task_id)`，shape = create-info）或按 TensorMap 逻辑区域解析，并保证句柄**只在下游的惰性 thunk 内被消费**、不进控制流 | §3.4 约束、§9.3 |
| **T4 kernel 身份前置** | 把 `MixedKernels`（`active_mask`）作为 `rt_submit_*` 直接实参，确保 compete 在参数打包前可判定任务类型 | §3.1、§3.4 |
| **T5 scope/循环不变量外提** | 保持 `PTO2_SCOPE` 结构不变；把 loop-invariant 的 `TensorCreateInfo`（如 `tile2d_ci`）仍在循环外构造一次，thunk 内只引用 | §9.4 |

### 8.4 所需的前端分析

- **def-use / SSA**：识别每个任务参数表达式的依赖链，判定它是否**流入控制流**（决定后续 submit 序列）。
  流入控制流 → 归公共路径（Tier 0 语义，所有核执行）；否则 → 可进惰性 thunk。**保守规则**：拿不准时归
  公共路径（宁可不省，不可让 submit 序列分叉）。
- **方向标注**：codegen 本就知道每个张量参数是 INPUT/OUTPUT/INOUT（它据此生成 `add_*`），直接映射到
  §3.4 的 Tier 分类，无需额外推断。
- **句柄生存期**：追踪 `add_output` 返回句柄的所有使用点；若存在**控制流使用**（极少见），必须把该输出的
  确定性重建上提为无条件执行（退化为不省该项），并告警。

### 8.5 边界情况

- **控制流依赖 tensor 读**（如 `context_lens` 决定循环次数）：始终在公共路径、所有核执行——这是 §5 的
  硬约束，codegen 不得下沉进 thunk。
- **动态 shape**：shape 计算若依赖控制流读，其结果已在公共路径可得；thunk 内 `view` 只做纯构造。
- **INOUT 的双侧**：codegen 对一个 `add_inout` 生成一个 thunk，runtime 内部按角色分别用于 produce 登记
  （Tier 1）与 consume lookup（Tier 2）；codegen 不需要拆成两个。
- **`tensormap == shared` 变体**：codegen 产物**不变**（同一份惰性 builder）；private/shared 的差异完全在
  runtime 回放策略里（§3.3 表）。但 shared 变体要求输出堆全局共享分配（§4 第 4 条），且下游解析必须走
  共享 map / 确定性地址（T3）。
- **错误处理**：`SubmitBuilder` 沿用现有 `has_error`/`report_fatal` 路径；thunk 内构造失败仍能上报。

### 8.6 分阶段落地

1. **runtime 先行**：新增 `SubmitBuilder` + “单 builder 回调”ops 项与 compete-first 提交路径；旧
   `rt_submit_*(kernel_id, L0TaskArgs)` 保留（§7）。此步不改 codegen，用手写用例验证语义。
2. **codegen 双模式**：codegen 增加一个开关，可继续输出旧的直线式，也可输出新的 builder 形态；先对
   `paged_attention` 等基准用例产出新形态。
3. **T2/T3 分析接入**：实现 def-use 分离与输出句柄确定性化；对无法安全惰性化的项自动退化为即时求值
   （正确性优先，§3.3 退化说明）。
4. **全量切换 + 清理**：所有用例校验通过后，将新形态设为默认；旧路径转为兼容/回归对照。

### 8.7 验证

- **Golden 一致**：新旧 codegen 产物在全部用例（bgemm / paged_attention / paged_attention_ringbuffer /
  mix_coown 等）上结果逐位一致——因为惰性化只改“何时/是否求值”，不改任务 id 与 map 内容（§4）。
- **确定性不变量**：抽查各核的 per-core map 与 `heap_top` 演化一致（§4、§9.3）。
- **性能**：用 §6.2/§6.3 的 skip-exec 口径与 `paged_attention_orch.cpp` 自带 profiling，对比新旧
  `prof_param_setup` + `prof_tensor_view` 及片上编排墙钟，验证 loser 快路径确实压低了 replay。

## 9. 相关文档

| 文档 | 关联性 |
| ---- | ------ |
| [fully_distributed_within_core.md](fully_distributed_within_core.md) | 本增强的主设计文档；见 §6.8（本方案的并入版）、§4（TensorMap 一致性）、§6.4（winner-only fan-in）、§9.3（确定性布局） |
| [make_replay_faster_review(1).md](<make_replay_faster_review(1).md>) | 对本文档的设计审计；§10 针对其结论给出修正分析 |

## 10. 看过 review 后方案的修正分析

本章回应审计 [make_replay_faster_review(1).md](<make_replay_faster_review(1).md>)，重点针对其 §4.3 指出的
**“shared loser 直接跳过存在 producer 发布竞态”**，记录一个具体的改进思路——引入全局
`producer_publish_cursor`——并推演它能解决什么、不能解决什么、以及由此得到的修正后落地方案。

### 10.1 问题复述：shared 模式的 producer 发布竞态

private（每核复制）模式没有竞态，是因为**每个核按 task id 顺序 replay**：核 B 走到 task 11 之前，必然
已经在自己的 map 里登记过 task 10 的 producer（哪怕 B 输掉了 task 10）。于是本核 lookup 永远看得到所有
更早的 producer。

shared（全局单份 map）模式若让 loser **完全跳过**，这个“本核顺序 replay 保证历史 producer 本地可见”的
性质就消失了（审计 §4.3）：

```text
核 A 赢得 task 10，但尚未把 task 10 的 producer 元数据发布进 shared map
核 B 输掉 task 10 → 完全不 replay
核 B 继续推进、赢得 task 11 → 为 task 11 在 shared map 做 fan-in lookup
=> task 10 的 producer 尚不可见 → task 11 漏依赖（错误）
```

根因：不同核并发推进不同 task id，**物理的 build/producer 注册并不按 task id 全局保序**，而 shared map
是所有核共享的唯一一份。

### 10.2 改进思路：全局 `producer_publish_cursor`（连续发布前沿）

**定义。** 引入一个全局单调计数器 `producer_publish_cursor`（记作 `P`），表示 **“所有 id ≤ `P` 的任务，
其 producer 元数据都已发布进 shared map”** 的连续前缀水位线。它与已有的**连续完成前沿 `F`**
（[§11.4](fully_distributed_within_core.md)）是**同构**的机制，只是跟踪的事件不同：`F` 跟踪“执行完成”，
`P` 跟踪“producer 已登记”。

**发布与推进（生产者侧）。** 每个任务的 winner 在把该任务的 producer 元数据（fresh `OUTPUT` 的
`owner_task_id`/描述符、`INOUT`、`OUTPUT_EXISTING`）写进 shared map 之后，`release` 置位一个 per-task
`published[N]` 标志；`P` 用与 `F` 相同的**协作式 CAS** 推进：

```text
publish_producers(N):                 # winner 完成 shared map 登记后
    ... 写入 shared map 各 producer 条目 ...
    store_release(published[N], true)
    while load(published[P+1]):        # 任意核可协作推进，开销摊薄
        CAS(P, P, P+1)
```

**门控依赖发现（消费者侧）。** 一个任务 `N` 的 winner 在做 fan-in lookup **之前**，先 `acquire` 等待
`P ≥ N−1`——即“所有更早任务的 producer 都已发布”——然后再查 shared map：

```text
claim 赢得 task N（winner）
wait_until(load_acquire(producer_publish_cursor) >= N - 1)   # 关键新增门
fanin = shared_map.lookup(inputs of N)                       # 此刻历史 producer 保证可见
... build / execute ...
publish_producers(N)                                         # 登记本任务 producer，推进 P
```

这样，审计 §4.3 里“核 B 为 task 11 lookup 时 task 10 尚不可见”的窗口被直接消除：B 会阻塞到
`P ≥ 10` 才 lookup，而 `P ≥ 10` 蕴含 task 10 已发布。

### 10.3 为什么它能有效降低 shared 模式的实现复杂性

审计 §4.3 列出 shared/sharded 模式要补齐的四件事，`producer_publish_cursor` 用**一个已被验证的模式**
（连续前沿 + 协作 CAS + acquire 等待，与 `F` 同构）**一次性覆盖其中三件**：

| 审计 §4.3 要求 | `producer_publish_cursor` 的对应 |
| -------------- | -------------------------------- |
| producer 元数据的有序发布 / 每 shard published watermark | **正是** `P`（连续发布前沿）；分片见 §10.5 |
| consumer lookup 前确保历史 producer 已发布 | **正是** winner 的 `wait_until(P ≥ N−1)` 门 |
| lookup 的时间过滤（只选 `producer_id < N`） | 仍保留，但因 `P ≥ N−1` 保证 `<N` 者已全部在场，过滤只需防**回绕别名**（选 `producer_id < N`），逻辑简单 |
| retire/reuse 与发布协议配套 | 复用既有 `R = F − H` 回收（§11.4）；`P` 与 `F` 都单调，回收窗口不变，见 §10.6 |

也就是说，它把“**如何保证跨核有序可见**”这个原本含糊、易错的点，收敛为“**再加一条和 `F` 一模一样的
前沿**”，正确性论证可直接沿用 `F` 的既有结论。这实质性降低了 shared 模式的设计与验证负担。

### 10.4 无死锁 / 正确性论证

- **无死锁（DAG 流水线）。** 依赖跨度契约保证 producer id < consumer id（同一 task id 空间上的偏序，
  [§9.5/§11.4](fully_distributed_within_core.md)）。task 0 无依赖，可直接 lookup（空）并发布 → `P→0`；
  task 1 等 `P≥0` 后发布 → `P→1`；……形成**按 task id 的流水线**，不存在环，故不死锁。winner 的等待只
  依赖**更早**任务发布，而更早任务的发布不回头依赖更晚任务。
- **不阻塞执行、只序列化“登记”。** 该门只挡在 **fan-in lookup 之前**，挡的是“producer 元数据可见性”；
  任务的**实际执行**仍由完成标志 `F` 异步解耦（producer 早已 build/exec，consumer 靠 `flag(N)` 轮询）。
  因此它**不会**把执行吞吐串行化，只把“producer 注册”这一步排成 task-id 序的流水线。
- **与 `F` 的关系。** 恒有 `P ≥`（登记先于完成所需的时刻）：producer 在**提交/build 时**登记，在**执行完
  成时**才置 `flag`。故 `P` 通常领先于 `F`，`wait_until(P ≥ N−1)` 远早于 `flag` 就绪，等待代价低。
- **确定性不受影响。** `P` 只影响“何时可安全 lookup”，不改变 task id、不改变 winner 分布、不改变 map
  内容，golden 不变。

### 10.5 分片以降低原子开销（`producer_publish_cursor[G]`）

与 cursor 分片（[§6.6](fully_distributed_within_core.md)）同理，可把 `P` 扩成 `G` 个 per-shard 连续水位
`P_g`，shard `g` 只跟踪 `N ≡ g (mod G)` 子序列的连续发布前缀。但注意：**判定“所有 ≤ M 已发布”需要对
G 个 shard 取 min 覆盖**，而非任取一个：

```text
published_up_to() = min over g in [0,G) of (P_g 映射回全局 id 的覆盖上界)
wait_until(published_up_to() >= N - 1)
```

- **写竞争摊到 G 条 cache line**（发布置位/推进各打各的 shard），与 §6.6 收益一致；
- **读侧多读 G 个水位取 min**，`G` 小（如 4）时开销可忽略；
- 与 §6.6 不同的是：claim cursor 是**独立 max**、可各自推进，而这里是**连续前沿**、消费者要 min 覆盖。
  因此 `G` 不宜过大（读放大 + min 语义），单 NUMA 区间同样取 `G=4` 量级即可。

### 10.6 它解决了什么、仍缺什么（与审计 §4.4 的关系）

`producer_publish_cursor` **只解决审计 §4.3（依赖发现的发布竞态）**。审计 §4.4 指出的**输出句柄协议**是
**正交**的第二个问题，需与本机制配套才能让 shared-loser 真正全跳过：

- **句柄来源要从“本核 C++ 变量”改为“shared map 查询结果”。** 现状下游 `add_input(x)` 里的 `x` 来自上游
  `outs.get_ref(0)`；shared-loser 跳过上游 output 后本核 `x` 无效。修正：shared 模式下 codegen 把下游输入
  解析改为**按逻辑区域 / 符号句柄查 shared map**（§8 的 T3），winner 在 `P ≥ N−1` 后 lookup 得到的 map
  条目**携带完整输出描述符**（地址/尺寸/形状），据此重建句柄——**不依赖本核是否 build 过上游**。
- **发布集合要含 fresh `OUTPUT`。** 现状 fresh output 不进 map，靠输入 Tensor 的 `owner_task_id` 直传
  （审计 §3.2）。shared-loser 跳过后这条直传断链，故 shared 模式必须把 **fresh OUTPUT 的 producer 描述符
  也发布进 shared map**（纳入 `publish_producers(N)` 与 `P` 的覆盖），下游才能查到。
- **确定性地址不是纯 task id 函数（审计 §4.4 修正的事实）。** 地址还依赖此前所有输出大小/对齐/wrap 与
  `heap_next`。因此“地址 = f(task_id)”不能作为 shared 全跳的依据；正确做法是**把描述符随 producer 一起
  发布到 shared map**，由 lookup 取回，而不是各核各自重算。

**结论：** `P` 是让 shared 模式**依赖发现**正确且低复杂度的关键一块，但要让 shared-loser 全跳过，还需
配套 (a) fresh OUTPUT 纳入发布、(b) 下游按 shared map 解析句柄（含完整描述符）、(c) 全局/共享的输出分配
使描述符可被单点发布。三者齐备后，shared-loser 才能既跳过参数构造、又跳过本地 output 物化。

### 10.7 性能分析

- **聚合登记工作量：O(N×cores) → O(N)。** private 模式每个核都重放完整 map（每 producer 被登记
  `cores` 次，正是 §6.4 优化的对象）；shared + `P` 模式下每个 producer **只被其 winner 登记一次**，全局
  O(N)。核越多，这一项相对 private 的节省越大——与“随核数摊销”（§6.2）同向。
- **新增代价：发布前沿的流水线耦合 + shared map 竞争。** `wait_until(P ≥ N−1)` 把 producer 登记排成
  task-id 序流水线；由于登记很快且早于执行完成，稳态下等待通常已满足（`P` 领先 `F`）。真正的风险是
  **shared map 本身的并发读写竞争**（多核查同一份 map）+ `P` 的原子——前者靠 map 结构（分桶/分片）与
  §10.5 的 `P` 分片缓解。
- **不重蹈被否决的 per-task 阻塞。** 这与 §3.2 里被否决的“follower 走位等 cube”不同：那是把**执行吞吐**
  绑死在另一类核上；这里只挡**producer 可见性**、且是 winner 自己在 lookup 前的一次轻量等待，执行仍由
  `F` 异步解耦。
- **需实测确认（呼应审计 §5、§7.2）。** 审计已指出当前 PA 的 param-setup 占比证据不足、收益应按个位数到
  低双位数百分比预期。shared + `P` 的净收益（单份 map 省下的 O(N×cores) 登记 vs 新增的发布耦合与 map
  竞争）**必须在真实 A5 上 A/B**，尤其在高核数档，才能定论。

### 10.8 修正后的分阶段方案（吸收审计结论）

综合审计 §6/§9 的“private-first”建议与本章的 `P` 机制，推演出的落地顺序为：

1. **阶段一（private-only，先行落地）**：完全采纳审计 §6——runtime 内部/codegen 两阶段
   `prepare/finish` 接口，保留 `task id → execute-first drain → claim → 条件参数`（**不越过 execute-first
   drain**，审计 §4.2）；winner 造完整参数，private loser 只造 fresh `OUTPUT` + `INOUT` +
   `OUTPUT_EXISTING`（含物化、`heap_next` 推进、`owner_task_id`、retire）。**不引入** `P`（private 无竞态）。
   先用它拿到真实 A/B 数据。
2. **阶段二（shared 依赖发现，引入 `P`）**：在阶段一稳定后，为 shared map 增加
   `producer_publish_cursor`（可分片，§10.5）+ winner 的 `wait_until(P ≥ N−1)` 门 + fresh OUTPUT 纳入发布
   + 时间过滤。此阶段 shared-loser 可跳过**消费侧参数**，但仍需登记 producer（尚不能“全跳”）。
3. **阶段三（shared-loser 全跳过）**：补齐审计 §4.4 的输出句柄/分配协议（§10.6 的 a/b/c）——下游按 shared
   map 解析句柄、全局共享输出分配、描述符单点发布。此后 shared-loser 才能既跳参数构造、又跳本地 output
   物化，逼近“只推进 cursor + 竞争”。
4. **每阶段**都按审计 §7 做逐 task 等价校验（task id/winner 分布、`heap_next`、producer 集合、fan-in、
   payload、scope/wrap/容量边界）+ CCEC 形态与真实 A5 A/B。

> 一句话：`producer_publish_cursor` **有效**且**恰当**地把 shared 模式最棘手的“依赖发现发布竞态”降为一个
> 与完成前沿 `F` 同构的成熟机制，显著降低实现复杂性；但它**不单独**构成 shared-loser 零开销的充分条件，
> 必须与输出句柄/分配协议（审计 §4.4）配套，并按上面的分阶段推进、以真实 A/B 定论其性能。

### 10.9 修复 §4.4 输出句柄缺陷：shared-loser 的补充动作，以及 `producer_publish_cursor` 的去留

> 前置澄清：审计 §4.1–§4.2 与本文 §10.1（即“private loser 非零成本”“不能越过 execute-first
> drain”“存在发布竞态”）**不是方案缺陷**，而是对落地边界的正确约束，本文已吸收。**唯一的真实缺陷是
> 审计 §4.4——输出句柄协议缺失**。本节展开如何修复它，以及修复后是否还需要 `producer_publish_cursor`。

#### 10.9.1 缺陷精确定位：shared-loser 全跳过时断裂的两条数据流

1. **fresh-output 句柄断链。** 现状下游用 `x = outs.get_ref(0)` 拿到上游 output 的**本核 Tensor 句柄**，
   再 `add_input(x)`。shared-loser 若跳过上游 output 物化，本核 `x` 无效。
2. **地址不是 task id 的纯函数。** 现状地址由**每核确定性 bump**决定，依赖此前所有输出的大小/对齐/wrap
   与 `heap_next`（审计 §4.4）。若 loser 不再逐任务推进 `heap_next`，各核 `heap_next` 发散，**同一任务的
   输出在不同核上会算出不同地址**——不能靠“各核各自重算”得到一致地址。

#### 10.9.2 修复所需的协议动作（多数落在 winner/协议侧，loser 反而更少）

要让 shared-loser 真正“全跳过”，需要把“**本核确定性物化 + C++ 句柄穿线**”换成“**全局发布 + 查表取
句柄**”。具体补齐三件事（对应审计 §4.4、§10.6 的 a/b/c）：

| 动作 | 由谁做 | 内容 |
| ---- | ------ | ---- |
| **A. 全局输出分配** | winner（原子）+ 协议 | 用单一全局 `heap_top` 原子 bump 决定输出地址（一次性、权威），取代每核确定性 bump。地址由 winner 决定后**发布**，不再各核重算——消除 §10.9.1(2) 的发散 |
| **B. 发布完整输出描述符** | winner | 把 **fresh OUTPUT** 的完整描述符（地址/尺寸/形状/`owner_task_id`）连同 `INOUT`/`OUTPUT_EXISTING` 一并写进 shared map（现状 fresh OUTPUT 靠 `owner_task_id` 直传、不进 map，全跳后断链，故必须纳入发布） |
| **C. 下游按 shared map 解析句柄** | codegen（§8 T3）+ 消费者 winner | 下游对“引用某上游 runtime 输出”的输入，改为**按逻辑区域/符号 id 查 shared map**取回完整描述符并本地重建 Tensor——**不再依赖本核是否 build 过上游**。哪个核赢下游任务，它查表即可得句柄 |

**关键点：loser 本身并没有“新增动作”，反而更少。** 在 A/B/C 之下，shared-loser 只需推进确定性
**task-id 计数器**（`local_cursor`，仅用于身份，与输出大小无关）并参与竞争；输掉后**什么都不做**——不
物化 output、不推 `heap_next`（全局分配接管）、不写 map。真正“新增动作”的是 **winner 与协议**（全局分配
+ 发布完整描述符 + 提供查表解析）。

#### 10.9.3 修复后是否还需要 `producer_publish_cursor`？—— 仍需要，且更必要

**结论：加了 §4.4 的句柄协议后，`producer_publish_cursor`（或等价的有序发布水位）不能删除，反而更关键。**
理由是这两者作用在**不同角色、治理不同问题**，正交：

- **`producer_publish_cursor`（§4.3）治的是 winner→winner 的“producer 有序可见”**：producer 任务的 winner
  发布，consumer 任务的 winner 在 lookup 前等待。**它与 loser 做多做少完全无关**——loser 从不 publish。
  所以“给 loser 补充动作”这件事**根本不触及** `P` 的必要性。
- **§4.4 句柄协议治的是“任意核（尤其 consumer winner）如何拿到输出描述符”**：数据面/句柄重建。

而 §4.4 的修复方向（A/B/C）恰恰把**更多依赖信息路由进了 runtime shared map**：现状 fresh output 靠
`owner_task_id` **穿线**（不查表，天然无发布竞态）；全跳过后 fresh output 也必须**改为查表**（动作 B/C）。
于是原本不经过 map 的那部分依赖发现，现在也要经过 map → **发布竞态的覆盖面扩大** → `P` 不但不能删，作用
反而更吃重。

**链式（chicken-and-egg）论证，说明为何不能用完成前沿 `F` 代替 `P`：** consumer 要等 producer 的完成
标志 `flag(producer_id)` 才能读数据；但它**得先知道 `producer_id`**——而 `producer_id` 正是从 shared map
lookup 得到的。故“lookup 成功”必须发生在“等 flag”之前，`F` 无法门控 lookup 自身。门控 lookup 的只能是
**发布序 `P`**。因此 `P` 不可被 `F` 取代。

#### 10.9.4 唯一能删掉 `P` 的路径——但它与“loser 全跳过”互斥

存在**另一条设计轴**可以去掉 `P`：把**依赖发现从 runtime map 移到编译期**——

- codegen 静态解析每个消费者的 producer task id，作为 **explicit deps** 直接发到任务上（消费者无需查表找
  producer → 无发布竞态）；且
- 保留**每核确定性布局**，使输出地址可被任意核**重算**（无需查表取地址）。

在这条路径下 `P` 可以删除。**但代价是**：可重算地址要求**每个核都推进 `heap_next`**、都得算 fresh
output 的大小——即 loser **必须保留输出侧的确定性 replay**（退化为 private-loser 的输出行为），**不再“全
跳过”**；而且 producer 必须**静态可知**（数据依赖/动态形状场景难以静态定 producer）。这本质上是把“map”
从 runtime 挪到编译期。

**由此得到清晰的二选一（duality）：**

| 设计 | 依赖发现 | 输出地址 | loser 能否全跳过 | 是否需要 `P` |
| ---- | -------- | -------- | ---------------- | ------------ |
| **设计 I：runtime shared-map 查表 + 全局分配** | runtime lookup | 全局分配、发布取回 | **能**（真正 do-nothing） | **需要 `P`** |
| **设计 II：静态 explicit deps + 每核确定性布局** | 编译期静态 | 每核可重算 | **不能**（loser 仍算 output/推 heap） | **可删 `P`** |

**因此，直接回答你的问题：**

- **shared-loser 需补充哪些动作？** 若走**设计 I**（目标：loser 全跳过），loser 本身不新增动作、反而更少；
  新增动作在 winner/协议：全局分配（A）、发布完整描述符含 fresh OUTPUT（B）、下游查表解析句柄（C）。
- **补齐这些后还需要 `producer_publish_cursor` 吗？** **仍然需要，且更必要**——因为 §4.4 修复把更多依赖
  发现路由进 shared map，而 `P` 治的是 winner→winner 的发布可见性，与 loser 无关，无法被 `F` 取代。
- **能否删 `P`？** 只有改走**设计 II**（静态 deps + 每核确定性布局）才能删，但那样 loser 就**不再全跳
  过**（必须保留输出侧确定性 replay），与“shared-loser 零开销”的初衷互斥。

> 一句话：**“让 shared-loser 全跳过”与“删掉 `producer_publish_cursor`”不可兼得。** 想要 loser 真正
> do-nothing，就得靠 runtime shared map 承载依赖发现与句柄解析，从而必须保留 `P` 来保证有序可见；想删
> `P`，就得把依赖发现与地址计算做成编译期静态 + 每核可重算，那 loser 又回到要做输出侧确定性 replay。
> §4.4 的句柄协议修的是“怎么拿到句柄”，`P` 修的是“何时查表安全”，二者正交，修前者不能省后者。

**决定：采用设计 I。** 既然目标是让 shared-loser 真正 do-nothing、并把每核 O(N×cores) 的 map 维护降到
全局 O(N)，就接受“保留 `producer_publish_cursor`”这一代价。**§11 给出设计 I 的完整规范**（数据结构、提交
协议、全局分配、发布与查表、句柄重建、回收、分片、正确性与 loser 快路径成本）。设计 II 作为“无 `P` 但
loser 不全跳”的备选，仅在此登记，不再展开。

## 11. Shared TensorMap 模式完整设计（采用设计 I）

本章是 §10.9 选定的**设计 I（runtime shared-map 查表 + 全局分配 + `producer_publish_cursor`）**的完整落地
规范，仅适用于 `tensormap == shared` 模式；`private` 模式仍按 §3/§4 的“winner 造全参、private-loser 造输出侧
参数”执行，不使用本章的全局分配与 `P`。设计 I 的核心目标：**让 shared-loser 真正 do-nothing**（不求值任何
参数 thunk、不物化 output、不推进任何全局状态），把 producer 登记从每核 O(N×cores) 降到全局 O(N)。

### 11.1 数据结构

| 结构 | 归属 / 类别 | 作用 | 相对 private 的变化 |
| ---- | ----------- | ---- | ------------------- |
| **shared TensorMap** | 全局单份（分桶/分片，winner 写、任意核读） | region / 符号 id → producer 描述符 `{owner_task_id, addr, size, shape, dtype, …}` | 取代 private 的**每核复制 map** |
| **global heap allocator** `heap_top` | 全局单个原子 bump（可分 arena，见 §11.6） | 决定 fresh `OUTPUT` 的权威地址；配 `R=F−H` 回收 | 取代 private 的**每核确定性 `heap_next` bump** |
| **`producer_publish_cursor` `P`**（可分片 `P_g`，§10.5） | 全局连续前沿 + 协作 CAS | “所有 id ≤ `P` 的 producer 元数据已发布进 shared map”的水位线 | **新增**（private 无此需求） |
| **完成前沿 `F`、回收水位 `R=F−H`** | 全局（沿用 [§11.4](fully_distributed_within_core.md)） | 执行完成与地址回收 | 不变 |
| **`local_cursor`（每核 task-id 计数器）** | 每核私有 | 仅承载**确定性任务身份**，与输出大小无关 | 不变（§3.1） |
| **private task ring / lane_inbox** | 每核私有（执行结构） | winner 的 build/execute/check | 不变 |
| **`SubmitBuilder` 惰性 thunk 清单** | 每次提交 | input/output/inout/scalar/deps 全部惰性登记（§3.3、§3.4） | 不变 |

要点：shared 模式把**两处“每核各算一份”的结构**（复制 map、确定性 heap）替换为**全局单份 + 发布/查表**，
这正是 loser 能全跳过、且聚合工作量降到 O(N) 的结构性来源。

### 11.2 提交协议（shared 模式，融合 execute-first drain + compete-first + 全局分配 + 发布门控）

```text
submit_shared(kernel_id, build_callback):        # tensormap == shared
    N = ++local_cursor                           # (1) 确定性任务身份，纯计数、无参数求值
    drain_lane_inbox(); drain_phase_b()          # (2) execute-first drain，先于 claim（审计 §4.2）
    win = compete_cursor(type(kernel_id), N)      # (3) claim == compete_cursor（§3.1）

    if not win:                                   # ---- shared-loser：真正 do-nothing ----
        return symbolic_handles(N, kernel_id)     #      仅返回符号句柄（§11.3），不求值任何 thunk

    # ---- winner 路径 ----
    args = eval_all_thunks(build_callback)        # (4) 求值 input/output/inout/scalar/deps 全部 thunk
    for o in fresh_outputs(args):                 # (5) A. 全局输出分配（权威地址）
        o.addr = atomic_bump(heap_top, o.size, o.align)   # heap 满则等 R 前进（背压，§11.5）
    publish_producers(N, args)                    # (6) B. 发布完整描述符 + 推进 P（先发布，后查表）
    wait_until(load_acquire(P) >= N - 1)          # (7) 发布门：历史 producer 全部可见后才查表
    fanin = shared_map.lookup(consumed(args),     # (8) 依赖发现：过滤 owner_task_id < N（防回绕别名）
                              filter=owner_id < N)
    build_task_into_ring(N, args, fanin)          # (9) 建 private 环槽、发起执行；完成时置 flag(N)→推进 F
    return concrete_handles(args)                 # winner 手里已有真实句柄
```

关键顺序说明：

- **(6) 发布先于 (7) 门等待。** winner 拿到全局地址后即可发布本任务 producer（`published[N]` 置位并协作推进
  `P`），**不必**等自己的 fan-in。这样即便本 winner 随后卡在 (7) 等 `P≥N−1`，它的 `N` 也早已可见，不会
  阻塞下游对 `N` 的消费。
- **(7) 门只挡 lookup，不挡执行。** 挡的是“producer 元数据可见性”，执行仍由 `flag(N)`/`F` 异步解耦
  （§10.4）。稳态下 `P` 领先 `F`，等待通常已满足。
- **(2) execute-first drain 不可越过。** 竞争前必须先把已认领的活干掉，否则 ring 反压/死锁（审计 §4.2）。

`publish_producers` 的形态（沿用 §10.2，扩为发布**完整**描述符，含 fresh OUTPUT）：

```text
publish_producers(N, args):
    for p in outputs(args) + inouts(args) + output_existing(args):
        shared_map.insert(key(p), {owner_task_id: N, addr: p.addr,
                                   size: p.size, shape: p.shape, dtype: p.dtype})
    store_release(published[N], true)
    while load(published[P+1]):                  # 任意核协作推进，开销摊薄
        CAS(P, P, P+1)
```

### 11.3 输出句柄的获取（符号句柄 + 查表重建，修复审计 §4.4）

设计 I 下**下游不再穿线 C++ 句柄**，而是由符号身份 + shared-map 查表解析：

- **符号句柄（loser 也能零成本产出）。** `submit_shared` 对任意角色都返回
  `SymbolicTensor{producer_task_id=N, output_slot=i}`。`N` 由 `local_cursor` 确定性得到、`i` 是输出序号，
  **都不依赖地址或是否物化**，故 shared-loser 跳过一切后仍能返回合法符号句柄。
- **shared map 的键。** fresh `OUTPUT` 以**符号身份 `{N, slot}`** 为键（loser 无法算地址，只能按符号发布/
  查表）；`INOUT`/`OUTPUT_EXISTING` 以既有 region（地址已知）为键，另建 region 索引供 view/子区重叠的
  依赖发现使用。两类都纳入 `P` 覆盖。
- **使用即解析（lazy resolve on use）。** 当下游任务 `M` 的 winner 求值某 input thunk、其中引用了上游的
  `SymbolicTensor{N, i}`（`N < M`）时，runtime 对该符号做 `shared_map.lookup({N,i})` 取回完整描述符并本地
  重建真实 `Tensor`。该解析**复用 (7) 的同一发布门**：`M` 的 winner 在 lookup 前已 `wait_until(P≥M−1)`，而
  `N ≤ M−1`，故 `{N,i}` 必已发布——**一道门同时覆盖 fan-in 发现与符号输入解析**。
- **winner 短路。** winner 自己刚分配、手里已有真实句柄，可短路跳过查表直接用；codegen 为保持形态统一
  也可一律走符号解析（多一次 winner 侧 lookup，语义等价）。

由此，审计 §4.4 的两条断裂（fresh-output 句柄断链、地址非 task-id 纯函数）都被闭合：句柄来源改为
“**符号身份 → 查发布后的 shared map → 完整描述符**”，与“本核是否 build 过上游”彻底解耦。

### 11.4 依赖发现与发布门控

沿用 §10.2 的机制，在 shared 上下文中即协议 (6)(7)(8)：**发布先行、`P` 连续前沿、winner lookup 前
`wait_until(P≥N−1)`**。时间过滤仅需 `owner_task_id < N` 以防回绕别名（`P≥N−1` 已保证 `<N` 者全部在场）。
无死锁与不阻塞执行的论证见 §10.4。

### 11.5 全局分配与回收

- **分配。** fresh `OUTPUT` 地址由全局 `heap_top` 原子 bump 决定（唯一、权威），winner 分配后随描述符发布；
  下游取回地址而非各核重算——消除 §10.9.1(2) 的地址发散。
- **回收。** 复用 `R = F − H`（[§11.4](fully_distributed_within_core.md)）：`owner_task_id ≤ R` 的输出区域
  可回收；同时**按同一 `R` retire shared map 条目**（键为 `owner_task_id`），使 map 体积有界——等价于
  private 的 `cleanup_retired`（§6.4），只是作用在单份 shared map 上。
- **背压。** 全局 heap 满时，想分配的 winner 等待 `R` 前进（同 [§9.5](fully_distributed_within_core.md)
  语义），不额外引入死锁面。

### 11.6 分片以降低原子竞争

- **`P` 分片 `P_g`。** 见 §10.5：判定“≤M 已发布”需对 `G` 个 shard 取 min 覆盖，`G` 取 4 量级。
- **`heap_top` 分片。** 设计 I 的地址**不要求确定性**（靠发布取回），故可把全局 heap 拆成**每 shard/每 lane
  子 arena**并行 bump，去掉单点原子热点；每个子 arena 各自 `R` 回收。这是纯性能旋钮，不影响正确性。
- **shared map。** 分桶 + 分片以支撑多核并发读写；键（`{N,slot}` 与 region）各自散列。

### 11.7 正确性与不变量

- **确定性身份。** task id、winner 分布不受 `P`/分配策略影响（§10.4）；golden 只比较逻辑结果，不比较物理
  地址（地址在 shared 模式本就是发布量）。
- **无漏依赖。** 发布先于 lookup + `P` 门（§10.4、§11.4）。
- **无死锁。** 发布链沿 task-id 偏序（producer id < consumer id）成流水线，无环；执行由 `F` 解耦（§10.4）。
- **地址一致。** 每个 fresh output 由其 winner **恰好分配一次**、地址权威发布，所有消费者读同一描述符——
  跨核无发散。
- **loser 真 do-nothing。** shared-loser 只推进 `local_cursor` 并竞争，不触碰 map/heap/flag/`P`；因此不影响
  上述任何不变量。
- **执行语义不变。** `flag(N)`、`F`、`R` 与 [§11.4](fully_distributed_within_core.md) 完全一致。

### 11.8 loser 快路径成本（设计 I 达成的目标）

shared-loser 的稳态成本 = `++local_cursor`（一次计数）+ execute-first drain（干的是**已认领**的活，不计入本
任务开销）+ `compete_cursor`（一次原子竞争，输）+ 返回符号句柄。**不求值任何 thunk、不物化 output、不推
`heap_top`、不写 map、不碰 `P`**——即 §3/§10 追求的“接近只推进 cursor + 竞争”的零开销 replay。

### 11.9 与 private 模式的对照与选择

| 维度 | private（每核复制） | shared（设计 I） |
| ---- | ------------------- | ---------------- |
| TensorMap | 每核一份 | 全局单份（分片） |
| 输出地址 | 每核确定性 bump（可重算） | 全局分配 + 发布取回 |
| 依赖发现 | 本核顺序 replay 保证可见 | shared map lookup + `P` 门 |
| 句柄来源 | 本核 C++ 句柄穿线 | 符号身份 + 查表重建 |
| loser 成本 | 需造输出侧参数（fresh OUTPUT/INOUT/物化/推 heap/owner/retire） | **全跳过** |
| 聚合登记工作量 | O(N×cores) | O(N) |
| 额外机制 | 无 | `producer_publish_cursor` + shared map 并发 |

codegen 对两种模式**发出同一份 `SubmitBuilder` 惰性清单**（§8）；差异全在 runtime 侧的 replay 策略与句柄
解析方式（穿线 vs 符号查表）。runtime 依部署选择模式，编排/codegen 无需分叉。

### 11.10 与 §10.8 分阶段方案的对齐

§10.8 的**阶段三**即本章设计 I：在阶段二（引入 `P` + fresh OUTPUT 纳入发布 + winner 门）之上，补齐
§11.3 的符号句柄/查表解析（审计 §4.4 的 C）与 §11.5 的全局分配（A），shared-loser 方可全跳过。每阶段仍按
审计 §7 做逐 task 等价校验（task id/winner 分布、fan-in、producer 集合、payload）与真实 A5 A/B——设计 I 的净
收益（省下的 O(N×cores) 登记 vs 新增的发布耦合与 shared map 竞争）必须实测定论（§10.7、审计 §5/§7.2）。
