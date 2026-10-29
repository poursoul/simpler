# A5 HBG resident SPMD 中文设计评审

评审结论：保留现有 ready/wake 路由、FIFO 数据结构和 Mix 两级完成聚合的方向；
在开放 SPMD/sync_start 前，先修正条件退队、远端执行责任发布和跨执行通道的顺序保证。

审阅分支为 poursoul/simpler 的 a5-hbg-resident-spmd，固定提交
`1a4d30630d6309f41bfe07ae497708d99a53a476`，评审日期为 2026-10-23。
[设计原文](https://github.com/poursoul/simpler/blob/1a4d30630d6309f41bfe07ae497708d99a53a476/tests/docs/a5-hbg-resident-spmd-design.md)
与下文源码链接均指向该提交，避免后续更新改变评审依据。
同时对照了 nalinaly/simpler 的 feat/fully_distributed_within_core@f5cc679bb。

这是对设计稿计划中的 S1 协议及其引用原语的评审，不把 S0 当作已经开放 SPMD。
反例来自源码核对及协议级确定性交错，未进行 A5 真机复现。

## 值得保留的设计

- 复用完成驱动的 ready/wake 路由，不为 SPMD 重建依赖系统。
- 复用 FIFO 和 task 级领取游标，减少重复的共享状态。
- Mix 的各 lane 先由本地 tracker 聚合，再对逻辑 task 贡献一次 block 完成，
  不把全局原子数量放大为 block 数乘 lane 数。
- 单 block 使用真实的 block_idx 为 0、block_num 为 1，不引入无必要的哨兵。
- slot 携带完成路径必需的信息，避免重复读 metadata；无 SPMD 的图不增加队首轮询。
- 区分队列类型与物理核类型，并关注 local state 膨胀带来的栈和寄存器压力。
- 用 resident 来源与完整 block/subtask 身份验收，不把 golden 通过当成新调度路径已被执行。

## 必须修正的正确性协议

### 1. 必须修改：延迟出队不能直接调用无目标约束的通用 pop

位置：[设计 §6.3](https://github.com/poursoul/simpler/blob/1a4d30630d6309f41bfe07ae497708d99a53a476/tests/docs/a5-hbg-resident-spmd-design.md#L381-L407)，[scheduler_ready_pop_from_inbox](https://github.com/poursoul/simpler/blob/1a4d30630d6309f41bfe07ae497708d99a53a476/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h#L1156-L1218)。

现有 pop 的语义是移除“当前队首”。CAS 失败后，它会重新读取当前 endpoints 并重试，不保证仍在移除调用方刚刚检查过的那个 task。

可构造以下时序：

1. 队列为 A → B → C；A 的 block 已全部领取，B 尚未领取。
2. X、Y 都观察到 A 已耗尽，准备帮助推进队首。
3. X 调用 pop，移除 A。
4. Y 调用 pop，直接读取 B；或者使用旧快照 CAS 失败后，重新读取 B。
5. Y 成功移除 B，尽管 B 的 block 尚未分发。

CAS 保证队列结构修改原子性，不会自动保留“A 已耗尽”这一业务前提。这里并不需要 ABA 就会出错。

建议引入带 `expected_task_ref` 的条件推进：每次 CAS 重试都重新确认 head 仍是原 task，head 已变则返回“已由其他核推进”，不得继续 pop 新队首。不能在 pop 成功后再检查返回值，因为那时 B 已脱链。

请补两个帮助者同时观察 A 耗尽的确定性交错测试，断言 B 仍保留且最终执行完整。

### 2. 必须修改：领取游标到 N，不等于 Mix 的远端执行责任已经建立

位置：[设计 §6.2–§6.3](https://github.com/poursoul/simpler/blob/1a4d30630d6309f41bfe07ae497708d99a53a476/tests/docs/a5-hbg-resident-spmd-design.md#L319-L404)，[Mix placement 探测](https://github.com/poursoul/simpler/blob/1a4d30630d6309f41bfe07ae497708d99a53a476/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h#L1100-L1153)。

当前计划的步骤为：探测 placement → CAS 领取 → 准备 payload → 向 AIV 发布。同时又允许其他核看到 `next_block == N` 后帮助推进队首。

这会留下一个独立于问题 1 的窗口。即使条件出队已经修好，仍可发生：

1. A 为 Mix cohort，已有参与者进入 A 并等待其余参与者。
2. C1 成功领取 A 的最后一个 block 后暂停，尚未向目标 V1 发布 A 的执行责任。
3. 另一个空闲 AIC C2 看到 A 已全部领取，帮助推进队首到独立的 AIV cohort B。
4. V1 尚看不到 A，领取并进入 B；B 等待仍在 A 内的 V0。
5. C1 恢复并发布 A，但 V1 已在 B 内；A 又等待 V1 加入，形成循环等待。

这里可使用三个 cluster：第三个 AIC 只帮助推进队首，A 只需要其中两个 cluster；B 的其他 AIV 参与者正常进入后仍会缺少 V0。

placement 探测确认的是协调者所拥有的可写 dispatch 槽，并不等价于目标 AIV 已承诺按顺序执行该 block。

需要明确“领取成功”“执行责任建立”“完整发布”之间的合同。例如让目标核能识别尚未到达 payload 的前序义务，或只有在必要调度记录完整发布后才允许相关后继越过。具体机制可以讨论，但不能只用 `next_block == N` 作为全部证明。

修复不应改成等待 A 全部执行完成，也不需要全员同时到达；需要保证的是后序任务无法越过已经分配的 A 义务。

### 3. 必须修改：本核 generation 顺序不能保证普通与 Mix 两条通道同序

位置：[设计 §8.1–§8.2](https://github.com/poursoul/simpler/blob/1a4d30630d6309f41bfe07ae497708d99a53a476/tests/docs/a5-hbg-resident-spmd-design.md#L539-L614)，[两套 generation](https://github.com/poursoul/simpler/blob/1a4d30630d6309f41bfe07ae497708d99a53a476/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h#L294-L296)，[普通及 Mix 执行入口](https://github.com/poursoul/simpler/blob/1a4d30630d6309f41bfe07ae497708d99a53a476/src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp#L366-L494)。

“先领取 A、后领取 B，所以 generation 更大，必然先执行 A”的证明仅覆盖同一套本核 slot。

现有代码中：

- 普通任务在本核 `slots`；
- 远端 Mix 子任务在独立 Mix 通道；
- 两者使用不同的 generation 序列；
- 执行器先取普通 slot，有工作就执行并 continue，之后才扫描 Mix。

即使 A 的所有 payload 都已发布，也可以出现：

- V0 已进入 Mix cohort A，等待 V1 加入 A；
- V1 的 Mix 通道已有 A，但本核通道领取了较后的纯 AIV cohort B；
- V1 优先进入 B，等待仍在 A 内的 V0。

可用两个完整 cluster 构造：A 为两个 Mix block，B 为四个纯 AIV block，均不超过实际容量；其余参与者进入对应任务后，仍无法解除 V0/V1 的互等。

因此，Mix 让行不能无条件归类为“只影响延迟、不影响正确性”。修复也不是简单改成“永远 Mix 优先”：较早纯 AIV A、较后 Mix B 的反向组合同样需要正确顺序。

需要跨普通／Mix 通道检查本核最早未履行的义务，或者提供等价的统一执行顺序机制。请至少覆盖两个方向、部分参与集合以及满额 cohort。

## 需要澄清的取舍和文档结论

### 4. 需要明确的设计取舍：单一 SPMD 队首会阻塞不相交的核类型

位置：[§4.2–§4.5](https://github.com/poursoul/simpler/blob/1a4d30630d6309f41bfe07ae497708d99a53a476/tests/docs/a5-hbg-resident-spmd-design.md#L138-L227)、[§12](https://github.com/poursoul/simpler/blob/1a4d30630d6309f41bfe07ae497708d99a53a476/tests/docs/a5-hbg-resident-spmd-design.md#L862-L886)。

例如队首 A 是尚未领完的纯 AIC 大任务，后面 B 是已 ready 的独立 AIV SPMD。AIV 可以继续执行普通单 block，但不能领取 B。

这不是在“一次只允许一个 SPMD task 被领取”的约定下的隐藏 bug，而是该约定的明确限制。若目标包含“不参与 A 且无资源依赖的核可以执行后续独立任务”，当前单队首机制没有满足它。

请在范围和验收中明确这项限制，不要把“没有全局完成屏障”进一步表述为“所有非参与核均不受影响”。也应区分领取串行与执行串行：队首在全部领取后让位，并不必然把两个 task 的全部执行严格串行化。

### 5. 建议修正内存及性能论证的适用范围

位置：[§3.7–§3.8](https://github.com/poursoul/simpler/blob/1a4d30630d6309f41bfe07ae497708d99a53a476/tests/docs/a5-hbg-resident-spmd-design.md#L68-L81)、[§5.2](https://github.com/poursoul/simpler/blob/1a4d30630d6309f41bfe07ae497708d99a53a476/tests/docs/a5-hbg-resident-spmd-design.md#L239-L266)、[§11](https://github.com/poursoul/simpler/blob/1a4d30630d6309f41bfe07ae497708d99a53a476/tests/docs/a5-hbg-resident-spmd-design.md#L847-L860)。

- 原子字段不能混入会被其他核整行回写的缓存行，这一约束正确；但“只能放 TaskControl 第一条线”是当前布局下的实现约束，不是硬件只允许第一条线。独立且不会被普通缓存写回覆盖的区域仍是可选方案，是否增加空间需要另行权衡。
- 绕过 DCache、观察原子更新、取得 payload 可见性是不同问题。同分支的 [scheduler_memory.h](https://github.com/poursoul/simpler/blob/1a4d30630d6309f41bfe07ae497708d99a53a476/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_memory.h#L114-L143) 已将 `scheduler_gm_query` 定义为观察／预筛选，并为关键链接使用 `atomicAdd(0)`。新增字段需要逐项说明读取和发布合同。
- §11 引用 ascend.md 说明 GM `ld_dev` 严格串行，但该规则实际描述的是 [AICPU 对 MMIO COND 的 LDR](https://github.com/poursoul/simpler/blob/1a4d30630d6309f41bfe07ae497708d99a53a476/.claude/rules/ascend.md#L34-L39)，不能直接用作 AICore 读取 GM 的性能依据。
- 按文档的 32 AIC＋64 AIV 配置，纯 AIC cursor 的竞争者最多为 32，纯 AIV 最多为 64，Mix 则是 32 个 AIC 协调者。96 个核观察队首，不等于 96 个核都 CAS 同一个 block cursor。
- 每次领取一个 block 可以作为首版固定策略，但全局 CAS 成本、重试以及领取／完成计数之间的干扰仍需实测，不能仅以“现有 ready 已使用 CAS”推定代价相同。

### 6. 必须消除文档中的 mode 4 矛盾

位置：[§9](https://github.com/poursoul/simpler/blob/1a4d30630d6309f41bfe07ae497708d99a53a476/tests/docs/a5-hbg-resident-spmd-design.md#L619-L669)、[阶段表](https://github.com/poursoul/simpler/blob/1a4d30630d6309f41bfe07ae497708d99a53a476/tests/docs/a5-hbg-resident-spmd-design.md#L688-L709)、[§10.4](https://github.com/poursoul/simpler/blob/1a4d30630d6309f41bfe07ae497708d99a53a476/tests/docs/a5-hbg-resident-spmd-design.md#L766-L783)。

§9 已要求删除 mode 4、超容量 cohort 硬失败；但阶段表仍保留更换 mode 4 判据，§10.4 又要求把测试改成超容量 cohort 回退 mode 4。按不同章节实施会得到相反行为，请统一。

另外，应限定“AICore 调度无法运行，AICPU 也无法运行”的适用范围：资源上无法满足共同驻留，与 resident 尚未支持某种合法形状，是两类不同情况。前者不应靠回退掩盖，后者不能由同一句话排除。

## 修改与验收建议

建议保留现有 ready/wake、FIFO 数据结构、单 block 0/1 表达及 Mix 两级完成聚合，但在开放门禁前完成以下验收：

- [ ] 条件队首推进：两个帮助者同时处理耗尽的 A，不得移除尚未领取的 B。
- [ ] 发布窗口：暂停最后一个领取者于 CAS 成功后、远端发布前，其他核不能借后序 cohort 破坏前序义务。
- [ ] 跨通道顺序：同时测试 Mix A → 纯 AIV B，以及纯 AIV A → Mix B。
- [ ] 快核可提前推进：无真实数据依赖时，存在 B 开始早于 A 最后一个 block 返回的合法记录；不能用全局完成屏障修复上述问题。
- [ ] 完成唯一性：非末 block 不释放后继，最后 block 只发布一次逻辑完成；逐项核对 block/subtask 身份，不只检查总数。
- [ ] 核类型隔离：用独立 AIC/AIV SPMD 明确验证或记录单队首限制。
- [ ] 文档、能力门禁与 mode 测试使用一致口径。

建议先修正以上协议缺口及证明，再进入相应功能验收。尤其不要将“领取顺序一致”直接等同于“所有执行通道顺序一致”，也不要将 FIFO 的结构正确性等同于延迟退队的业务正确性。

## 验证边界

- 本文是设计评审，不声称已经在 A5 真机复现上述卡死。
- 复现依据是实际 pop 重试逻辑、普通／Mix 独立执行通道，以及设计允许的确定性交错。
- S0 基础设施测试通过不等同于尚未开放的 S1 领取、发布及 cohort 活性已经被验证。
- 不要求引入全局 task 完成屏障，也不要求照搬 TMR 的全局 drain／ack 树；修复应保持逐核推进和合法的跨 task 重叠。
