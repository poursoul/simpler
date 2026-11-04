# A5 HBG 异步输出的消费准备

[返回总览](../a5-hbg-async-completion.md)。本文定义从完成条件所保护的 payload 到实际读取核的映射，以及消费者取得执行许可之前的数据获取协议。它适用于同步和异步消费者，独立于消费者的 `may_defer`。

## 1. 输入合同与消费计划

`CoreAsyncCapability` 只描述该 kernel 是否可能登记完成条件。消费者即使 `may_defer=false`、没有 group/join，也可能读取异步产生的数据。TaskDone 只保证发布侧完成；FINISH 或 Graph control 路由依赖时不能替消费者完成本核缓存获取。

Host 使用三种信息构造消费计划：参数及 predicate operand 的读写/别名关系、当前 run 的资源绑定、backend 提供的完成条件与 payload 使用关系。条件绑定指向 signal/workspace；payload 绑定指向实际数据窗口，两者不能混为一个地址范围。

在 `AsyncUseSet` 中为每项异步发布用途建立不可变的 `AsyncPublishedUse`，至少包含：

| 字段组 | 内容 |
| ------ | ---- |
| 发布身份 | `TASK_OUTPUT` 来源用 run、稳定 producer task ref、用途索引组成 `publication_id`；对应一版数据，区别于 allocation generation |
| 完成责任 | `TASK_OUTPUT` 允许保护该用途的 group/条件绑定集合；producer 的正常 TaskDone 是统一完成边界 |
| payload | payload binding index、资源 generation、窗口内 offset/bytes；源、目标和 signal 的角色明确分开 |
| 获取要求 | backend 可见性策略、允许的读取核/读取路径、实际 cache line 粒度和范围所有权要求 |

SDMA 的目标参数与 workspace 完成记录由 adapter 的已验证描述关联；COUNTER waiter 通过显式用途声明关联通知与 payload 参数。PyPTO 从通信操作和参数谱系产生声明，原生 C++ 使用等价的绑定入口。纯控制通知可以显式声明没有 payload；不能将缺失声明当成纯控制通知。

对 launch 前已经就绪、但实际读取路径仍需本核获取的外部输入，使用 `READY_INPUT` 来源：身份由入口参数的实际 owner/binding generation 和本次输入发布版本组成，不伪造本 run 的 producer TaskId。Host 只能依据既有入口提交协议提供的就绪及版本稳定证明接受它，设备不新增外部事件或 CHECK 等待；没有该证明时拒绝。其范围和准备规则与任务输出一致，因此全 run 同步也可以只分配消费准备区。

允许声明覆盖多个运行分支的静态安全上界。提交/登记时必须验证实际传输和被保护范围落在该用途内，不允许 kernel 临时声明 Host 不知道的新 payload。条件条数、窗口大小和 signal 地址都不能单独推导出消费范围。

消费计划包含以下 run 内不可变记录：

| 记录 | 合同 |
| ---- | ---- |
| `ConsumerAcquirePlan` | 以 `(task_ref, block_idx, subtask)` 定位，保存有界条目区间；无获取要求时使用显式空计划 |
| `ConsumerAcquireRange` | `publication_id`、payload binding index/generation、范围、读取阶段（predicate/body）、可见性策略及读取路径资格；多个合法别名可引用同一条目 |

相同计划可共享只读条目，但执行进度不能共享。SPMD 无法静态细分 block 范围时，可采用该参数已验证且允许维护的完整范围作为上界；不能扩大到不属于本次使用的窗口或整份 cache。新类型采用[接口文档](interfaces-and-layout.md)的版本、offset/count、布局和溢出检查合同。

## 2. Host 构造与 Graph 传递

bind 在持有资源使用、展开实际任务后执行以下步骤：

1. 从入口参数 owner 身份建立谱系，derived view 保留 generation 与检查后的范围；按 run 内实际参数解析 alias。未知 owner、歧义别名和越界在 launch 前拒绝。
2. 用 tensor 读写关系和已有 DAG 确定每个读取范围所对应的数据发布版本。控制依赖只证明顺序，不能凭一条依赖边猜测 payload。同一范围存在未排序的写者或互相矛盾的发布版本时拒绝，不能取一个最大 counter 代替证明。
3. 对读取或 read-modify-write 的实际 kernel 参数，以及独立的 predicate operand，把相关 `AsyncPublishedUse` 与读取范围相交，生成每个物理消费者的计划。predicate 可能只存在于 task/Graph 的 predicate 描述符而不在 body 参数中，仍须从其原始 Tensor/Graph 谱系取得 owner、范围及发布版本，不能仅扫描 kernel args 或凭最终裸地址猜测。按资源 generation、publication、策略和读取路径去重；不同版本或不兼容策略不能合并。纯写参数的提交前 dirty-cache 处理仍属于发布侧合同。
4. 将范围按 backend 实际维护粒度扩展，检查整个 cache line 的所有权与共享限制；非连续视图使用有界范围表或经过证明的安全包络。表大小、初始化/H2D 字节与执行准备状态一起纳入预算。
5. 验证 TASK_OUTPUT 用途能沿真实依赖到达消费者，READY_INPUT 则验证入口就绪证明；下一写者都须受到已有 DAG 或 peer 使用协议约束。消费计划不新增隐式依赖边、不提前等待完成条件，也不允许下一 epoch 在准备与读取之间覆写数据。

多 fanin 消费者合并所有相关读取范围的要求，依赖已就绪不意味着只需获取最后完成的 fanin。中间同步任务若产生新的数据版本，按其真实输出谱系处理；传递别名的节点不凭空终止原发布用途。

Graph Definition 只保存参数/内部值的逻辑谱系。每个 invocation 按 `PARAM → boundary binding`、`SUB_TASK → 本实例 heap/输出 binding` 解析，运行期 generation 和准备状态不写入可复用 Definition。ENTER/EXIT 不执行 payload 获取，跨它们传播的是数据谱系和依赖顺序，最终计划落在读取 payload 的业务 kernel 上。

例如 `A 异步写 X → Graph EXIT → 同步 C 读 X`：C 的计划包含 X 的发布身份、范围和获取策略，即使 C 没有 async group。另一个 Graph invocation 使用同一 Definition 时必须重新绑定，不能沿用第一个实例的资源代次或准备结果。

## 3. 本核准备与执行许可

`consumer_prepare_step(plan, execution, cursor, budget)` 返回 `READY`、`YIELD` 或 `ERROR`。调用前该任务的真实依赖已经就绪，参数地址/布局已按当前执行物化，但尚未读取需要获取的 payload；它只做有限的数据获取步骤，不等待 DMA 或 peer 满足条件。每步维护一个声明粒度的范围，并计入普通服务工作额度；较大的范围通过持久 cursor 续接。

predicate 求值也是 payload 读取。dispatch 物化须拆开“地址/布局准备”和“读取 predicate”：先完成 predicate 范围获取再求值；true 时继续 body 范围准备，skip 时直接进入既有 runtime 尾部。不能在原物化 helper 已读过 predicate 之后才增加获取 hook。

`ConsumerPrepareState` 属于预备执行槽，绑定完整的 run、task/block/subtask、plan、worker/core、slot generation，以及适用的 cohort/group sequence。状态为 `UNPREPARED → PREPARING → PREPARED → ENTERED`；错误进入 run 失败。cursor 在有界步骤之间保存，READY 只有在最后范围及所需屏障完成后才成立。

准备结果只在同一核、同一槽代次和同一数据发布版本内有效。slot 回收/复用、映射改变或重新分配执行核都必须清除旧结果并重新准备；不能把另一核的 PREPARED 当成可移交 lease。已公布的 cohort 准备结果不能由普通调度悄悄撤回，成员身份保持到该 cohort 执行释放或整个 run 错误退出。

普通候选准备完成但 admission 尚未允许时，不阻塞更早的 cooperative obligation。若本核先执行了另一业务 kernel，保守清除这些普通候选的准备结果，下一次尝试前重新获取；不能只凭 slot generation 未变沿用它们。cohort 成员在公布 prepared ack 后则保持绑定，不插入其他业务 kernel。

| 入口 | 准备与最终许可的接线 |
| ---- | -------------------- |
| 普通单 lane / 非 sync-start SPMD | 每个物理执行在自己的候选 slot 上分段准备；只有 PREPARED 才调用最终 entry gate，仍须满足原 admission/error 条件 |
| MIX | 每个实际 active lane 在本核准备完成后发布带完整身份的 prepared ack；coordinator 收齐本 block 的全部 active lane 后发布该 group 的 start 标记 |
| sync-start | 先按既有容量/成员规则确定整个 cohort；所有实际 block/lane 的准备结果由唯一协调者汇合，全部到齐后发布 cohort start；MIX sync-start 覆盖全部 block 的全部 active lane |
| predicate | predicate 自身若读取异步 payload，该读取也必须先完成相应获取；合法 skip 仍走既有零登记/释放尾部，不进入业务 kernel |

现有 admission record 的 `ADMITTED`、MIX 的 arrived ack、payload materialized 或 local READY 仅证明各自原有事实，均不替代 prepared ack。coordinator 可以复用已有唯一角色；sync-start 增加任务级唯一汇合职责，不将每个 block 的局部到齐误当全体到齐。inactive lane 不贡献虚假的准备结果，也不能拿其既有 release ack 代替 active lane 的准备。

coordinator 的控制责任独立于它自己的 lane 是否执行 kernel。例如 MIX mask 6 的 AIC 不执行 body、可以没有 slot 且已发布本 lane 的 release，但仍须持续执行有界 prepared 汇合并发布 start；不能把这项工作仅放在 active lane 的 PREPARING/READY 分支，也不能等全部 lane release 后才发 start。sync-start 的任务级协调者同样保留这项职责，直到 start 发布或 run 错误退出，不为 inactive lane 伪造 prepared ack。

整个 cohort 没有消费准备要求时保留原准入路径，不分配新 ack 区或增加汇合屏障。只要任一成员需要准备，该 cohort 的全部实际成员都参与上述汇合；空计划成员立即完成自己的准备并发布当前身份的 ack，不能沿用历史记录。

prepared ack 使用成员独占发布区与隔离的原子有效标记；发布顺序为本核获取/屏障、完整身份、有效标记。协调者按固定游标、有界扫描验证全部预定成员，匹配 run/cohort、block/lane、worker/slot generation；只按人数计数不足以排除旧 ack。全部匹配后发布 start sequence，成员以权威读取确认当前 sequence，再通过最终 entry gate。该过程不使用 CHECK/FINISH 的一次性位图原语操作可复用 mailbox。

候选 slot 和成员选择属于执行准备资源；准备期间不持有 async lease、不增加业务 TaskId，也不取得最终 kernel 许可。已确认的成员不插入无关业务 kernel，仍以有界方式执行错误/完成/异步服务和汇合检查。准备步骤不等待外部完成，所有参与者必须持续获得准备推进机会；不得占满 cohort 容量后等待尚无位置可进入的成员。

PREPARED 到执行之间，backend 必须证明允许的 scheduler 服务不会在目标范围产生旧缓存或 dirty 写回；范围所有权及稳定数据版本也必须持续成立。若某服务会破坏此合同，必须调整其缓存访问/准备顺序，不能带着已失效 ack 发放许可，也不能通过关闭全部异步服务掩盖问题。

最终许可要求原 admission 条件、当前执行的 PREPARED，以及适用的 group/cohort start 同时成立。取得许可后直接进入对应 kernel，不再插入 CHECK/FINISH、可让出的 cache 维护或其他业务。准备并不改变原 MIX/SPMD 的执行和返回计数，执行释放仍按[完成协议](protocol.md)贡献 B/G。

## 4. 拒绝条件与无额外操作的资格

以下情况必须在 bind 或第一次副作用之前明确拒绝：无法关联通知与 payload；范围、别名或发布版本无法验证；cache line 维护会损坏其他合法使用；实际读取核/读取路径没有 backend 资格。运行时发现身份不符、准备错误或超出静态用途，锁存 run 错误，不能作为空计划继续执行。

需要区别两种空操作：

- **没有消费用途：** Host 已证明该执行没有需要本协议获取的读取，计划为空；不增加 group/join，也不逐 task 清 cache。
- **有用途但无需额外维护：** 具体 A5 silicon、backend/API 版本、核类型和实际 load 路径已证明满足可见性。仍保留用途映射和身份校验，只把对应获取步骤作为经验证的 NOOP；缺少资格不能默认 NOOP。

NOOP 资格需同时提供指令/缓存合同、实际生成路径核对及真实 A5 旧-cache/payload 测试。一次冷 cache 通过、观察核执行过 invalidate 或编译成功均不足以授权。若读取原语自身完成获取，其整个读取序列必须受资格约束，不能将同一结论外推到普通标量读取。

初始化 payload 的旧 dirty line 必须在 DMA/peer 写入前按发布侧合同处理；消费侧不能在新数据到达后通过 flush 旧 dirty 数据来“获取”。消费准备也不释放 UseSet，不代替 peer 最后使用证明或后续 epoch 的覆写约束。

## 5. 验收与实现入口

Host 在 `runtime_maker.cpp` / `native_graph_plan.cpp` 构造用途、范围及执行映射；参数谱系跨普通任务与 Graph 绑定完整传递。backend 实现有界获取步骤并提供核/读取路径资格。`scheduler_ready.h` 拆分物化与 predicate 读取，`scheduler_dispatch.h` 保留候选身份与准备进度，`scheduler_gm_mix.h` / `scheduler_admission.h` 汇合准备结果，`aicore_executor.cpp` 在最终 permit 前统一检查。所有普通、ordered、MIX、Graph child 入口使用同一合同。

诊断记录 plan/range、worker、slot generation、cohort sequence、准备推进、PREPARED 和执行许可的因果关系；关闭诊断后协议仍独立成立。验证须覆盖同步消费者、多 fanin/Graph control、slot 复用、各 lane 准备速度不同，以及提交核、观察核、消费核三者不同且消费核已有旧 payload cache 的场景。完整判据见[验证文档](validation.md)。
