# A5 HBG 异步完成验证与性能验收

[返回总览](../a5-hbg-async-completion.md)。本文定义测试方法和通过条件。抽象模型、CPU helper 测试、目标编译、仿真、真实设备和性能资格分别报告，不能互相替代。

## 1. 协议与接口验证

模型应允许在原子操作返回、局部责任产生、记录写入、状态发布、目录修改、计数贡献及 FINISH 激活之间暂停。唯一完成者必须由原子返回值确定，不能在模型中预先用全局布尔锁强制唯一。

| 验证项 | 必须观察的结果 |
| ------ | -------------- |
| 既有登记接口、容量 0/64/65 | 无新注册声明也能登记；合法零登记完成；条件溢出锁存首错且不提前发布完成 |
| 资源绑定 | 同地址不同 generation、旧 run、跨 domain、未知 kind、越界、未对齐均按合同处理 |
| 消费计划 | 同步消费者也取得 payload 范围；signal 与 payload 分别绑定；alias、多 fanin、Graph PARAM/SUB_TASK 谱系及数据版本不丢失 |
| 消费准备身份 | worker/slot generation 或 cohort sequence 不匹配不能取得许可；旧 PREPARED/ack 不能被复用 |
| 范围与 NOOP 拒绝 | 无法确定必要获取范围、非法 cache line 共享、无资格读取路径明确失败；不能隐式生成空计划 |
| 登记预留 | token 和 sidecar 一起 commit；开放预留不能被 count 提前公布；失败不生成成功条件 |
| lease 交接 | 旧版本 CAS 失败；同一记录同时至多一个 owner；交权前进度可见 |
| 活动目录 | bit 0/63、重复置/清、同字并发不改邻位；合法 126 次邻位变化后仍可完成 |
| 目录超界 | 超出一次性生命周期合同明确失败，不能跳过未完成发布后继续业务 |
| 静态 B/G | 最后 block/最后 group 任意次序，零登记、skip、多 block MIX 均只激活一次 FINISH |
| 长 fanin/fanout | 每一步受预算约束；跨核续接不丢失、不重复 waiter 或 ready 发布 |
| 责任移交 | waiter 挂到下一 producer 后立即被其改写，旧 FINISH 仍沿自己保存的旧链继续 |
| resolve/退休 | TaskDone 可以早于 drain，run 完成及 backing 回收不能早于最后尾部 |
| 服务点 | publication 持续 RETRY、消费准备/汇合未完成、admission 失败、持续业务 ready 时各类服务仍可达 |

CPU 测试必须调用生产算法或生产 helper。可控原子适配器用于构造最坏合法交错，真实线程测试用于发现并发访问问题；不能只测试与实现同构的另一套计数器。

至少保留以下有效负对照：漏掉零登记 group 结算、让普通观察者读零后再次激活、resolve 前计入完成、挂链后重新读取旧 waiter 链接。每个负对照都必须能使相应测试失败。

消费准备的负对照至少包括：仅对登记条件的任务调用获取、跨 Graph EXIT 丢失范围、复用上个 slot generation 的 PREPARED、只检查 MIX 一个 lane 或 sync-start 一个 block 的 ack。定向测试必须拒绝这些错误实现；真实 A5 可见性还须用旧 cache 和完整 payload 建立独立证据。

### 固定执行资源饱和

基础场景为 [defer_wait_progress](../../../../tests/st/a5/host_build_graph/defer_wait_progress/test_defer_wait_progress.py)。核数由当前 run 的 `rt_available_aiv_count()` 获取，不固定设备或核数；waiter、普通 probe 和 publisher 使用同一 AIV 资源池。

令 C 为实际参与测试的同类执行核数。依次测试 C−1、C、C+1 个普通 waiter，先提交 waiter，再提交使用相同核资源的 publisher。publisher 不能依赖 waiter；消费者依赖 waiter 并校验实际 payload。C 个阻塞 waiter 已足以形成资源饥饿，无须超过 C。

必须观测 waiter 登记返回、publisher 获得执行、条件迟后满足、消费者最终运行；未满足条件不能放行消费者。CHECK 每次有界并交回调度权，不能把阻塞循环包装成新 task。另测持续 ready 业务下条件仍能被检查、长 PENDING 不虚报进展。

## 2. 真实异步与任务形状

### SDMA 基本闭环

任务 A 发起真实搬运并登记条件，B 依赖 A 消费输出，C 独立。用可观察的记录证明以下全部成立：

1. A 的 kernel 已返回且实际完成记录仍为 PENDING。
2. C 在这个窗口内推进，A 不再占用执行资源。
3. B 在条件满足之前不进入业务 kernel。
4. 条件满足后 B 执行，搬运结果和 B 的计算结果均正确。

未观察到真实 PENDING 窗口只能证明普通正确性，不能获得异步重叠资格。通过手工提前改标志或跳过条件产生的结果不算真实 SDMA 证据。

### 通知与缓存

使用真实 `remote_store → notify → waiter → consumer`。先让 waiter 达到实际 AIV 饱和压力，再提交 publisher，确认释放执行核后 publisher 能推进。保持原 workload 的依赖图，不通过给 waiter 增加不必要的 publisher 依赖来回避阻塞问题。

为每种受支持条件验证 AIC、AIV、自核和异核观察组合；消费核要覆盖与提交核、观察核不同的情况。预热消费者旧 cache，并构造目标旧 dirty cache，验证完整 payload 与依赖输出，不能只校验标志。

消费准备需在实际生产准备入口和最终 entry gate 上验证以下矩阵：

| 消费场景 | 验收判据 |
| -------- | -------- |
| 不登记条件的同步消费者 | 仍满足必要的本核获取；已有旧 cache 时读到完整新 payload |
| 多 fanin、重叠 view/alias | 覆盖全部相关范围，兼容范围合法去重；不同数据版本不被误合并 |
| `异步写 X → Graph EXIT → 同步读 X` | 获取落实到业务消费者；多 Graph invocation 保持各自 实例地址及运行身份 |
| 消费核与提交/观察核均不同 | 诊断确认实际核身份，旧 cache 确实在消费核建立；不能依靠偶然同核通过 |
| slot 复用及更换执行核 | 准备在新 slot generation/核重新执行；旧 ack、旧 cursor 不可获得许可 |
| 普通候选准备后暂未获准入 | 本核先执行其他业务后清除旧准备结果，重新获取；不阻塞更早的 cooperative obligation |
| MIX 3/5/6/7 准备错峰 | 给各 active lane 不同的真实维护范围/步骤；最慢 lane 完成前所有 lane 均无最终许可 |
| MIX mask 6 的 inactive coordinator | AIC 无业务 slot、已 release 本 lane 后仍能汇合 AIV 准备并发布 start；不得等待 AIV 执行 release 才发 start |
| sync-start / MIX sync-start | 不同 block/lane 准备错峰；全体到齐前无 kernel 进入，后续执行保持原 cohort 合同 |
| predicate 读取异步 payload | 覆盖 predicate-only Tensor 不在 body 参数中的普通/Graph 任务；仍生成范围计划且获取先于求值，合法 skip 不发 DMA，并按原合同结算 |
| 经验证的 NOOP 路径 | 给出具体指令/backend/核资格，旧 cache、跨核、多轮结果通过；换成无资格读取路径时拒绝 |
| 已就绪外部输入 | 全同步 run 的 READY_INPUT 保留入口 owner/发布版本和本核获取，不生成伪 producer/async group；缺少既有就绪证明时拒绝 |

错峰准备期间独立 CHECK/FINISH 和错误服务必须推进；取得最终许可后不再出现准备或异步服务事件。预热/marker 本身不得增加会改变被测 DAG 的依赖，旧 dirty line 必须按发布侧合同在外部写入前处理；不能把事后 flush 旧数据当成有效获取。

| 任务形状 | 重点 |
| -------- | ---- |
| 普通单 lane | 非零、零登记、已经满足、实际延迟、1/64 条条件、多类型/多资源混合 |
| SPMD | 超过并发容量的分波、不同 block 不同条件数、零登记与非零混合 |
| MIX 3/5/6/7 | 同步 lane、零登记 lane 和 pending lane 混合；coordinator 只贡献一次 B |
| sync-start | 容量边界，取得许可后不插入检查；释放后的等待不扣住下一 cohort |
| 单 lane predicate | skip 不发 DMA，预分配 group 仍结算；非法组合拒绝 |
| Graph | 多实例、内外 fanin、多个 sink、不同实例相同 local id、普通任务交错 |
| PyPTO waiter | 沿用现有 waiter 登记接口，多 peer、多窗口，保留禁止组合的负例 |

SDMA 还须覆盖同核连续提交、多个公共完成 record、flag/SQ 压力和零字节请求。已知非零请求返回异常零 handle 时必须失败，不能被归类为合法零登记。

### 多卡用例复用与补齐

优先复用 a2a3 已有多卡异步通信场景的 DAG、通信顺序和 golden，再适配 A5 HBG resident。以下入口提供可复用的业务语义，其原有 runtime 或平台测试结果不构成新 resident 路径资格：

| 现有入口 | 可复用内容 | A5 HBG 适配要求 |
| -------- | ---------- | --------------- |
| [a2a3 SDMA deferred demo](../../../../examples/a2a3/tensormap_and_ringbuffer/sdma_async_completion_demo/test_sdma_async_completion_demo.py) | 两卡 peer fetch，搬运输出及依赖消费者输出的 golden | 接入 A5 的 workspace、完成记录和全核观察；证明返回后真实 PENDING 窗口 |
| [共享 deferred_notify](../../../../tests/st/worker/comm_domain/deferred_notify/test_deferred_notify.py) | 两卡远端 payload、signal waiter 与消费者 | 保留原依赖，验证 A5 缓存可见性和 resident 完成；该入口本身不列 A5 onboard |
| [共享 async_notify](../../../../tests/st/worker/comm_domain/async_notify/test_async_notify.py) | a2a3/A5 两卡通知 counter 和依赖结果 | 从 TMR 接到 HBG resident，增加饱和 waiter 与异核检查 |
| [A5 SDMA deferred demo](../../../../examples/a5/tensormap_and_ringbuffer/sdma_async_completion_demo/test_sdma_async_completion_demo.py) | A5 通信域 SDMA 资源配置与两卡 golden | 复用 backend 配置，验证HBG resident 调度和既有生命周期 |
| [HBG async boundary](../../../../tests/fdwic/pa_hbg/spmd/test_async_boundary.py) | 未提供合法异步上下文时的拒绝边界 | 保留适用负例；不能用不发起异步工作的边界测试替代真实通信 |

没有对应正例或故障判据的组合必须构造新增用例；不能只修改平台列表和 runtime 名称。新场景须验证实际运行的是 A5 HBG resident 及目标产物，不能由 legacy、TMR 或 kernel 内阻塞 fallback 获得通过。

| 多卡覆盖 | 必须覆盖的情形 |
| -------- | -------------- |
| 基础通信 | 两卡双向 SDMA、remote_store/notify、真实消费者结果与独立任务进展 |
| 多 peer | 至少四卡的多个 peer 等待；通知乱序、迟到 peer、每 peer 数据身份可区分 |
| 多条件/多资源 | 同组多 peer、多窗口及 SDMA/COUNTER 混合，必须等待全部必要条件 |
| 饱和与任务形状 | waiter 达到真实 AIV 饱和压力；SPMD、MIX、sync-start、Graph 与通信交叉，包含同步消费者与错峰准备 |
| 重复与代次 | 多轮 run、保留域复用、合法独立槽、旧 token/旧窗口、peer 提前覆写下一 epoch |
| 失败与关闭 | peer 未发布、提交/登记失败、长 PENDING、pending close、核故障和未知外部访问 |

多卡 fixture 保留逐 rank golden、延迟条件及 peer 身份，不以各 rank 结果相同掩盖串数据。故障用例的预期结果可以是明确失败和资源隔离，但不能是提前成功或未知资源被释放。四卡或故障环境不可用时对应项保持未验证，不用两卡正常场景代替。

## 3. 错误、复用与关闭

| 注入位置或输入 | 验收判据 |
| -------------- | -------- |
| 提交前容量或输入验证失败 | 该次请求没有设备副作用，未使用预留被归还 |
| 已尝试提交后失败或登记异常 | 首错保留，相关 owner/backing 仍持有，没有成功 TaskDone 或自动重放 |
| peer 提前写下一 epoch | 协议阻止复用或明确拒绝；allocation 存活不能被当作正确性证明 |
| 旧 token/旧窗口 | 旧操作未结束时不得复用资源，不能只靠 generation 检查 |
| 长时间 PENDING | watchdog 不被纯轮询刷新，错误有具体 task/group/条件身份 |
| owner 暂停在交权/清位/贡献边界 | 正常路径不提前退休；恢复后没有重复贡献 |
| owner 硬故障 | 不强制接管半写记录；未知外部访问阻止复用和释放 |
| pending 时 close | 不接收新调用，正常 drain 或明确失败并保留未知使用 |
| 返回/ACK 已到而 DMA/peer 仍访问 | 实际资源账本拒绝提前释放 |
| 清理步骤失败 | 原始错误不被覆盖，尚未释放的 owner 可追踪，不能报告清理成功 |

检查实际资源 owner、引用及后续 admission 行为，不能只检查 scheduler 的 FAILED 位。诊断截断应保留“证据不足”状态，不补造未观察到的完成事实。

## 4. 性能与选型实验

正式性能测试只在完整新增功能、单卡/多卡用例和受影响的原有功能回归通过之后执行。性能标准采用 [PR #2864](https://github.com/hw-native-sys/simpler/pull/2864) 的用例集合，复用当前仓库中的清单及采集器；历史耗时只作背景，必须在同机同卡重新采集冻结基线与候选。

| 基准输入 | 范围 |
| -------- | ---- |
| [fixed-cases.json](../../../../tools/benchmarks/a5_hbg_graph_task/fixed-cases.json) | 7 项固定无 Graph benchmark |
| [selected-st-cases.json](../../../../tools/benchmarks/a5_hbg_graph_task/selected-st-cases.json) | 11 项具有调度性能价值的 ST |
| [csa-cases.json](../../../../tools/benchmarks/a5_hbg_graph_task/csa-cases.json) | 1 项 CSA 补充场景 |
| A5 Qwen3 decode | 40 层、B16、seq_len=3500，验证完整 out/K/V，再单独采集 |

共 19 项无 Graph 用例加 Qwen。使用 [collect.py](../../../../tools/benchmarks/a5_hbg_graph_task/collect.py) 核对实际 pytest case 的参数/config、golden 和产物身份，清单与当前代码不一致时先修正原因。19 项至少各采三批独立进程、每批 100 稳态轮；Qwen 预热 5 轮后采集 100 稳态轮。`spmd_paged_attention` 不加入该参考性能集，但 SPMD 功能验证仍是必需项。

新增异步和多卡性能对照作为补充，不能替换上述原有性能集。完整 sweep 在同一设备 allocation 内串行安排同设备的基线/候选，保留所有原始数据，禁止选择历史不同产物的最佳结果拼成当前资格。

正式性能使用 L1；L3/L4 用于诊断归因，结果不混入 L1。相同机器、设备、输入、计算 kernel 和工作量，先采集 A/A 噪声，再采用 B/A/A/B 的成组对照；建议每组 5 次预热、100 次正式调用，并至少重复三组成对测量。保存所有样本及异常，不只报告最优值。

同步回归至少包含 CSA、SWA、Qwen 和短任务调度负载；分别测整个 run 零登记、少量异步 task 混排、各执行分支零登记的情形。异步对照使用等工作量的 kernel 内等待与 deferred；阻塞饱和负例单独报告正确性，不能将其超时当作正常性能样本。

| 场景 | 主要观测量 |
| ---- | ---------- |
| 非零 group 比例从低到高 | 静态 G/动态 P 的真实更新成本、Device 与 Host 开销 |
| 大 E、小在途数及稀疏尾部 | 目录扫描数、满足到发现延迟、初始化字节 |
| 单热点、多核竞争与同字集中结束 | 有效观察、lease 碰撞、目录更新及最后 CHECK 清位并发布 FINISH 的固定尾部 |
| 每组 1/4/64 条条件 | 每条有效观察的移交和屏障成本 |
| 短/长业务 kernel 混合 | 业务吞吐、检查响应、FINISH 尾延迟 |
| 长 fanin/fanout、control 混排 | resolver slice 成本、ready 发布和业务公平性 |
| 不同 payload 范围及 cohort 准备错峰 | 消费计划/准备状态字节、本核维护与汇合成本、满足到消费者执行延迟 |
| 连续 SDMA 提交 | SQ/flag 背压、kernel 返回时间、实际重叠窗口 |

每组报告 Device 总时间、kernel window、Host bind/端到端、均值及 P95，并报告完整分配/初始化/H2D 字节、峰值在途 group 和总 group 数。同步路径若出现跨重复组稳定、超出 A/A 噪声的退化，必须定位并解决，不能用局部异步收益抵消或隐去；需要改变验收目标时先明确新目标。

目录与尾部实验分别报告有效条件观察、目录更新的次数/重试和耗时、lease 交权及屏障、条件满足到首次发现、到消费者进入 kernel 的延迟。覆盖同字集中结束、单热点多核争用、短业务 kernel 和稀疏尾部；单独列出最后 CHECK 两次目录更新的 256 原子操作预算及额外 lease/join/屏障，不能把它隐入 8 个工作单元。用 L3/L4 诊断解释组成，在相同负载的 L1 对照中判断整图影响。

候选仅在满足相同功能、错误和生命周期合同后比较。选择依据见[选型文档](choices.md)，不将单次 CAS 或局部微基准的收益直接外推为整图收益。

## 5. 执行环境与交付检查

硬件测试先识别实际 A5 silicon，通过仓库 `onboard-arch-precheck`，并按 `task-submit` 规则独占所需设备、隔离每次运行的日志。构建与运行使用项目独立 venv；源码、编译产物、PTO-ISA、CANN 和实际加载路径必须一致。架构检查通过本身不构成功能资格。

交付必须提供支持矩阵、正常与故障结果、sim/CPU/A5 分层结果、同步性能回归、异步真实窗口及数值结果、原始 L1 样本，以及明确的未支持组合。关闭诊断后的正确性也必须验证；诊断字段不能成为运行时所需的隐藏身份来源。

基础实现与选定优化均应具备针对生产路径的回归测试。仅有设计模型、单个原语测试或目标编译，不能声明整项异步功能交付完成。
