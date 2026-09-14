# A5 device 0：仅 Scalar scheduler + task 执行

基于 nalinaly/simpler 的 fdwic-swimlane-deps 分支
（基础提交 6a0378b），新增 cross_core_only_scheduler；原 ordinary 不变。
设备实测为 Ascend950PR_958b，CANN 9.1.0 weekly 20260708，
PTO 使用该 CANN 附带头文件。用户已授权直接运行 device 0；
环境没有 task-submit，因此这些是非独占实测，不承诺消除外部干扰。

## 正式泳道

仅保留当前流程优化版本，逐次 Atomic + 完整业务描述：

- [6/28/4/1 流程优化后](only_scheduler_a5_8aic8aiv_b256_real_compute_6_28_4_1_flow_optimized_swimlane.json)
- [1/1/1/1 流程优化后](only_scheduler_a5_8aic8aiv_b256_real_compute_1_1_1_1_flow_optimized_swimlane.json)

以上均为 B256、8 个 active AIC + 8 个 active AIV，
16 个独立 Scalar，每核两条 track，共 32 条。
将 JSON 拖入 Perfetto 即可打开。没有 Host/AICPU track。

Host 在 launch 前完成完整构参、依赖、分配元数据与执行镜像准备，
并实际分配计算 workspace。设备不执行 Build：
256 个 Alloc 已完成，1024 个 QK/SF/PV/UP 等待 Scalar pull 执行。
保留每核 4 个 token、每次 2 个 Execute ticket、发射后同步等待完成。
完整语义和 synthetic PA 地址限制见 [实验说明](../../README.md)。

## 业务 profiling：如何理解空白

当前默认 raw schema 为 v2、scheduler_detail=business_spans_v1。
每次 fanin/fatal/drain 轮询都显示独立 Atomic，不再跨整个 FinalDrain
聚合成 PollBatch。DCCI 仍按源码的 region 调用记录，标明 cache-line 数
和尾部 DSB；不是把多条 cache-line 指令伪装成逐条精确测量。
fanin Atomic 详情同时保留 producer 和 consumer task ID。

| 阶段 | 实际业务含义 |
| --- | --- |
| Startup / Admission | 启动到达、静态 header 校验、准入和节流 fatal 检查 |
| Execute dispatch | 推进已有 token，判断占用数和是否继续领取 ticket |
| TicketBind | 领取 Execute ticket、查找空槽并绑定任务 |
| TokenScan | 固定检查 4 槽；必要时再次推进前一轮可能被解锁的任务 |
| Acquire payload + bind | claim、payload acquire、描述符及执行上下文绑定 |
| Fanin | 区分依赖未就绪轮询和 ready → inflight 转换 |
| Complete | completion/DONE 发布、owner token 复位 |
| Drain check | 本核排空判定、arrival 发布、root 汇总 8 组完成数 |

TokenScan 的 completed_tokens / completed_slot_mask 说明本轮完成什么，
pass_index > 0 表示上一轮有任务完成后再检查。
**TokenScan 包含内部 Kernel、Atomic、DCCI 和其他阶段，不能全算扫描开销。**
exclusive_residual_us 是减去子事件并集后的残余，不是纯 Scalar busy。

每个阶段内部没有子事件覆盖的区间用 `[residual]` 业务描述展示。
它只定位到所属源码路径，包含普通内存访问、循环/状态检查和 trace 写入；
不宣称这些时间可以全部优化掉，也不把它们当成 idle。
按互斥区间核算，两个 FinalDrain 窗口内未加业务标签的 Scalar 时间均为 0；
这是**描述覆盖完整**，不是证明每条指令已独立测量。

计算泳道补上 No task issued，并指向同期 Scalar 活动。
它不是独立测出的 engine idle。任务结束边界后的间隙显示留 1 ns，
避免 Perfetto 把 outgoing flow 绑定到间隙标签；真实任务和依赖时间不变，
完整 gap 时间保留在 args。初始化和 observer 尾部也有单独描述，
但不计入性能窗口。

以下为优化前的历史数值摘要；对应旧泳道文件已删除。
当前两份泳道的统计见下一节。

| 次数 | 优化前业务泳道 / ms | Atomic 次数 | TokenScan 轮数 | 完成数为 0 的扫描 |
| --- | ---: | ---: | ---: | ---: |
| 6/28/4/1 | 2.708070 | 12149 | 2613 | 2077 |
| 1/1/1/1 | 0.809124 | 9646 | 1502 | 852 |

两种均有 1024 次 bind、1024 次 ready fanin、1024 次完成发布；
扫描完成 mask 合计均为 1024。未就绪 fanin 检查分别为 5144 和 2628 次。
零完成扫描包含等待依赖和扫到空槽，不能直接叫作无效工作。
observer finish 窗口分别为 2.733574 / 0.831423 ms。

作为性能研究线索，1/1/1/1 的互斥汇总中，payload/context bind
扣除内部 Atomic/DCCI 后，AIC/AIV 分别为 914.448 / 1284.885 core-μs；
除以各自的 512 个任务，约 1.786 / 2.510 μs/task。
这仍含普通访存、校验与观察开销，不是可直接扣除的端到端收益。
因此优先研究绑定/owner 状态路径和依赖检查，而不是由空白面积估算加速比。

曾试过同一次 progress 调用复用 occupied count，以减少重复统计扫描，
无可重复性能收益，已撤回。
详细 A/B 结论见[调查记录](../../docs/occupied_count_reuse.md)。
当时的源代码只增加观察；无新增调度策略。回归 perf-clock 两种各 1 次为
2628.531 / 728.377 μs，均通过，不作为新的性能中位数。

## 保留的流程优化

用户明确的保留标准是：有明确的重复流程消除，且语义得到验证，
不因小幅性能执行波动而撤回。当前组合版本保留：

1. fanin 检查时一次性读取并解码不变的 token 元数据，逐边使用本地值。
2. 由一个 helper 完成 fanin 检查与 inflight 转换，不再提前检查一遍。
3. 本核四位 mask 缓存每个 token 的路由校验结果，首次使用完整校验；
   WaitingBuilt 入口及完成时清除，防止槽位重用继承旧证明。

token 从 claim 到 reset 只有 execute owner 修改；动态依赖 Atomic、
payload DCCI acquire、claim CAS、完成发布和 kernel 发射前验证均保留。
仍是 4 token / 2 ticket / Scalar pull / 同步 issue+wait，没有引入 DAG、
AICPU 调度或 ready 策略变化。LocalStats 大小仍为 1152 B。

组合版本与本轮开始时冻结的二进制在同一 device 0 交替各测 3 次，
perf-clock 无逐任务插桩；仍为非独占、未锁频测量。

| 次数 | 基线中位数 / μs | 组合版本中位数 / μs | 变化 |
| --- | ---: | ---: | ---: |
| 6/28/4/1 | 2621.900 | 2630.195 | +0.316% |
| 1/1/1/1 | 725.527 | 727.277 | +0.241% |

目前未确认端到端提速。保留依据是明确的流程去重与协议等价，
不是声称上表胜出。单独试验的诊断结论也保留：
fanin 两处合并版本各 7 组 A/B，中位数变化 +0.270% / +0.504%；
单独 route-cache 各 5 组，变化 -0.223% / +0.096%。
各组使用当时配对基线，不能把变化相加或跨组挑最快值作为结论。
最初按性能门槛临时撤回 fanin 候选，用户明确流程保留标准后已恢复，
最终源代码是三处简化的组合版本。

流程优化后的两份联合泳道窗口分别为 2.708326 / 0.807319 ms；
observer finish 分别为 2.734612 / 0.829940 ms。
Atomic 次数为 12391 / 9685，TokenScan 轮数为 2700 / 1514。
每份仍有 1024 次 bind、1024 个 kernel、1024 次完成发布；
逐核最大记录数为 2955 / 1797，零丢失、零 PollBatch。
调度顺序和轮询次数会随竞争变化，不能把次数差异直接当成原语删减。
两个 FinalDrain 窗口未加业务标签的 Scalar 时间仍为 0。

新样本中 AIV fanin 阶段扣除内部 Atomic/DCCI 后的累计时间，
1/1/1/1 从 868.476 降为 788.852 core-μs；6/28/4/1 从 1334.568
降为 1178.917 core-μs。这只是有插桩样本的局部观察，其他阶段也会变化，
不能把它换算成无插桩端到端收益。

本轮新增 3 项 C++ 回归，连同 26 项转换/归因测试共 29 项通过：
4080 次 fanin 参考算法等价检查，覆盖返回值、前缀及精确轮询顺序；
共享执行协议；Execute ticket/drain 及路由缓存生命周期。
生命周期测试覆盖重复等待、完成清除、异类任务重用和旧缓存不能放过
新 ticket 的错误 owner。小任务图仅用于协议测试，不观察 B1 性能。
CPU 两种 layout-diagnostic 均通过，perf-clock 仍只有 32 次性能边界读取。
组合版本上板 perf-clock、两份泳道、golden 和原生 Perfetto 导入均通过，
每份为 16 组 / 32 track / 1024 task / 1024 条正确依赖 flow，
没有非零 error/data_loss。

## 时间口径与历史基线

统一 AICore SYS_CNT、1 GHz；不使用 Host 时钟，不做逐核时间对齐。
时间窗口是最早 Scalar startup increment 前到最晚 FinalDrain 结束；
不包括 Host 准备、H2D、设备入口前段初始化和最终 trace 回写尾部。
它是 scheduler 与 task 的联合耗时，不是纯调度开销。

以下仅保留早期轻量阶段版本的历史测量摘要，旧泳道文件已删除。

| QK/SF/PV/UP 次数 | 完整泳道窗口 / ms | 无泳道中位数 / ms | 无泳道范围 / ms |
| --- | ---: | ---: | ---: |
| 6/28/4/1 | 2.690446 | 2.627425 | 2.622671–2.628543 |
| 1/1/1/1 | 0.795583 | 0.724222 | 0.723216–0.724292 |

当时泳道每种 1 次，constant 输入。无泳道为独立 perf-clock 编译产物，
两种次数交替各执行 3 次，每核只有两个性能边界读数，
watchdog 的正确性时钟读数另计。三次观测分别为：

- 6/28/4/1：2627.425、2622.671、2628.543 μs。
- 1/1/1/1：724.222、723.216、724.292 μs。

表中泳道窗口按 FinalDrain trace 结束取值，比 host 打印的相邻
边界读数晚 1 ns。到 observer finish 的窗口分别为
2.709636 ms 和 0.815001 ms，不应混入上述 scheduler 窗口。
完整 profiling 会扰动执行和任务归属；不能把两种产物的差值
机械解释为纯 Atomic 开销或无扰动的调度成本。

Task track 是同步 kernel 调用窗口，包含发射、搬运与等待，
不是独立测出的 Cube/Vector busy。Scalar 同期的 issue + wait
明确表示阻塞。旧版本 PollBatch 显示为 instant，详情保留原始窗口与调用数，
不画成额外的持续 busy；当前版本逐次显示 Atomic，无 PollBatch。
依赖箭头读取真实 payload fanin；
位于 launch 前的 Alloc 不伪造时间戳。

## 校验结果

两份均通过：

- 16 个 worker 启动，设备 Build ticket 与全部构建计数为零。
- 1024 个 kernel 各执行一次，每种 256 个，角色与输出 golden 正确。
- payload/执行任务表保持不变，token 排空，8 组 drain 精确闭合。
- trace 丢失数为零；逐核记录、Atomic 加权次数和 DCCI 计数闭合。
- fanin 完整，依赖时间顺序正确，每核同步 kernel 不重叠。
- Perfetto Trace Processor v58.2 实际导入：
  16 个 process、32 个 thread、1024 个 task slice、1024 条 flow，
  无非零 error/data_loss 统计项。

另外，A5 6/28/4/1 的 layout-diagnostic 输出校验通过；
CPU scalar-NOP、两种 layout-diagnostic / perf-clock 回归通过，
perf-clock 的 CPU 计数证明每核 2 次、合计 32 次性能边界读取。
最新转换器及区间归因共 26 项测试通过，覆盖 mask/完成计数、
阶段完整性、嵌套、残余区间并集和依赖端点显示保护。
当前两份也通过上述上板和原生 Perfetto 校验；最大逐核记录数为
2955 / 1797，均远低于 65536，无丢失、无 PollBatch。
CPU 最新 scalar-NOP 回归完成且导出成功；之前一次 CPU 运行的 trace
header 校验失败（任务语义通过），重采通过，未将失败样本用作交付，
也未放宽完整 trace 校验。CPU 时间没有作为 A5 性能交付。

本目录仅保留两份流程优化后的正式 JSON 和此说明；
无 raw/log/临时分析过程数据，
不生成文件 SHA 校验。
