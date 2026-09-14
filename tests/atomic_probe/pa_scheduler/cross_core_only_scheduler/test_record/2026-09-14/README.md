# 单 CAS 领取 + dispatch 预绑定实测

在上一轮预绑定版本上，去掉 Claim 前的独立原子状态读取。
完整 `acquire payload + bind` 中位数从约 1.29–1.55 μs
降到约 1.07–1.23 μs。每份 B256 减少 1024 次状态预读，
Claim CAS、DCCI 和动态依赖协议保留。

## 两份正式泳道

- [6/28/4/1 单 CAS](only_scheduler_a5_8aic8aiv_b256_real_compute_6_28_4_1_single_cas_swimlane.json)
- [1/1/1/1 单 CAS](only_scheduler_a5_8aic8aiv_b256_real_compute_1_1_1_1_single_cas_swimlane.json)

直接用 Perfetto 打开。均为 B256、8 AIC + 8 AIV、16 个 Scalar、
32 条 track、1024 个 kernel 和 1024 条执行依赖 flow。
没有 Host/AICPU 泳道，统一使用 AICore SYS_CNT、1 GHz。
完整逐次 Atomic/DCCI 与业务 residual 描述保留。

环境：device 0，Ascend950PR_958b，CANN 9.1.0 weekly 20260708，
使用该 CANN 的 PTO 头。没有 task-submit 独占锁，未锁频。
输入为 layout-diagnostic；两种配置只改变完整计算 pipeline 迭代次数。
仍是 controlled real-compute workspace，不是完整 PA 数值链。
正式目录只保留两份最终 JSON 和说明，无 raw/log/文件校验清单。
此前流程优化和预绑定版本的旧泳道 JSON 均已删除。

## 实际改变与保留的边界

本轮只改变 Claim 前的状态取得：`ExpectedPrebuiltPaControl` 根据
固定 G1 task ID、PA 参数形状、Host owner 重建完整 BUILT 控制字。
Host 准备时验证实际控制字与它一致；设备直接用该值做
BUILT→CLAIMED CAS，不预读可变 cell，也不在 acquire 前读取 payload。
CAS 返回值不匹配即失败，错 task、owner、engine、长度、阶段和
重复领取不会被接受。不是无条件 exchange，也不是取消排他检测。
仍使用同一个 Claim/acquire helper，未重构 token 状态或本地存储。
Host/device ABI generation 为 55，防止新旧产物误混。

上一轮已保留的预绑定机制如下。

Host 准备完整执行镜像并分配 SchedulerState 后，以最终设备地址填入
descriptor 指针、scalar 参数和任务私有的只读 Local/GlobalContext。
随后 H2D、launch。设备不新增 Build；仍然分布式 Scalar pull，
每核 4 token、每次 2 ticket、同步 issue + wait。

每个 kernel task 使用原 payload 空闲容量中的 512 B 只读表。
不增加 SchedulerState、SharedExecCell 或 ExecutionToken 大小；
但增加了有效数据和冷 acquire 范围，不能称为零内存代价。
QK/SF/PV 的 payload acquire 从 10 条 cache line 增至 18 条，
UP 从 16 条增至 24 条，都计入原 site 12 DCCI span。

从 Scalar 热路径移走 header/layout/保留位复核、固定 PA 参数形状复核、
tensor/scalar 逐项填表、context 初始化和发射前重复 context 检查。
这些依赖完整、不可变镜像的静态保证，由 Host 准备检查或固定构造承担；
不是不检查就任意接受坏 payload。
保留 claim CAS、owner/engine/task 检查、acquire 边界与 DCCI、
动态 fanin、fatal、inflight、completion/DONE 和 drain 协议。

新 residual 名称为
`Claim bookkeeping / attach prebuilt dispatch and task metadata [residual]`。
最后一段仍包含 acquire 后的普通 GM 读取、将参数表地址和任务元数据
关联到 owner token、转入 WaitingFanin，以及 DCCI observer 记录写入等。
它不是单独测出的纯检查或某一条指令耗时，不能断言剩余时间均不可优化。

实现入口见 [预绑定准备](../../common/prebuilt_dispatch_host.h) 和
[单 CAS 控制字策略](../../common/pa_exec_adapter.h)，
payload acquire 仍在 [prepared claim](../../common/shared_exec_protocol.h)。
适用前提及 synthetic tensor 地址限制见 [实验说明](../../README.md)。

## 局部 profiling 对比

基线是本轮修改前的预绑定工作区，冻结源码和已验证的 Host/kernel 产物；
不是更早的未预绑定提交，也不混用 9 月 11 日旧样本。
基线与新版本在同一 device 0 各重新采一份完整泳道。
每格为同种任务的 256 次观测中位数，单位 μs。
这里的最后 residual 严格取 payload DCCI 结束至 ExecBind 结束，
逐任务确认对应同一残余区间；不是 ExecBind 所有 residual 的总和。

| 次数 | 任务 | 完整 ExecBind 前→后 | 最后 residual 前→后 | DCCI 前→后 |
| --- | --- | ---: | ---: | ---: |
| 6/28/4/1 | QK | 1.323 → 1.089 | 0.356 → 0.373 | 0.051 → 0.050 |
| 6/28/4/1 | SF | 1.512 → 1.211 | 0.519 → 0.519 | 0.060 → 0.053 |
| 6/28/4/1 | PV | 1.359 → 1.092 | 0.429 → 0.421 | 0.051 → 0.051 |
| 6/28/4/1 | UP | 1.379 → 1.118 | 0.431 → 0.424 | 0.072 → 0.069 |
| 1/1/1/1 | QK | 1.291 → 1.075 | 0.415 → 0.348 | 0.051 → 0.053 |
| 1/1/1/1 | SF | 1.549 → 1.227 | 0.544 → 0.455 | 0.057 → 0.054 |
| 1/1/1/1 | PV | 1.319 → 1.068 | 0.427 → 0.411 | 0.051 → 0.051 |
| 1/1/1/1 | UP | 1.385 → 1.113 | 0.431 → 0.358 | 0.069 → 0.069 |

两份 state peek 均为 1024 → 0，Claim CAS 均保持 1024。
所有 Atomic 总次数分别为 13631 → 12788、10395 → 9429；
总数还受动态 fanin/fatal/drain 轮询影响，不能要求恰好减少 1024。

完整插桩窗口：6/28/4/1 为 2638.338 → 2640.023 μs，
1/1/1/1 为 739.211 → 730.361 μs。前者没有下降，
不能把局部 span 的缩短直接称为完整泳道窗口提速。
普通访存与 trace 写入仍包含在 residual 内；不能把各 task 的节省
直接相加当成总窗口收益，也不能由 DCCI 时长推断所有数据已命中缓存。

## 独立计时与验证

另编译 perf-clock：关闭逐任务泳道插桩，每核只有两个性能边界读数，
watchdog 正确性读数另计。取最早 startup 到最晚 final drain，
不包含 Host 准备、H2D 或 trace 导出尾部；不是完整端到端模型耗时。
两种次数分别做 3 轮 A/B；顺序为基线→新、 新→基线、 基线→新，
同一设备串行运行。每行列出三次观测及中位数，单位 μs。

| 次数 | 版本 | 三次观测 | 中位数 |
| --- | --- | --- | ---: |
| 6/28/4/1 | 预绑定基线 | 2590.684 / 2604.109 / 2592.819 | 2592.819 |
| 6/28/4/1 | 单 CAS | 2570.502 / 2573.556 / 2571.863 | 2571.863 |
| 1/1/1/1 | 预绑定基线 | 680.543 / 678.501 / 680.762 | 680.543 |
| 1/1/1/1 | 单 CAS | 662.431 / 666.106 / 670.228 | 666.106 |

本轮中位数分别下降 0.81% 和 2.12%。只有各 3 次、非独占且未锁频，
仅作为当前环境观测，不外推新芯片或完整 PA 的收益。

37 项测试通过：原协议、fanin 等价、token 路由生命周期、
独立目录、预绑定重定位/context 初值、不可变参数表、动态 claim
拒绝路径、固定控制字覆盖以及 residual 改名不改变测量边界。
新增转换检查明确拒绝状态预读、缺失/重复 Claim 和错误领取者；
C++ 测试删除 observer 的 LoadCellState 方法，确保预期值生成不依赖它。
CPU 两种配置的 perf-clock 数值及任务闭环通过。
A5 两份完整泳道和 12 次 perf-clock A/B 全部通过数值、
1024 task 恰好完成一次、token/drain 闭合和零 device Build 检查。
新泳道参数表及 payload 执行前后保持不变，trace 零丢失、零 PollBatch，
最多每核 3519 / 2009 条记录，依赖完整且时序正确。
两份均通过原生 Perfetto Trace Processor v58.2 导入：16 process、
32 thread、1024 task slice、1024 flow，flow 端点都落在实际 task，
无非零 error/data_loss。调度窗口内未加业务标签的 Scalar 时间为零；
这表示业务描述覆盖完整，不代表逐条指令均被独立测量。

复现时从实验目录运行，先加载匹配 CANN 环境：

~~~bash
bash setup.sh
bash test.sh
bash build.sh ccec swimlane
bash run.sh swimlane ccec --device 0 --real-compute-counts 6,28,4,1 \
  --real-compute-pattern layout-diagnostic
bash run.sh swimlane ccec --device 0 --real-compute-counts 1,1,1,1 \
  --real-compute-pattern layout-diagnostic
bash build.sh ccec perf-clock
bash run.sh perf-clock ccec --device 0 --real-compute-counts 6,28,4,1 \
  --real-compute-pattern layout-diagnostic
bash run.sh perf-clock ccec --device 0 --real-compute-counts 1,1,1,1 \
  --real-compute-pattern layout-diagnostic
~~~
