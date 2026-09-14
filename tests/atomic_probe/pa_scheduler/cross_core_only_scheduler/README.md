# Scalar-only scheduler：8 AIC + 8 AIV

本目录从 nalinaly/simpler 的 fdwic-swimlane-deps 分支、
cross_core_ordinary 复制而来。原目录不变。
只增加这个独立实验，不实现 DAG 插入模式，不创建 build_graph。

## 独立目录与依赖

本目录可以整体复制到其他位置，不要求位于 Simpler 仓库内。
构建、运行、泳道转换、空白分析和回归测试均只读取本目录中的源码；
不需要 PyPTO、simpler_setup 或预先构建 Simpler runtime。

- CPU：Linux、Bash、支持 C++17 的 g++、标准 C++ 库和 pthread。
- Python 工具：Python 3.10+ 与 venv 支持，仅使用标准库，无需 pip 安装。
  `setup.sh` 创建本目录自己的 `.venv`，不查找父目录环境。
- A5：匹配的驱动和 CANN，含 ccec、ld.lld、ACL/runtime 库与 PTO 头文件；
  另需 readelf、rg、awk。当前构建目标固定为 dav-c310-cube/vec，
  使用 x86_64-linux 工具包路径，并非其他芯片/Host 架构的通用后端。
- 上板预约：存在 task-submit 时保留独占预约和架构检查；检查脚本已随包
  放在 `tools/check_onboard_arch.sh`，不依赖仓库 `.claude` 目录。
  该检查使用 npu-smi、CANN platform_config，并沿用一小时检测缓存。
- 内存：SchedulerState 约 1 GiB，另需 workspace、trace 和 Host 镜像空间。

`trace_abi.py` 固化本实验实际使用的 Atomic/DCCI 编号、名称和操作类型，
与 common 中的 raw ABI 对应，不再 import 父目录转换器。
`fixtures/` 包含原 ordinary 的两份协议测试副本，已落实 8+8 拓扑及
路由缓存回归接线，不再运行时读取或改写兄弟目录测试源码。
原许可证随 [LICENSE](LICENSE) 保留；历史 occupied-count 结论也已
[收进本目录](docs/occupied_count_reuse.md)。

迁移时复制整个目录即可，但不要带走 `build/`、`.venv/`、
`__pycache__/` 或 `*.pyc`；在目标机器重新执行 setup 和 build。
正式 A5 JSON 可以随 `test_record/` 一起复制。
目录独立化不改变调度/计算协议，也不代表已移植到其他芯片。

## 执行边界

Host 在 launch 前复用 ordinary 的构参、分配元数据和依赖生成代码，
生成完整不可变执行镜像。构建时的 CPU executor 只推进元数据生命周期，
不执行计算；随后将 1024 个 kernel cell 复位为 BUILT，
将执行 ticket/token/drain 状态复位。256 个 Alloc 已完成。

取得设备 SchedulerState 的实际 GM 地址后，Host 在同一准备阶段调用
`PrepareStaticExecutionBindings`，完成 tensor descriptor 地址重定位、
scalar 参数及只读 Local/GlobalContext 填充，随后才 H2D 和 launch。
这就是本实验中 build/prepare 侧的预绑定，不新增设备 build_graph。
CPU 回归则使用 CPU 执行镜像的实际地址。

每个 kernel task 在 payload 原有空闲容量内放入 512 B 参数/context 表；
SharedExecCell、SchedulerState、ExecutionToken 的大小均不变。
Scalar claim 成功并 acquire 后只关联只读参数表，缓存完成及 fanin 元数据，
不再逐项填写 descriptor/scalar/context，也不重复检查已固定的
payload header/layout、PA 参数签名和 context 初值。这些静态检查在
Host 准备阶段执行，失败时不 launch；动态 Claim/owner/engine 检查、
payload 边界检查、fanin、fatal、完成与 drain 协议仍保留。
acquire 范围真实增加 8 条 cache line，计入原有 DCCI span。

预绑定限定于完整且不可变的执行镜像：内联 descriptor、数字 function ID、
固定 PA 参数形状、稳定的设备地址；所有 active worker 的 sub_block_id=0，
kernel 同步执行，不修改 async context。未来接入可变/异步 context 或
运行期扩图时必须重新设计，不能直接沿用这个静态保证。
通用动态绑定 helper 仍用于 Host 镜像构建和协议回归；只有
RunOnlyScheduler 选择预绑定。Host/device ABI generation 更新为 55，
防止旧 Host 与新 kernel 混用。

领取采用单 CAS：`ExpectedPrebuiltPaControl` 根据固定 G1 的 task ID、
PA 参数形状和 Host owner 重建完整 BUILT 控制字；Host 准备时逐项
核对实际 cell 与该值一致。设备不提前读取 cell 或 payload，直接
BUILT→CLAIMED CAS，返回值不匹配即失败，不等待运行期 Build。
成功后才执行原有 payload DCCI acquire。每份 B256 不再有 Atomic
site 45 状态预读，仍有 1024 次 site 48 Claim；没有新运行开关。
本轮不改变 token 布局、初始化方式或本地存储策略。

设备入口是 RunOnlyScheduler，不调用 Submit、Materialize、Register、
Build 或依赖构建。保留 ordinary 的：

- AIC/AIV 分别领取 Execute ticket，每次 2 个。
- 每个 Scalar 4 个 token，分布式 pull。
- 等待 payload 中的 fanin，发射后同步等待 kernel 完成。
- 发布 completion/DONE，8 个 drain group 每组汇合 2 个 Scalar。
- 启动到达检查、错误收敛和 watchdog。

复用 ordinary 的初始化和执行协议，不表示启动成本为零，
也不是一个经过性能精简的最小 scheduler。
本次不改变 ready 策略或引入新的并发调度策略。

已保留的流程简化：

- 每次 fanin 检查只解码一次 owner 私有元数据，逐依赖推进就绪前缀。
- 用同一个 helper 完成 fanin 检查和 inflight 转换，删除成功后的重复检查。
- 用本核的四位 mask 缓存 token 路由校验结果，首次使用仍完整校验；
  WaitingBuilt 入口与完成时清除，不能跨任务复用旧校验证据。

这些不省略共享状态 CAS、DCCI acquire、依赖 Atomic 或完成发布，
不把未就绪任务提前发射。LocalStats 仍为 1152 B，无新增外部配置开关。
按用户明确的保留标准，流程冗余消除且协议等价的改动可以保留；
小幅性能波动不作为撤回理由，但不据此宣称端到端提速。

B256、默认 context length 8192、每 batch 一个计算组固定不变。
总计 1280 个逻辑任务：256 Alloc + 256 QK + 256 SF + 256 PV + 256 UP。
实际 payload 依赖为：

- QK：无 producer。
- SF：等待 QK。
- PV：等待 SF。
- UP：等待 SF、PV、Alloc。

Host 标记为 build_owner=254，真实 execute_owner 仍为 0..15。
Payload 使用内联描述符和数字 function ID，不将 Host 指针带到设备。
设备执行后校验 payload 和执行任务表未发生修改。

**内存与计算口径**：沿用 ordinary 的 controlled real-compute 负载。
PA tensor 的地址和分配元数据用于调度建模，不解引用这些 synthetic 地址。
实际 Cube/Vector 计算使用 launch 前通过 aclrtMalloc 分配并初始化的
独立 128×128 workspace。它不是完整 PA 数值链，不能冒称实际模型耗时。
原来的约 1 GiB SchedulerState/worker arena 地址跨度也保留。

## 核数与泳道

Launch 8 个 mixed block。A5 的 mixed metadata 仍创建每 block 1 AIC + 2 AIV
context，但 AIV subblock 1 在进入 scheduler 前立即退出；
参与调度/计算的是 8 AIC + 8 AIV，共 16 个独立 Scalar。
这不是关闭芯片上其他核的电源，也不是把全芯片结果按比例缩放。

每个逻辑核在 Perfetto 中独立成组，只有两条 track，共 32 条：

- Scalar scheduler：调度阶段、Atomic、DCCI、issue + wait。
- Cube/Vector task execution：同步 kernel 调用的起止区间。

Task track 包含发射、搬运和等待，**不是单独测出的 engine busy**。
Scalar 上同时间的 issue + wait 表示阻塞，不表示另有调度并行。
依赖箭头读取真实 payload fanin；Alloc 位于 launch 前，不伪造时间戳。

A5 所有事件统一使用 AICore SYS_CNT，频率沿用 ordinary 的 1 GHz。
只减去同一个全局 origin，不按核单独对齐；没有 Host/AICPU track。
CPU 回归文件明确标记 CPU_steady_clock，不能当作 A5 性能。
当前默认输出 v2：所有 polling Atomic 都逐次记录，禁止 PollBatch。
v1 历史文件仍可读取，其 PollBatch 仅显示 instant marker。
完整 profiling 会扰动执行，不能把泳道窗口当作无插桩性能基线。

Scalar 业务阶段包括启动/准入、Execute dispatch、TicketBind、TokenScan、
payload/context bind、fanin、完成发布以及 drain 检查。
每轮 TokenScan 固定检查 4 个槽，记录完成槽位 mask 和 pass index。
它包含内部任务执行；不是纯 Scalar 扫描时间。
详情中的 exclusive_residual_us 扣除已测子事件的时间并集，
仍包含普通访存、控制和 profiling 写入，不等于某条指令耗时。

后处理把各业务阶段中没有子事件覆盖的区间标为 `[residual]`，
明确这是源码路径归属，不是新增的指令测量。
`dispatch_binding=host_prebound` 的新采集将 ExecBind 内 residual 改名为
`Claim bookkeeping / attach prebuilt dispatch and task metadata`，
避免继续把它解释成逐项填写 descriptor/context；时间边界不变。
未带该标记的历史采集仍显示旧业务名称。
计算 track 的间隙标为 No task issued，不冒称 engine idle/PMU。
为保证 Perfetto 的依赖箭头仍绑定真实 producer，间隙显示在 producer
结束后留 1 tick（A5 为 1 ns）；完整间隙与显示保护值保留在 args。
启动前 owner 初始化与最后 observer 导出尾部单独标注，
不改变 startup-to-final-drain 的性能窗口。

## 构建与运行

以下命令从本目录执行，脚本也可通过绝对路径从其他工作目录调用。

~~~bash
bash setup.sh
bash test.sh
bash build.sh cpu perf-clock
bash run.sh perf-clock cpu --real-compute-counts 6,28,4,1 \
  --real-compute-pattern layout-diagnostic

# 先 source 现有 CANN 的 set_env.sh，再编译 A5。
# 默认使用该 CANN 的 PTO 头；可使用已有的 PTO_ISA_ROOT。
bash build.sh ccec swimlane
bash run.sh swimlane ccec --device 0 --real-compute-counts 6,28,4,1
bash run.sh swimlane ccec --device 0 --real-compute-counts 1,1,1,1

# 无泳道独立计时产物：每核仅两个性能边界读数，watchdog 读数另计。
bash build.sh ccec perf-clock
bash run.sh perf-clock ccec --device 0 --real-compute-counts 6,28,4,1
~~~

存在 task-submit 时，run.sh 自动完成上板预检查和设备预约，
不要再在外面套一层 task-submit。
若缺少工具，只能在确认设备可用后直接运行，且结果明确标为非独占；
host 在 launch 前还会检查实际 SoC 必须为 Ascend950/A5。

run.sh 输出 build/<backend>/swimlane/capture.*/swimlane.json，
可直接用 Perfetto 打开。过程文件和二进制留在被忽略的 build/ 下，
不作为 test_record 交付；最终记录只保留命名明确的 JSON 和说明。
不生成文件 SHA 校验。

## 校验和当前状态

构建检查 device ELF 不得包含 ordinary Build 入口。
运行校验要求：

- 16 个 worker 启动；Build ticket 和所有构建动作计数为零。
- 1024 个 kernel 恰好完成一次；每种 256 个，核型匹配。
- token 排空，drain 到达和完成计数精确闭合。
- 真实计算输出符合 golden，未使用输出保持 sentinel。
- trace 无丢失，逐核记录/Atomic 加权调用数/DCCI 计数闭合。
- 依赖完整，producer kernel 结束不晚于 consumer 开始。
- 每核同步 kernel 不重叠，最终导出固定为 16 组、32 条 track。

~~~bash
# 从本目录执行；不需要仓库根目录的 Python 环境。
bash setup.sh
bash test.sh
~~~

2026-09-11：已获用户授权直接使用 device 0；
实际 SoC 为 Ascend950PR_958b，未使用 task-submit 独占锁。
两份 A5 泳道、layout-diagnostic 数值回归及各 3 次 perf-clock 均通过。
CPU scalar-NOP、两种次数的 layout-diagnostic 和 perf-clock 回归通过；
29 项测试通过：26 项转换/空白归因与 3 项 C++ 协议回归。
其中包括 4080 次 fanin 参考算法等价检查和路由缓存生命周期检查。
CPU 文件不进入正式记录。
最新 v2 两种负载的设备泳道与 perf-clock 校验均通过；
此前 occupied-count 复用试验的结论单独保留在调查记录中，
它与本轮保留的 fanin/路由流程简化不是同一个候选。

2026-09-14 独立化验证：在仓库外的带空格目录中重新 setup，
31 项测试通过；副本没有父目录转换器或兄弟实验目录，
并以不同工作目录和 Python isolated 模式检查导入及协议。
CPU/A5 的 swimlane、perf-clock 四种构建通过。
两种 real-compute 次数的 CPU perf-clock 数值与协议校验通过；
device 0 上两种次数的 A5 swimlane、perf-clock 和数值校验均通过。
A5 两份新导出均为 16 核、32 条 track、1024 个 kernel，
空白分析脚本可独立运行，业务未标注时间为零。
新采集仅用于搬迁验证，不替换正式记录、不作性能对比。
原始两份正式采集经本地 ABI 重新转换，输出与原正式 JSON 内容完全相同。

**CPU 完整 trace 的已知限制**：本机两种 real-compute 次数均遇到
trace 校验失败；6/28/4/1 在校验入口读取内存确认多个 worker
达到每核 65536 条上限且 dropped 非零。CPU 算术及等待时间远大于
A5，逐次记录 polling 会耗尽固定容量；数值和任务闭环仍通过，
但不完整 trace 必须拒绝导出。本轮没有更改容量、关闭校验或修改
调度逻辑；CPU real-compute 回归使用 perf-clock，实际泳道使用 A5。

2026-09-14 预绑定优化：33 项测试通过，包含参数重定位、context 初值、
参数表不可变、claim 竞争失败和边界拒绝、旧绑定协议及独立目录回归。
CPU 两种次数的 perf-clock、A5 两份完整泳道和两种次数各 3 轮 A/B
perf-clock 均通过数值及任务闭环校验。局部 residual 与完整 ExecBind
都下降，额外 DCCI 开销计入对比。

随后单 CAS 优化：37 项测试通过，包括固定控制字覆盖、无状态预读、
错 task/owner/engine/长度/阶段拒绝，以及转换器的逐 task Claim 闭合。
两种次数的 CPU perf-clock、A5 完整泳道和各 3 轮 A/B 独立计时通过。
新泳道通过 Perfetto 实际导入，旧正式 JSON 已按用户要求删除。

最新两份正式 JSON、静态化边界和实测数据见
[单 CAS + 预绑定记录](test_record/2026-09-14/README.md)；
[此前流程优化记录](test_record/2026-09-11/README.md) 仅保留历史说明。

设备统计状态必须由入口持有独立的 block-local LocalStats。
当前 mixed ELF 各保留一个 1152 B 的 AIC/AIV 对象，位于 4 KiB 预留区内；
CPU 每线程持有独立对象。二者均在调度前清零。
最初使用函数栈对象时，上板能完成全部任务，但部分 Build 计数与时间边界
异常；改为 block-local 后，swimlane/perf-clock 的所有校验通过。
这是本实验的已验证实现约束，尚未将底层原因归结为某条 store 指令或
特定编译器优化。未在末尾强制清零结果，也未放宽零 Build 校验。
