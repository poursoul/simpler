# A5 HBG 异步完成接口与数据布局

[返回总览](../a5-hbg-async-completion.md)。本文规定 Host、kernel adapter 和 resident scheduler 之间的输入与存储合同。新增类型名是实施接口名称；字段宽度、布局断言和序列化版本须随实现一起定义。

## 1. 能力描述与兼容性

每个 CoreCallable 携带一份与二进制身份共同注册的 `CoreAsyncCapability`：

| 字段 | 合同 |
| ---- | ---- |
| `schema_version` | 能力描述的格式版本，未知版本在 bind 前拒绝 |
| `writer_abi` | kernel 写入 AsyncCtx/条件区的 ABI，须与运行时一致 |
| `may_defer` | 是否可能登记条件；不能从元数据缺失推导为 false |
| `max_conditions` | 单次物理执行的完成记录数上界；异步为 1–64，明确同步为 0 |
| `condition_kinds` | 允许登记的条件类型集合，运行期再次检查 |

能力描述采用版本化注册附属数据，与 CoreCallable/ChipCallable 的复制、序列化、缓存和上传保持关联。既有 POD callable 的 padding 不承载新字段；附属数据的内容进入 callable identity 和相关编译/运行缓存 key。

同步描述必须同时满足 `may_defer=false`、`max_conditions=0` 和空类型集合；异步描述必须满足 `may_defer=true`、容量 1–64 且类型集合非空。相互矛盾的描述在 bind 前拒绝。

PyPTO 从 waiter 标记及静态登记上界自动产生描述。原生 C++ kernel 使用相同声明入口；声明错误仍由运行期容量、类型和上下文检查拒绝。模型作者不需要重复手填编译器已经知道的上界。

旧异步生成物必须重新编译或通过明确的兼容转换获得 writer ABI、容量和绑定信息；不能保证任意旧二进制自动识别。无能力描述的原始二进制属于 UNKNOWN，resident bind 不把它当作已证明同步。编译器必须为同步 kernel 生成明确的同步描述。

新 writer ABI 在无有效 deferred 上下文时调用登记 API，必须明确失败或 trap，不能静默改为 kernel 内等待。声明异步但当前分支实际登记零条是合法行为。

## 2. 资源绑定与登记 API

保留公开 `CompletionToken` 及 32 B `DeferredCompletionEntry`。SDMA 的 64-bit post id 继续由 `backend_cookie` 表达；每条 resident 条件另有一个 `uint32_t binding_index`，不占用 entry 的 `_pad`。

### 资源绑定

`AsyncResourceBinding` 是 run 内不可变的设备描述：

| 字段组 | 内容 |
| ------ | ---- |
| 身份 | 当前 run identity、资源 generation、资源类别 |
| 地址 | 整数形式的设备基址、字节长度、合法完成记录区间及对齐要求 |
| 能力 | 允许条件类型、记录宽度、经过验证的观察核集合、可见性策略标识 |
| 使用关系 | 指向本 run `AsyncUseSet` 的索引；关联源、目标、signal、workspace 和外部使用 |

Host 的实际 owner/lease 保存在既有资源体系中，不把 Host 指针或 STL 对象复制到设备。UseSet 是对既有 owner 的使用登记，不另建资源释放器。绑定建立和资源持有要求见[后端与生命周期](backends-and-lifetime.md)。

`AsyncBindingRef` 由当前执行的已验证参数映射取得，包含当前 run identity 与 binding index。绑定在提交或 COUNTER 登记时传递，不能等到 kernel 返回后仅按裸地址反查；地址被新 generation 复用时，裸地址无法区分旧事件。

同一 group 可以包含多个 binding。COUNTER 从 signal Tensor 的窗口身份建立绑定，不要求 SDMA workspace。同一窗口多个合法信号槽可以共享窗口 binding；不同资源代次必须使用不同 binding。

UseSet 还记录完成条件与 payload 的发布用途；signal/workspace binding 不等于 payload 范围。Host 为所有实际消费者建立独立的 `ConsumerAcquirePlan`，包括 `may_defer=false` 的同步消费者。用途、别名/Graph 传递、范围表示及本核准备协议统一见[消费准备](consumer-preparation.md)。这些计划不能通过消费者是否分配 async group 来筛选。

### 登记接口

保留 `get_async_ctx(args)`，为 resident 新增带绑定的登记和预留合同：

```text
binding_ref = resolve_async_binding(ctx, argument_or_workspace_binding)
reservation = reserve_conditions(ctx, required_upper_bound)
commit_condition(reservation, token, binding_ref)
finish_reservation(reservation, actual_count)
register_completion_condition(ctx, token, binding_ref)
```

`resolve_async_binding` 只查当前执行可访问的预绑定映射，不创建 owner，不搜索任意地址。纯 COUNTER 登记可以使用单条登记入口；SDMA 必须在可能产生副作用前预留足够槽位。

每个 group 最多有一个开放预留；`count` 表示已提交的条件数，预留期间不提前增加。成功提交将 token 和 binding index 一起写入，再推进 count；释放未使用的预留不形成可见条件。kernel 返回时仍有开放预留属于错误，不得把半写记录作为已封口条件。

登记验证 run identity、binding index、generation、类型、宽度、对齐、合法区间和地址算术溢出。仅地址落在一个大窗口内不构成合法条件证明。旧 raw-address 入口不能在 resident 路径自行猜测绑定。

`AsyncCtx` 的 resident 扩展至少提供 group ref、run identity、count/error、token 区、binding-index 区、容量、当前执行允许的绑定视图以及 writer ABI。legacy/TMR 按其原版本解释上下文，不能将新旧结构 reinterpret_cast 混读。

## 3. 身份与 GM 记录

Host 按实际 Graph 实例和逻辑 task 展开物理执行。令 `async_mask` 为 active subtask 与异步能力 subtask 的交集，`r` 为 subtask 在该 mask 中的紧凑序号：

```text
group_index = task.group_base + block_idx * popcount(async_mask) + r
G0(task) = logical_block_num * popcount(async_mask)
```

不同 lane 的条件容量从其 callable 描述取得。没有异步能力的 lane 不分配 group。不同 Graph invocation 即使复用同一 Definition，也拥有不同 task ref 和 group 区。

公开 TaskId、Graph parent/child、block、subtask 和原执行 worker/slot generation 只用于身份核对及诊断；路由使用稳定内部 task ref，group 定位不依赖 DFX 数组。

| 记录 | 数量 | 主要字段与写者 |
| ---- | ---- | -------------- |
| `AsyncGroupData` | 每个计划异步物理执行一个 | 身份、容量、封口 count/error、token 与 binding index；执行核唯一写入，封口后只读 |
| `AsyncGroupLease` | 每 group 一个 | phase 与 lease generation；只允许原子访问 |
| `AsyncGroupProgress` | 每 group 一个 | 未满足 mask、条件游标、可见性处理进度；lease owner 独占 |
| `AsyncTaskJoin` | 每个异步逻辑 task 一个 | 静态剩余 B/G；原子递减，定义见[协议](protocol.md) |
| `AsyncTaskFinish` | 每个异步逻辑 task 一个 | 独立 lease 和可移交的 resolver cursor、待发布批次、完成贡献状态 |
| CHECK/FINISH 目录 | 每类记录各一份 | 每 64 条记录一个 64-bit 活动字，原子访问 |
| `AsyncResourceBinding` / UseSet 描述 | 按 run 实际绑定数 | Host 完成构造并发布，运行中只读 |
| `AsyncPublishedUse` / `ConsumerAcquirePlan` / 范围表 | 按实际发布用途与消费者计划数 | Host 构造发布版本、payload 范围、策略和物理执行映射；运行中只读 |
| `ConsumerPrepareState` / cohort prepared ack | 按需要准备的 live slot / cohort 成员容量 | 本核独占进度；ack 带 run、cohort 和 worker/slot generation，协调者汇合 |

FINISH cursor 至少保存 producer ref、当前 waiter、旧链下一引用、fanin 索引、路由阶段、目标队列类型、总容量为 8 的待发布批次、预留身份及完成记账状态。不能引用原核的栈或私有 pending 描述符。

### 布局规则

设备 image 内引用使用相对 image 基址的 offset 或整数索引；外部设备地址使用整数值。wire 类型必须具备 standard-layout 和 trivially-copyable 断言，布局大小、对齐、字段偏移和编码范围必须有编译期检查。

基础隔离单元为 128 B。atomic-only、owner 可变进度和执行核发布区分别对齐并隔离；不同执行的 cached 写入不共享可回写行。身份、封口 header、token 和 binding-index 数组可以合并在该执行独占的发布区内，避免重复对齐。Host 初始化在设备使用前结束，之后遵守唯一写者规则。

token 及 sidecar 按容量 `c_e` 预留，不按运行期实际 count 分配；所有地址在 launch 前稳定。64 条件的 mask 构造必须专门处理 count=64，不能执行 `1 << 64`。

## 4. 容量、初始化与同步路径

令 E 为异步物理执行数，T 为相关逻辑任务数，R/U 为绑定和使用描述数，P/V 为消费计划/范围条目数，S/M 为准备 slot/cohort 成员容量，`A(x)=align_up(x,128)`：

```text
group_bytes = Σ A(group_header + 32*c_e + 4*c_e) + E*(lease_stride + progress_stride)
task_bytes = T*(join_stride + finish_lease_stride + finish_cursor_stride)
directory_bytes = 128*(ceil(E/64) + ceil(T/64))
resource_bytes = R*binding_stride + U*use_descriptor_stride + mapping_bytes
consume_bytes = P*plan_stride + V*range_stride + consume_mapping_bytes
prepare_bytes = S*prepare_stride + M*prepared_ack_stride + cohort_control_bytes
total_bytes = group_bytes + task_bytes + directory_bytes + resource_bytes
              + consume_bytes + prepare_bytes + alignment_gaps
```

每个 stride 以真实类型布局为准，不能只报告 token 内容。U 包含发布用途描述；ack 的身份区、原子标记隔离和 cohort 控制状态都计入实际字节。只读计划可共享条目，准备进度仍按执行核/槽独占；无消费用途时 S/M 及相关控制区为零。Host 同时报告分配、初始化和上传字节。空间按总执行数增长，不声称只与峰值在途数相关。

Host 在 launch 前检查：所有乘加和对齐算术无溢出；每个 `c_e` 在 1–64；B/G0 符合当前 `INT16_MAX` 物理执行限制；全局记录索引可表示；设备预算足够；资源及 backend 能力满足。失败不进入设备，也不通过等待运行期回收来获得记录空间。

初始化发布区和目录时与设备访问完全隔离。group/FINISH lease 从 DORMANT 开始，目录为零，join 为该 task 的 `(B0,G0)`；资源引用在 launch 前持有。重复 run 使用新的 run identity，只有前次真正退休后才能重置同一 backing。

同步 task 不创建 group/join/FINISH，但读取异步输出时仍分配或引用消费计划。准备状态每次 slot 新代次从 UNPREPARED 开始，不能沿用上次执行或另一核的 PREPARED。

整 run 的能力描述均明确同步且没有额外消费获取要求时，Host 不分配异步完成/消费准备区域，resident 不执行目录服务。若外部输入合同另有消费获取要求，只保留相应准备区；它不凭空产生 CHECK/FINISH。UNKNOWN 元数据不能进入已证明同步的快速路径。
