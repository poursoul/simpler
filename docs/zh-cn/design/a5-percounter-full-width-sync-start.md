# A5 Percounter 全宽 sync_start 连续调度

基线为主线 `798fea3fb`。Host 首次排序时把全宽 `sync_start` 尽量排成连续区间，再基于固定顺序生成区间静态链接、启动预派发表和尾部预派发表。设备沿固定 owner 接续，利用实际经过的全核 barrier 裁剪 fanin。

## 适用范围与任务合同

当前 Percounter 把资源拓扑序投影到 AIC 和 AIV/MIX 两张 order list。全宽 `sync_start` 已有静态 block 映射。普通 MIX 仍由 AIV0 领取并发布，参与核用本地握手入场，各自发布完成。参见 [Percounter 调度说明](../../../src/a5/runtime/percounter/README.md) 和 [PR #3195](https://github.com/hw-native-sys/simpler/pull/3195)。

本方案在 `PERCOUNTER_PENDING_CAPACITY == 1` 时启用，普通 pending slot 和共享 MIX 记录继续按现有方式管理。`X=2/4` 使用现有调度。连续区间仅收纳顶层、无 predicate、全部活跃 callable 为 `SYNCHRONOUS` 的全宽 `sync_start`。候选 kernel 的实现合同是在访问依赖数据前，与本任务全部参与核完成握手，并建立数据可见性。Host 以该合同和实际静态路径判断 barrier 覆盖。

设实际 launch 有 `N` 个 `1AIC + 2AIV` 集群。

| 形态 | `active_mask` | `block_num` | 静态 block 和 callable |
| ---- | ------------- | ----------- | ---------------------- |
| `1c` | `0b001` | `N` | AIC 使用 `cluster_idx` 和 AIC callable |
| `2v` | `0b010` | `2N` | 两个 AIV 使用 `2 * cluster_idx + vector_lane`，调用原纯 AIV callable |
| `1c2v` | `0b111` | `N` | 同簇三个角色都使用 `cluster_idx`，分别调用原角色的 callable |

启动预派发覆盖已就绪且可静态分配的 AICore 任务，包括单 block、非全核 SPMD 和全核 SPMD。尾部预派发覆盖单 block 和非全核 SPMD，包括窄 MIX。参与预派发的任务须在锚点处可解析 callable 和参数，保留原 predicate、defer 和入场语义。需要另行 materialize 或由 lane 外执行的任务沿现有路径派发。

单卡通知的进展依赖由编排在 DAG 中表达。区间内尚未满足的 kernel 阻塞式同步只等待本任务中有静态路径可达的同伴。

## Host 规划

### 首次排序

`host/state.h` 在 bind 收集任务并折叠现有控制节点。以未裁剪的逻辑 DAG 选定 #3204 终段后，从现有资源序按 Graph invocation 切开普通前缀，再由 `host/resource_order.h` 为每段生成候选拓扑序。Graph shell 与展开后的 body 保持为整体，候选排序和局部换位都限定在同一段内。资源调度的 rank 和深度只用于选序，允许无依赖任务跨深度换位，最终顺序须满足所有 DAG 边。

1. 开新区间前优先排拓扑就绪的普通任务。打开区间后，在拓扑 ready 集里优先接能安全连续执行的全宽任务。独立的 `1c` 和 `2v` 在同一 ready 集内配对，分别占用 AIC 和 AIV。
2. 每种参与核型都须有真实入口。全宽 `1c2v` 提供两条链的共同入口。独立 `1c` 和 `2v` 配对开头时各保留一个入口。单核型入口只延伸本类型，遇到缺入口的另一核型时结束区间。
3. 后继的外部 fanin 必须在入口按真实依赖等到完成，或被该后继能经过的入口 barrier 覆盖。无法证明时在该任务之前结束区间。可能 defer 的外部前驱须由实际 DONE 等待保证进展。
4. 尽可能让区间以 `1c2v` 收尾。若区间已打开，独立的 `1c`、`2v` 和 `1c2v` 同时就绪且都能加入，可以依次排 `1c, 2v, 1c2v`。若 `1c2v` 是后续任务的前驱，先排此前驱，再争取用后续的 `1c2v` 收尾。

每取出一个任务，就更新其后继的拓扑入度。配对先用同一 ready 快照选出两项，再更新两项的后继。

#### 二次合并检查

候选排序时按连续段记录全宽区间和普通间隔，单个全宽任务也作为一个区间。各段使用候选序中的半开 rank 范围，并登记同一 Graph 分段内需要检查的间隔。扫描原始依赖边，为每个普通间隔 `G` 汇总两项。`last_full_pred` 是 `G` 中任务的全宽前驱最大 rank，`first_outside_succ` 是由 `G` 指向段外的最早后继 rank。取极值时忽略 `G` 内部边，无对应边时分别视为负无穷和正无穷。无边普通任务在候选排序时直接安排在本 Graph 分段的全宽任务之前。

从左到右检查 `L, G, R`，其中 `L` 是当前全宽区间，`R` 是相邻的后续全宽区间。三段必须处于同一 Graph 分段，保持各段内部顺序。

- **前移。** 若 `last_full_pred(G) < begin(L)`，可尝试把 `G` 移到 `L` 之前，随后拼接 `L` 和 `R`。若 `L` 已由多段拼接，先前前移的间隔已位于 `L` 之前，`G` 只跨过全宽任务。
- **后移。** 前移未采用且 `first_outside_succ(G) >= end(R)` 时，可尝试把 `G` 移到 `R` 之后并合并 `L,R`。完成这个合并后封闭该区间，后续从 `G` 之后继续。同一个间隔最多尝试两个方向。
- **接纳。** 用核型摘要检查新增部分的入口可达性。用预先按消费段索引的边检查 `R` 的外部 fanin，只查询入口显式等待、覆盖前缀或合并区间内已有前驱。采用后增量更新覆盖范围和末端。优先用 `1c2v` 作为共同末端。两个方向均不可行时保留该间隔。

**复杂度。** 设待查间隔数为 `K`，`E` 为总依赖边数。以下约束保证二次检查为 `O(E)`。

1. **一次汇总。** 在候选序中用 task ID 索引所属段和原 rank，扫描每条边一次，更新间隔的两个极值并按消费段登记入边。核型、起止位置和候选间隔在首次排序时登记。
2. **间隔数有上界。** 无边普通任务在首次排序时已前置，所以每个待查间隔至少关联一条边。一条边最多关联两个间隔，故 `K <= 2E`。每个间隔最多做两次常数时间的 rank 比较。
3. **复用已有区间的证明。** 每次只遍历原始右侧段 `R` 的入边列表，以段归属、rank、核型覆盖和入口显式等待集合做常数时间查询。为当前左侧入口登记等待集合时只遍历其显式 fanin 一次，用按 task ID 索引的代次标记复用存储。每个原始段只会作为待并入的 `R` 参与一个间隔的最多两种尝试。扫描成本上界为 `2 * sum_R indegree(R) <= 2E`，其中 `indegree(R)` 是指向该段任务的边数。合并后该段的边无需跟随左侧区间再次扫描。
4. **延迟展开。** 换位只改段引用的连接顺序，保留候选 rank 用于上述判断。完成整轮检查后，沿段顺序一次展开并写入最终 `global_order_id`。这一步随候选枚举及输出做 `O(V)` 遍历，`V` 是任务数。

所以额外的换序判定成本为 `O(E + K + 2E) = O(E)`。从候选序到输出完整任务顺序的总成本为 `O(V + E)`。当 `E=0` 时无待查间隔，直接输出候选序。换位结束后才固定区间边界并生成静态派发计划。仍未形成区间时沿用资源调度的全局序，启动批仍可在该序上单独选择。

### 静态派发计划

固定全局序后执行下列规划，再生成设备表。

1. **区间内接续。** 对每个区间反向扫描。每个任务分别记录该任务的 AIC 和 AIV 角色在本类型链上的下一项，链尾置结束值。只有真实入口经正常领取或启动表进入。以 `M → {C,V}` 为例，`M.next_aic = C`，`M.next_aiv = V`。若 `V` 的依赖包含 `C`，V 核仍从自己的链接准备 `V`，派发前等待保留下来的依赖。
2. **启动批。** 使用 READY 时已经满足全部活跃 fanin 的任务。首个区间入口已就绪且其适用前缀已证明完成时，可以由启动表提供该入口，随后按该入口的 START 和接续流程执行。
3. **尾部批。** 仅对 AIC、AIV 两条链共同以同一个全宽 `1c2v` 结束的区间规划。候选在最终序中位于尾任务之后，来自该段之后的普通单 block 和非全核 SPMD。用真实尾部 barrier 的完成证明加上「尾任务 DONE」计算剩余入度。选入剩余入度为零的任务，为执行核保留实际仍需等待的边。两条链须能走完全程，尾任务完成所需的工作必须可以在等待者恢复普通调度前执行。
4. **整项分配。** 每个启动或尾部锚点只预派发一波，每核最多一个子任务。候选按最终 rank 择取，单次静态分配须放下一个任务的全部 block 和所有实际角色，否则该任务保留普通入口。AIC 工作分给 AIC，纯 AIV 优先用 AIV1，再用 AIV0。每个 MIX block 的角色固定在同簇。选中任务不降低同一波内其他任务的入度。只在锚点所在的 Graph 分段内做有界扫描，遇到 Graph invocation 边界即停止。启动锚点在首段开头。扫描也须保留下一段真实区间入口的派发顺序。
5. **输出。** 给每个任务一个派发归属。普通任务和仍需普通领取的区间入口进入两张 order list。区间后继由静态链接取得，预派发任务由锚点取得，#3204 选中的任务由 terminal 执行。按最终列表及静态派发计划计算 layout，然后上传。

### Fanin 裁剪

Host 逐条处理原有 fanin，仅用全宽 SPMD 区间的 barrier 裁掉已覆盖的同步前驱。裁剪使用最终 rank 和实际途经的入口或静态链。前驱须严格早于 barrier，且其全部实际参与核都属于 barrier 的参与核集合，并在到达 barrier 前完成前驱的责任。后继须在该 barrier 放行后才访问这条依赖的数据。区间入口的证明来自 START，标记为区间后继的全宽 kernel 使用自身的握手。

| Barrier | 可覆盖的同步前驱 |
| ------- | ---------------- |
| 全宽 `1c` | 纯 AIC |
| 全宽 `2v` | 纯 AIV |
| 全宽 `1c2v` | 纯 AIC、纯 AIV 和 MIX |

在每个区间按核型记录实际经过的 barrier。前驱为普通纯核任务时，以入口门控证明其对应前缀已排空。前驱为普通 MIX 时，再要求覆盖它的所有实际角色，且这些角色均在更晚的 barrier 前执行完自身子任务。较早 MIX 的发布、发现和轮转仍按原有全局 rank、publication floor 和 sequence 处理。Graph shell 或 lane 外发布的任务用逻辑 DONE，可能 defer 的任务保留完成检查。

对每个目标，沿静态链传递已经过的 `1c`、`2v` 和共同 `1c2v` barrier 位置。纯核后继只继承本链已经经过的证明。共同 `1c2v` 握手产生的证明在握手后可沿两条链传递。逐条 fanin 查可使用的 barrier，只有 rank 早于该 barrier 且执行责任落入其覆盖范围时才省略。证据不全的边保留在设备行里。

- 直接 `1c2v → 2v` 保留对 `1c2v` 的等待，因为 `2v` 不覆盖它的 AIC。对称的 `1c2v → 1c` 同理。`2v → 1c2v` 若全部前驱 AIV 都直接接续到后继，后继握手可覆盖该边。
- `M0 → M1 → V` 中，`M1` 的全宽握手可以覆盖 `M0`，经过它的 `V` 可以裁掉对 `M0` 的直接依赖。对 `M1` 本身的依赖仍按 `M1` 的完成处理，或由更晚的 barrier 覆盖。
- 独立的 `C0,V0` 两入口各提供本核型覆盖。共同的 `1c2v` 握手可以汇合两条链。`C0,C1,M` 中如果 AIV 没有入口，区间止于 `C1`。
- 较小 rank 只用于定位已证明排空的前缀。沿 AIC 链派发的任务只有实际经过 AIV 覆盖的同步点后，才可以用 AIV 的证明。

设备只读取裁剪后的 `PercounterFanin` 行。行为空直接派发，Host 和设备均无需额外的跳过 fanin 标志。逻辑 DAG 保留，用于依赖导出、sink 发现、终段分析和 CntNotify chain 规划。同步输出及 completion credit 仍按现有发布顺序写入，barrier 的数据可见性和累计阈值语义须同时成立。

## 数据结构与镜像

### Host 计划

| 记录 | 字段和含义 |
| ---- | ---------- |
| 任务归属 | 每个 task ID 的最终 rank、派发路径和所属区间或锚点。逻辑 fanin 与裁剪后的设备 fanin 分开存。 |
| 区间 | 全局序范围、真实入口、AIC/AIV 链、共同末端。每个区间入口按列表记录较早普通任务的截止位置。 |
| 预派发批次 | 锚点、布局类型和静态分配。按物理 lane 记录下一个 task 与 block，未占用 lane 为 `invalid`。 |

`host/state.h` 在 `plan_layout` 前完成该计划。`host/region_plan.h` 按裁剪后的普通列表定义 segment、ticket interval、claim summary 和普通 MIX 边界。逻辑任务数、计数器、参数、sink 和 DFX 执行容量均按所有实际派发路径统计。

### 设备编码

| 位置 | 字段和编码 |
| ---- | ---------- |
| `PercounterCounter` 的 36 字节预留区 | 32 位派发标志区分入口和区间成员，两项 32 位 `next_aic`、`next_aiv` 使用 `UINT32_MAX` 表示结束。一个 64 位 `per_core_pre_dispatch_tasks` 表引用挂共同末端批，两个 64 位入口截止位置分别对应 AIC、AIV/MIX。保持 counter 128 字节及对齐。 |
| 入口截止位置 | 每项包含 32 位 segment 索引和 32 位排他 order position。定位到最终普通列表，`UINT64_MAX` 表示本核型无截止位置。 |
| `PercounterControl` | 只读启动表引用，保持全局 async 计数所在扇区的隔离。 |
| GM 预派发表 | 表引用为 8 字节对齐的镜像相对 offset，0 表示无表。低位表示启动按核型布局。该布局用两项 64 位字分别表达 AIC 和 AIV，block 按全宽映射推导。逐核布局按 `3 * cluster_idx + cluster_lane` 索引，使用现有 `percounter_order_entry(task_id, block_idx, block_num)` 编码。尾部固定使用逐核布局。task ID 为 `UINT32_MAX` 表示 `invalid`。 |
| 簇 SSBUF | 增加三项 32 位到达 token 和一项 32 位放行 token，放在 `PercounterSsbufRegion` 尾部预留区。参与角色只写自己的到达 token，最低活动角色写放行 token。0 表示未到达，启动批使用 1，尾部批使用尾任务的 `global_order_id + 2`。每轮在 READY 后、本簇首次派发前清零并发布初始化完成。 |

Host 验证 offset、对齐、索引范围、代次不溢出、block/role 唯一性，以及来源和目标的核型匹配。预派发各核经过的锚点顺序必须单调。每簇一波最多承载一个 MIX block，活动角色必须属于同一任务的同一个 block。普通 MIX 和 terminal 各自使用原先的 SSBUF 字段。

`plan_layout` 用实际普通列表计算 `order_count`、`boundary_task_count` 和 claim 区大小。START cohort 单独统计，包含普通入口及保留 START 的预派发和 terminal，区间内部无 START 资源。按实际 cohort 分配 `sync_index`、通知地址与 GM arrival，并校验各核型的最大宽度。将现有 `sync_cohort_count <= boundary_task_count` 检查改为分别校验实际 START 和 claim 数量。GM fanin 地址数组按裁剪后行数写入，延用 GM / CntNotify 地址和阈值编码。预派发表和只读字段纳入 `upload_size`，在 AICore 使用前统一发布。

## AICore 流程

### 区间入口

AIV0 照常领取并发布 MIX，参与核可提前发现并准备记录。对已规划的 `1c`、`2v` 和 `1c2v` 入口，每个实际覆盖的核都在入场承诺之前检查本核低 rank 工作。MIX 入口在发布 `ENTER_READY` 前检查，纯核入口在发布本核 START 到达前检查。双入口分别按各自的 rank 和覆盖核型执行。

1. **较早普通任务。** 普通槽中 rank 低于入口的任务先完成本核 kernel 并退休。它尚未 ready 时，继续正常调度及 fanin 轮询。检查 `source.segment`、入口的排他截止位置和 `revisit`。截止在 ticket 区间内时，确认 ticket 已覆盖该前缀。较早的领取责任全部处理后再报到。
2. **较早共享 MIX。** 检查 MIX publication floor 和 publication sequence，排除本核仍可能收到尚未发现的较早 MIX。按实际角色处理已发现的较早 MIX，包括已经锁存入场的本核执行快照。每个相应子任务须实际返回并完成本核退休。描述符被回收或计票耗尽只证明发布或领取责任结束。AIV0 即使不参与计算，也先完成自己较早的协调责任。
3. **放行。** 门控只推迟当前入口的承诺，较早的普通任务和 MIX 仍按现有规则推进。前缀通过后发出原有承诺并复用 `ENTER_IDLE` 和入口 START。START 到齐证明入口覆盖核上的较早同步工作已完成。例如 `2v` 要等每个参与 AIV 完成自己欠下的较早 MIX 子任务。若该 MIX 还包含 AIC，`2v` 对整个 MIX 的真实 fanin 继续等待整组 DONE。`defer_wait` 同样按逻辑完成判断。
4. **较晚任务。** 最终序中的区间连续，区间后继已从普通表摘除。在同一个核上，入口之外 rank 更高的候选属于区间之后。允许这些尚未承诺执行的普通任务和共享 MIX 保留派发状态，入口放行无需等待它们。较晚任务可能正依赖本区间，等待它释放会阻塞前驱。已发出的执行承诺继续按照原协议处理。
5. **交接。** 发出区间入口承诺后，本核只执行自己的静态链，较晚的普通票据继续留在本核槽里。若区间下一项不使用本核型，仍沿本核型链接走到下一项。链结束后恢复原有 source、slot 和 revisit，并按最新 publication sequence 发现共享 MIX。排空门控只用于入口，可在等待期间复查。链内接续直接使用已验证的静态归属。

### 区间接续和预派发

通过现有 dispatch helper 执行 kernel，沿用 argument context、错误传播、结果内存屏障及每个 block/role 的完成发布。

- **区间内。** 每核返回后先发布当前 block/role 的 completion credit。沿对应类型的链接准备下一项。用现有 meta 读取原角色 callable，按缓存的 `cluster_idx` 或展平 AIV block index 生成 context。读取设备 fanin 行，等待剩余边，再调用 kernel。参与核的下一步已固定，跳过 normal claim、候选选择、scheduler START 和普通 MIX 握手。
- **启动。** 每核完成运行初始化且观察到 READY 后，在普通调度循环前检查启动表。由此派发真实区间入口时，继续执行该入口的 START 和后续静态链。
- **收尾。** 每个角色从共同尾任务返回并发布自身 credit 后，检查该 tail 的逐核表。空表或 `invalid` 直接进入普通调度。有效条目使用裁剪后的 fanin 行，依赖尾任务自身就等待整组 DONE。尾任务握手已经覆盖的边可按计划裁掉，空行直接派发。每个锚点只消耗一次本核条目。
- **窄 MIX 与非全核 SPMD。** 每个 block 使用表中的物理 owner。MIX 活动角色在各自 fanin 满足、资源可用后发布本代到达 token。最低活动角色收齐本簇的到达，按需为该 block 计一次 START 并等待全局放行，再发布本簇的放行 token。不参与该 block 的 AIV0 可以继续普通调度。纯核的跨 block START 复用原按 block 计数。token 以 release/acquire 发布和观察，等待匹配本代或更新的 token。更新的代次证明写入者已履行旧代在本核的派发义务。同步失败与正常调度使用同一退出和错误路径。

### Defer、完成和退出

产生 `defer_wait` 的核继续处理本核挂链。若 `async_head` 非空，每次直接接续前做一次 `percounter_async_tick`，在等待 fanin、START 或参与角色时也推进本核轮询，并检查退出和错误。一次轮询后可继续派发就绪 kernel，使后续 kernel 能发出通知。

逻辑 task、block、role 和完成计数贯穿各路径。每个同步子任务发布一次 credit，defer 在条件满足后发布。各核完成本地静态派发义务再进入普通 quiescent 或 terminal。异常时走现有 `lane_error` 和 EXIT 路径。DFX 保留原 task、block 和角色，记录实际派发、kernel 启动和完成发布时点。

若实测单个长 kernel 显著延迟 defer 结算，后续可研究消费者协助轮询。需解决记录发现、唯一结算及无消费者任务的收尾。

### 与 PR #3204 的关系

[#3204](https://github.com/hw-native-sys/simpler/pull/3204) 在 bind 识别闭合终段，基本形态为全宽 `V → MIX → MIX`，可纳入更早的 bridge、gather 和全宽 `sync_start` MIX。用完整折叠 DAG 先决定终段唯一归属，压缩和预派发在其前缀内规划。终段前缀的 frontier 继续引用原逻辑完成计数，包含已摘表的任务。保留 `terminal_prefix_has_progress`、原 START 及本地排空合同。

## 实施阶段与验收

设计定稿后先验证 DSV4.1 性能基线，再分阶段实现。每阶段独立对照，先验证进展和结果，再作性能判断。后续阶段以通过验收的前一阶段为对照，最终与第 1 阶段的主线基线比较。

| 阶段 | 交付边界 | 通过条件 |
| ---- | -------- | -------- |
| 1. DSV4.1 性能基线 | 从无本方案改动的主线 `798fea3fb` 构建 Percounter，运行套件的七个原生用例。 | 每例按套件要求通过运行与输出校验，双卡用例具有 Device 会合证据。保存逐轮 L0 数据、七例均值及总和，并对每例复测以估计波动。 |
| 2. 入口 rank 检查 | 对 `1c`、`2v`、`1c2v` 的每个参与核检查较早普通任务和共享 MIX，继续使用现有列表和派发路径。 | 用显式门控覆盖较早任务未领取、已占槽但未就绪、MIX 延迟发布、描述符已回收但本核 MIX 尚未返回，以及 `X=2/4`。验证入口先等待较早任务，依赖入口的较晚持票任务可在入口后接续一次。以第 1 阶段为对照测入场检查开销。 |
| 3. 首次排序与区间接续 | 在 Graph 分段内生成候选序并线性检查换位，固定最终区间后生成链接及 order list，保留现有设备 fanin 检查。 | 覆盖跨深度前移与后移、DAG 边阻止换位、无依赖时仍保留 Graph 边界、连续合并多段及 `1c2v` 收尾。检查无边图的待查间隔数，以及连续合并时原始右段的边访问次数上界。验证 `C0,V0,M` 与 `C0,C1,M` 的入口可达性、唯一 owner 和退出行为。持票任务在本核链结束后才运行，区间内省去 scheduler START。 |
| 4. Barrier 裁剪 | 用入口及 kernel barrier 裁剪设备 fanin，保留逻辑 DAG。 | 覆盖直接 `1c2v → 2v` 保留边、后一个 `1c2v` 覆盖前一个 barrier、同步前缀与 deferred 前缀差异。以显式事件拖住一个角色，验证进入后继可以更早，跨核读取仍在实际同步之后。GM 和 CntNotify 两路径均校验数据与 DONE。 |
| 5. 启动预派发 | 增加启动表和同簇角色放置，覆盖全核、单 block 和非全核 SPMD。 | 两项布局和逐核布局均验证，含窄 MIX、部分宽度 START、启动负责首个区间入口和 `invalid` 回普通路径。检查候选止于 Graph 边界，完整分配失败保留原任务，验证 block/role 只执行一次和跨轮 token 复位。 |
| 6. 尾部预派发及集成 | 在共同 `1c2v` 末端挂逐核表，完成 #3204、defer 和 DFX 集成。 | 尾批覆盖窄 MIX、非全核纯核及单 block，Graph 边界后的候选保留原路径。拖住尾任务一个角色，分别验证空 fanin 直接接续和依赖尾任务的条目等待整组 DONE。同波有依赖的任务分开派发。已摘表任务作为 sink 或 terminal frontier 时正确结算，本核已有 defer 在相邻 kernel 间继续轮询。 |

Host 用例检查拓扑、列表、布局、剩余 fanin 和唯一派发归属。并发正确性用例用显式同步事件控制顺序，以超时终止故障。`a5sim` 验证流程，A5 上板确认真实内存可见性、跨角色同步、CntNotify 和退出行为。

### DSV4.1 性能基线与回归

使用套件默认的 `pypto_lib_latest` 七个用例。`swa`、`c2a_full`、`c1a_reindex`、`c1a_reuse` 和 `engram` 各用一张 A5 卡，`moe` 和 `layers_19_21` 各用两张卡。按 [构建、准备和运行说明](../../../tests/fdwic/dsv41_scheduler/README.md#build-from-source) 构建 Percounter 运行时及其 ABI 对应的 kernel 和 bundle，为双卡计时准备 Device start gate。用 `production/list_cases.py` 枚举七例，以 `run.sh` 的 `0 1 10` 参数运行 L0。构建及准备时选择 `--runtime percounter`，执行时传 `--runtime=percounter`。每个 Percounter 版本的单个用例独立通过 `task-submit` 申请其拓扑所需卡数，构建和结果汇总在锁外。

保留运行时 revision、实际加载产物、PTO-ISA/CANN、bundle、卡号和逐轮数据。以套件定义的 L0 AICore wall 为主指标，单卡对十轮取均值，双卡每轮先做 Device 会合再取较慢 rank 的时长，随后取十轮均值。六个回放用例校验输出。原生 MoE 按套件要求验证实际执行、Host 控制读取和完整 L0 端点。取回放的 `l0-timing.json` 与原生 MoE 的 L0 `result.json`，按未舍入的七例均值求和。完整口径见 [性能统计说明](../../../tests/fdwic/dsv41_scheduler/README.md#performance-presentation-contract)。

后续阶段在同卡、同输入和同一套 Percounter 业务 kernel 上逐阶段对照第 1 阶段和上一阶段。逐例报告 L0 均值、绝对差和变化百分比，按该用例的复测波动判断改善、退化或无明确变化。超出波动的退化需解释并复测。七例总和作为补充指标，最终总和需改善。Host bind、入口等待、kernel 间隙和 AIV0 的 MIX 发布间隙用于定位。L1/L3 另轮采集领取、fanin 和 START，辅以摘表任务数和裁剪边数。
