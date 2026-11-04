# A5 HBG 异步后端与资源生命周期

[返回总览](../a5-hbg-async-completion.md)。本文定义条件判据、真实提交结果、payload 可见性、资源持有和关闭行为。scheduler 只消费这些合同，不解析后端私有队列布局。

## 1. 条件与能力矩阵

| 条件 | 完成判据 | 资源来源 | 不允许的行为 |
| ---- | -------- | -------- | ------------ |
| COUNTER | 本 rank 的 INT32 signal 值达到 expected，比较为 Ge | signal Tensor 所属通信窗口及槽位绑定 | 因历史 engine 标签为 SDMA 而强制要求 workspace；检查器清零 signal |
| SDMA post-done | 当前 workspace generation 的 64-bit completed post id 达到 expected post id | 已有通信资源 owner 提供的 SDMA workspace 和公共完成记录 | 猜测私有队列布局；清零共享 post-done；把任意零 handle 当成功 |

COUNTER 的 expected 在 `[0, INT32_MAX]`，live generation 内只能单调推进。SDMA expected 使用后端实际允许的 post-id 范围，不能因 token 字段是 64-bit 就假定所有 64-bit 值都可编码。溢出前必须在相关使用全部结束后建立新 generation，禁止回绕。

以下是交付所需的能力矩阵，验证通过前不能把对应组合标为已支持：

| 操作 | 要求 |
| ---- | ---- |
| 登记 COUNTER | 所有支持执行形状中的合法 AIC/AIV writer |
| 发起 SDMA | 由 backend 声明并验证的提交核；首批以 AIV 的实际 SDK 路径建立闭环 |
| 观察 COUNTER32 | 所有参与调度的 AIC/AIV；覆盖自核与异核观察 |
| 观察 SDMA64 | 所有参与调度的 AIC/AIV；观察不依赖原提交核的栈、AsyncSession 或 UB |
| 消费 payload | 实际消费者核完成自己的数据获取合同，覆盖与提交/观察核不同的情形 |

提交资格不等于观察资格。若某 AIC/AIV 不能使用需要 UB 的 helper，不把该 helper 直接搬过去；按[选型文档](choices.md)验证设备直接读取或相应 backend 能力。目标全核观察不通过时，该功能组合不具备交付资格。

## 2. adapter 提交与登记事务

### SDMA

1. 从当前 task 参数和通信资源绑定取得源、目标、workspace 及其使用关系，验证提交核与物理 channel group 的独占映射。
2. 使用公开 Tensor/descriptor 信息检查地址、受支持布局、源/目标大小、字节数乘法溢出和明确的零工作分支；相应检查必须与 backend 实际传输量一致。
3. 根据已验证的 backend 配置计算本次最多产生的完成记录数并预留条件槽。一个 DMA 请求可能产生多个 queue-prefix record，不能按一次调用一条条件计算。
4. 发布源数据，处理目标旧 dirty cache，并确认所有资源使用已经持有，随后才允许提交。
5. 通过匹配接口版本的公共 `CompletionRecordCount/At` 取得记录，检查类型、地址、expected、实际条数与预留上界，连同 binding index 一起 commit。
6. 归还未使用预留。只有全部必要条件已登记或明确合法零工作，提交 helper 才返回成功。

合法零工作在提交前处理，不依赖零 handle 猜测。已知非零请求若返回零 handle、未知记录类型或异常条数，锁存失败，禁止成功 TaskDone；未能证明没有副作用时继续持有 UseSet。

预留条件空间不等于 SQ/flag 槽没有背压。后端提交内部的等待属于 kernel 执行成本；不能在提交后随意返回 WOULD_BLOCK 并重跑有副作用的 kernel。若要提供非阻塞提交，需要另有可判定副作用和提交续接的后端合同。

每次提交/登记入口先检查已有错误，组内错误只保留第一次，后续请求不能覆盖它或继续提交。即使业务 kernel 忽略 helper 的失败返回，runtime 封口检查也必须终止成功路径。

第一版从已有 A5 通信资源 owner 绑定 workspace，不新增独立 provider。共享 workspace 的地址、代次和最终使用由实际 owner 管理，不能因两个 domain 都引用它就分别释放。逻辑 block、物理 worker、AIV lane 与 DMA channel group 的映射必须有独立验证。

### COUNTER

`save_expected_notification_counter` 使用信号窗口 binding 和已验证 offset 登记纯观察条件，不发起 DMA，不需要硬件提交预留事务。多个 peer 的条件逐条登记并检查容量；未知类型、越界或非法 expected 必须失败。

PyPTO 保持注册专用 waiter：登记开始后只能继续登记或执行允许的纯标量记账，不进行 payload、缓存或通信操作；predicate、early-resolve 和嵌套 task-launch/SPMD 的既有门禁不因 runtime 扩展而自动撤销。

## 3. 数据可见性与条件观察

`observe_once` 对一个条件执行有界检查，返回 PENDING、READY 或 FAILED，内部不自旋直到满足。READY 只证明该类型的完成判据成立；group 在完成必要的数据获取步骤前仍不能结算。

```text
源数据发布 / 目标旧缓存处理
  → DMA 或远端 payload 写入
  → backend 完成记录或 notify 发布
  → 观察核读取条件并完成获取
  → group 与执行释放汇合
  → TaskDone / 后继发布
  → 实际消费者核获取数据
  → 消费者读取 payload
```

控制记录的原子顺序不能替代这条数据链。外部条件地址不能使用 runtime 的 atomic-add-zero 观察，以免向 DMA/peer 写入的地址发起非合同内 RMW。读取使用精确宽度和合法范围；若实现以更宽容器读取，必须先验证容器对齐、完整范围和字段提取语义。

缓存维护按实际 cache line 扩展后的范围核对所有权。并发任务即使写不同字节，也不能默认安全地共享会被整行写回的 dirty line；不满足范围/所有权合同的接入必须拒绝或由已有安全 backend 路径处理。

`acquire_completed` 的进度保存在 group，执行发布侧必要的获取和可见性动作；`before_consume` 由[消费准备协议](consumer-preparation.md)在实际消费者核执行。Host 通过发布用途把条件与 payload 范围关联，为同步消费者及经过 Graph control 的消费者也生成计划；不能用 `may_defer` 或 signal 地址决定是否需要获取。backend 必须提供范围粒度、核/读取路径资格和有界步骤。

处理大范围数据时必须分段或独立计价，不能把整段缓存维护算成一个固定成本条件读取。分段进度绑定实际执行核与 slot generation，结束前不取得最终执行许可；MIX/sync-start 还需汇合全部实际成员的准备结果。无需额外维护只在具体读取路径获得 NOOP 资格后成立，不能默认空实现或每个任务无条件清整份 cache。

单调条件允许缓存已满足结果，前提是资源代次稳定、记录不会重置、payload/signal 未被下一 epoch 非法覆写。这些前提由绑定和使用协议保证，不由 `counter >= expected` 推导。

## 4. 资源持有与槽位复用

基础实现保守持有整个静态 UseSet，直到正常 run 退休且对应 backend 的最后访问合同成立。申请、持有、释放沿用现有 run/domain owner；设备记录只携带索引和身份，不自行释放资源。

| 资源 | 至少覆盖的最后使用 |
| ---- | ------------------ |
| DMA 源 | 最后一次 DMA 读取 |
| DMA/远端写目标 | 最后写入及所有相关消费者读取 |
| signal / completion record | 所有 waiter 与 backend 最后访问 |
| workspace / 通信窗口 | 本地、backend 和 peer 的全部相关使用 |
| group / FINISH / 参数及诊断 backing | 最后记录写入、resolve drain 和全部参与核退出 |

kernel 返回后仍被引擎引用的地址不能指向栈或可复用 UB scratch。scratch 若只用于提交准备，其复用安全性由 backend 明确证明；不能仅因提交函数返回就作假设。

同一可写 payload/signal 槽首版不允许不同 epoch 重叠使用。槽位状态为：

```text
Available → Writing(epoch) → Published(epoch) → Consuming(epoch) → Available
```

只有旧 epoch 的所有本地消费者和 peer 使用结束，才允许下一 Writing。run 内可由真实消费者到下一写者的 DAG 依赖约束；跨 peer 需要通信协议实际提供的确认。分配仍存活只证明没有 free，不证明旧数据未被覆盖。

没有安全复用证明时采用独立 payload 与 signal 槽，或拒绝复用请求；仅双缓冲 payload 而共享可乱序通知 counter 不够。独立槽也不能免除最终释放时的最后访问证明，不新增全局 Host barrier 代替这些资源级约束。

## 5. 错误、超时与 close

首错保存在 run 的独立 latch，包含完整 task/group/条件身份、操作阶段和资源 generation。观察到错误的路径停止新的业务准入、DMA 提交和成功发布；已运行的 kernel 或已接受的外部操作仍可能继续访问资源，不能假定它们已经停止。

| 情形 | 处理 |
| ---- | ---- |
| bind 或提交前验证失败，证明无设备副作用 | 撤销未使用预留和本次未提交的资源持有，返回明确错误 |
| kernel 内登记溢出、非法条件、非零提交异常 | 锁存错误；不封装为零登记成功，不重放 kernel |
| 已尝试提交，副作用未知 | 保留完整使用关系，停止后续成功路径，进入错误收敛 |
| 条件长期 PENDING | 按真实进展 watchdog 超时；轮询次数不刷新进展 |
| CHECK/FINISH owner 故障 | 不覆盖 OWNED，不补偿完成计数，不复用记录；走 run 退出与资源隔离 |
| pending 时正常 close | 停止接收新调用，收敛已接受工作，取得最后访问证明后释放 |
| 错误 close | 证明停止提交、核退休和 backend/peer 最后使用；证明不足则保留 owner/backing、拒绝复用并报告失败 |

整个 run 的成功结果必须同时满足正常完成计数、无首错、既有核退出及资源退休合同。错误 run 不通过将剩余 B/G 清零来完成正常路径。首错之后发现的清理失败保留为附加诊断，不覆盖原始错误。

AICore ACK、普通 stream drain、目录为空、超时和失败返回都不是外部 DMA/peer 已停止的通用证明。只使用实际存在且已验证的 backend 停止/完成能力；没有证明时选择保留与隔离，不能臆造 cancel/drain API。
