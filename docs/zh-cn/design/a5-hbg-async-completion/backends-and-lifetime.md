# A5 HBG 异步后端与资源生命周期

[返回总览](../a5-hbg-async-completion.md)。延迟逻辑完成、资源持有、数据可见性是三个相互关联的要求，分别验证，优先复用现有机制。

## 1. 条件观察

COUNTER 为 INT32 单调通知，expected 在 `[0, INT32_MAX]`。观察使用合法对齐的精确 32-bit device load，不清零通知，不用 atomic-add-zero 读取外部地址。一次检查返回 READY/PENDING/FAILED，内部不循环等待。

SDMA 使用当前 SDK 公共完成记录 API 返回的 post-done 地址和完整 post id。复用已有 session、postId 延续及内部槽复用保护，不解析私有 SQ 布局、不自行清零共享完成记录。workspace 完整范围采用公共 `kSdmaWorkspaceBytes`，但 allocation 范围不代表其中每个地址都是完成记录。

提交资格与观察资格分开验证。参与调度的 AIC/AIV 均须能观察支持的条件；不能把仅在 AIV 上可用的提交 helper 原样搬到 AIC。编译与反汇编只证明指令形态，真实可见性仍需硬件用例。

## 2. 提交与登记

SDMA adapter 保持现有调用接口。提交前检查形状、字节数和明确的零工作，并为可能产生的完成记录预留足够容量；提交后使用公共记录数量/索引 API 登记实际条件，归还未用预留。非零请求得到零 handle、未知完成类型、登记溢出或 helper 失败必须锁存错误，不把它们当成零登记成功。

同 run 内多个 task、物理 worker、AIV lane 使用同一 channel/session 的独占和并发规则必须以实际 SDK 路径验证。defer_wait 不会使提交队列满、SyncAll 或 kernel 内轮询自动让出；kernel 必须正常返回，调度器才重新取得执行资源。

## 3. 复用已有 owner

| 资源 | 既有持有机制与所需最后访问 |
| ---- | -------------------------- |
| HBG heap、staging、调度 image | run 持有，直到所有逻辑完成、resolve 排空及参与核退休 |
| 调用参数 backing | 已有 source/import 引用及 run 记录，覆盖执行与数据交付 |
| 通信 context/window/workspace | CommDomain 使用登记和 run fence；scalar `device_context()` 同样登记使用 |
| DMA 源 | 引擎最后一次读取完成 |
| DMA 目标 | 最后写入和相关消费者读取完成 |
| signal、完成记录 | 检查者及 backend/peer 都不再访问 |

延迟 TaskDone、后继解析和 resolved_task_count 会自然延长既有 run 持有时间；kernel 返回或单 task 完成不会单独释放 HBG heap。不能把 scalar context 当成无人保活，也不能用一份新身份快照代替实际引用。

`owner_task_id` 表示依赖/生产来源，可用于现有记录索引，不等于 allocator 或通信 owner。Graph 的 PARAM、SUB_TASK 和 view 几何已经提供绑定线索；需要消费范围时从现有表示解析运行内旁表，不修改 TensorData 或 Graph 格式。

## 4. 复用与可见性

generation 只识别旧引用，不能阻止旧 DMA 写入已经复用的地址。资源保持分配也不自动保证 payload/signal 未被覆盖。复用必须等待旧访问和实际消费者结束：run 内由正确的 DAG/后端约束保证，跨 run/peer 由既有 fence 或明确的完成确认保证。

依赖满足不等于所有核 cache 已更新。优先验证已有 backend 发布、实际消费者读取指令和缓存维护路径。若这些已提供获取保证，不增加多余准备协议；否则只为缺失的读取范围和核增加有界获取。观察核的 invalidate 不能代替消费核的获取。

## 5. 错误和关闭

首错停止新的正常发布和业务准入，但不能假定已提交 DMA/peer 访问随之停止。不能靠清零 B/G、目录为空、超时、普通 stream drain 或核 ACK 宣告外部访问结束。

正常路径复用现有 run/domain 退休。异常路径须核实当前后端究竟提供什么停止或完成保证；仅覆盖其他引擎的 drained 判断不能证明 SDMA 停止。证明不足时复用现有资源隔离与保留机制，禁止提前释放和复用，并返回原错误。

不为本功能预设新的通用关闭回执 ABI。确有现有错误路径无法表达的缺口时，先给出可复现的实际访问问题，再在对应 owner/backend 做局部修复。2/4 rank 验收必须覆盖 peer 最后访问，不能用本地 reset 冒充远端停止证明。
