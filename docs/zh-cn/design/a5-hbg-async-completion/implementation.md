# A5 HBG 异步完成实施方案

[返回总览](../a5-hbg-async-completion.md)。实现以可运行的非阻塞等待为首要闭环，复用既有注册、参数、Graph 和资源 owner。

## 1. 修改边界

| 模块 | 职责 |
| ---- | ---- |
| A5 HBG Host planner/runtime maker | 根据现有图预分配完成记录，初始化稳定 task/block/lane 映射 |
| kernel context 与登记 helper | 保留地址式登记接口，写入本执行的稳定记录，检查容量与首错 |
| 执行返回与 completion/MIX 收尾 | 封口登记，释放实际执行资源，贡献 block/group 完成义务 |
| CHECK | 任意参与核领取条件记录，每次有界检查，保存进度并交回执行机会 |
| FINISH 与 resolver | 唯一发布 TaskDone，分段路由后继，排空发布后贡献 resolved_task_count |
| 已有 run/domain owner | 保持 heap、staging、导入 backing、通信 context/window/workspace 至实际退休 |
| 后端与消费者获取 | 验证实际 SDMA 提交、条件读取和消费者缓存路径，局部修补具体缺口 |

公共 Host 注册、ChipWorker 参数传输、A2/A3、TMR、TensorData 和 Graph Definition 不因 resident defer_wait 增加新协议。共享问题只有在目标用例证明已有路径不足时才单独修复。

## 2. 首个闭环

1. waiter 通过既有 helper 登记 COUNTER 后正常返回。
2. 返回路径发布稳定 group，释放 slot 及局部 block/MIX 义务；逻辑任务仍未完成。
3. 后提交的同类核 publisher 获得执行机会并更新 signal。
4. CHECK 有界地观察到条件满足，与计算返回义务汇合。
5. FINISH 发布唯一逻辑完成并放行消费者；消费者实际读取通知对应的数据。

首要测试取实际同类执行资源数 C，分别提交 C−1/C/C+1 个 waiter，然后提交 publisher。publisher 不依赖 waiter；消费者依赖 waiter。必须证明 publisher 可运行、未通知时消费者未执行、通知后消费者结果正确。仅有编译、同步回归或 mock 条件测试不算闭环完成。

## 3. 服务与完成顺序

普通业务与控制工作共享服务额度：最多 4 次业务返回后服务一次 CHECK/FINISH；空闲和准入受阻时仍提供控制服务。CHECK/FINISH 轮转，每次最多领取一个记录，基础工作预算为 8。PENDING 或失败 CAS 不计为真实进展，不延长 watchdog。

物理执行返回与逻辑任务完成独立。SPMD 的全部 block、MIX 的全部 active lane 必须按真实释放顺序记账；零登记及合法 skip 也必须恰好结算。最后一次合法原子汇合才有 FINISH 激活权。

FINISH 保存完整 resolver 游标及最多 8 项发布批次，跨核恢复不重放已发生的队列发布。终态记录和目录清理完成后，才贡献 run 完成计数；正常退出不能越过尚未收尾的异步记录。

## 4. 完整覆盖顺序

| 验证层 | 必须覆盖 |
| ------ | -------- |
| 基础 COUNTER | 延迟通知、C−1/C/C+1、独立业务推进、无提前消费者、0/1/64 条条件 |
| 执行形状 | 普通 AIC/AIV、SPMD、MIX 3/5/6/7、sync-start、predicate true/skip、Graph 多实例及普通任务混排 |
| 真实后端 | A2/A3 现有通信用例适配 A5 HBG；2/4 rank；COUNTER 与 SDMA 的实际完成记录 |
| 资源与可见性 | run/domain 持有、跨 run 复用、同 run channel/session 并发、实际消费者旧 cache |
| 异常 | 登记溢出、非法记录、提交失败、超时、核退出失败、pending close、DMA 最后访问证明 |
| 回归与性能 | 新增和受影响原有 A5 HBG 用例通过，再执行同机同配置基线性能对照 |

已有 API 或后端已经保证的行为用调用链和实际用例验证；缺口修复限定在对应层，不把所有安全要求合并成新的通用资源协议。

## 5. 交付与诊断

设计、实现与验收保持一致。开发记录和原始日志独立存放，文档不携带过程复述。阶段完成后及时提交、总结并继续后续工作；清理已有历史时按用户要求合并提交，之后按可验证单元组织。

复用已有 DFX 输出和协议观测入口。必要事件关联 run/task/block/lane、登记封口、实际执行释放、条件满足、逻辑完成及 resolve 排空；不在热路径逐次打印 PENDING，也不为了观测扩展用户可见接口。
