# Step 和 Memory 跟读笔记

这份笔记专门解释 `step`、`MemoryStep`、`PlanningStep`、`ActionStep`、`FinalAnswerStep` 这些概念。

适用场景：

1. 第一次读 `smolagents` 源码时建立整体心智模型
2. 看 `agents.py` 和 `memory.py` 时理清对象关系
3. 后续读 `ToolCallingAgent` 和 `CodeAgent` 时不再混淆“过程记录”和“真实执行”

## 1. 什么是 step

在这个项目里，`step` 不是普通日志，也不是某个函数的返回值。

更准确地说，`step` 是：

1. 一次 Agent 运行过程中的阶段性事件
2. 一个可以被记录、回放、序列化、重新提供给模型的过程单元

Agent 不是一次函数调用就结束，而是一个多轮循环过程：

1. 接收任务
2. 规划
3. 调模型
4. 执行动作
5. 观察结果
6. 重试或继续下一轮
7. 给出最终答案

如果这些过程不拆成一个个 `step`，系统会很难做好：

1. 运行历史记录
2. 下一轮推理的上下文构造
3. 调试和回放
4. token 和时间统计
5. 出错后的自我修正

所以 `step` 本质上是 Agent 的“过程单位”。

## 2. 什么是 MemoryStep

代码位置：

1. [memory.py](e:\Coding\smolagents\src\smolagents\memory.py:81)

`MemoryStep` 是所有 step 的抽象父类。

它的意义是：

1. 任何要进入 memory 的步骤，都必须满足统一接口
2. 这些步骤都必须能被序列化
3. 这些步骤都必须能转成下一轮喂给模型的消息

它定义的两个核心能力：

1. `dict()`
   用于结构化保存
2. `to_messages(summary_mode=False)`
   用于把步骤重新编码成 LLM 上下文

所以 `MemoryStep` 不是简单的数据类，它是一个“可进入 agent 记忆系统的步骤协议”。

## 3. 什么是 memory

在这个项目里，memory 不是一段松散的聊天记录。

它更像：

1. 一次运行中所有关键步骤的结构化历史

也就是说，Agent 记住过去，不是靠“我大概记得前面做了什么”，而是靠：

1. `TaskStep`
2. `PlanningStep`
3. `ActionStep`
4. `FinalAnswerStep`

这些 step 被依次追加到 `memory.steps`，然后在下一轮模型调用前，转换成消息列表。

简化理解：

1. `step` 是单个过程单元
2. `memory` 是这些过程单元的集合
3. `MemoryStep` 是这些过程单元的统一抽象

## 4. 为什么会有不同类型的 step

因为 Agent 的不同阶段，语义完全不同。

如果只用一个通用 `Step` 类，会有三个直接问题：

1. 字段混乱，很多字段只对部分场景有效
2. 很难根据类型做不同的消息序列化策略
3. 上层流程不容易判断“当前这一步到底是什么”

所以项目把不同语义的阶段拆成不同 step 类型。

核心类型包括：

1. `TaskStep`
2. `PlanningStep`
3. `ActionStep`
4. `FinalAnswerStep`

## 5. PlanningStep 是什么

代码位置：

1. [memory.py](e:\Coding\smolagents\src\smolagents\memory.py:211)

`PlanningStep` 表示一次规划事件。

它不是实际执行动作，而是告诉系统：

1. 当前我打算怎么解决任务
2. 我现在知道哪些事实
3. 接下来准备怎么推进

它的职责是“定方向”，不是“真正做事”。

为什么要单独拆出来：

1. 规划和执行不是一回事
2. 规划并不一定每轮都发生
3. 规划在序列化时可以有独立策略

例如在 `summary_mode=True` 时，旧的 planning 信息可以被省略，避免旧计划对新计划产生过强干扰。

## 6. ActionStep 是什么

代码位置：

1. [memory.py](e:\Coding\smolagents\src\smolagents\memory.py:90)

`ActionStep` 是最核心的步骤类型。

它表示：

1. 一轮真实的 ReAct 行动闭环

这一轮里通常会包含：

1. 发给模型的输入消息
2. 模型输出
3. 解析出的工具调用或代码
4. 执行后的 observation
5. 错误信息
6. token 使用统计
7. 是否已经得到最终答案

所以 `ActionStep` 不是“一个动作函数”，而是“一轮完整行动的归档容器”。

这也是为什么在 [`_run_stream()`](e:\Coding\smolagents\src\smolagents\agents.py:825) 中，每轮循环都会先创建一个 `ActionStep`，再由子类往里面填充内容。

## 7. FinalAnswerStep 是什么

`FinalAnswerStep` 表示一次运行的最终输出事件。

它的主要职责不是继续参与推理，而是：

1. 标志这次 run 已经结束
2. 把最终答案以统一结构对外暴露

它的存在保证了一个重要不变量：

1. 一次正常收尾的 run，最后一个 step 必须是 `FinalAnswerStep`

这也是 [`run()`](e:\Coding\smolagents\src\smolagents\agents.py:701) 中能直接断言最后一步类型的根本原因。

## 8. 一次 run 中 steps 大概如何增长

一个典型流程可能像这样：

1. `TaskStep`
   用户输入任务
2. `PlanningStep`
   第一次规划
3. `ActionStep`
   第 1 轮执行
4. `ActionStep`
   第 2 轮执行
5. `ActionStep`
   第 3 轮执行
6. `FinalAnswerStep`
   返回最终答案

注意：

1. `PlanningStep` 不一定每次都有
2. `ActionStep` 通常会有多轮
3. `FinalAnswerStep` 是稳定收尾标记

## 9. 为什么 ActionStep 必须能 to_messages

这是最容易被忽略的一点。

`ActionStep` 不只是给人看的日志，它还要变成下一轮模型的上下文。

也就是说，下一轮模型是否知道：

1. 上一轮调了什么工具
2. 工具返回了什么
3. 上一轮出现了什么错误

依赖的不是局部变量，而是 `ActionStep.to_messages()`。

所以它承担的是“双重角色”：

1. 运行记录
2. 下一轮推理输入

## 10. 为什么 FinalAnswerStep 不能和 ActionStep 合并

因为两者职责不同。

`ActionStep` 代表一轮推理和执行闭环。  
`FinalAnswerStep` 代表一次 run 的正式结束。

如果硬合并，会带来几个问题：

1. 语义变混乱，不知道这是过程步骤还是结束标记
2. 上层很难稳定判断“最后结果在哪里”
3. 生命周期处理会变复杂

单独保留 `FinalAnswerStep`，可以让整个执行协议更清晰。

## 11. 为什么 planning 不能只存在局部变量里

因为 planning 不是临时草稿，它会影响后续推理。

如果不进入 memory：

1. 后续模型调用看不到之前形成的计划
2. 回放时无法知道 Agent 当时的思路
3. 调试时无法区分“计划错了”还是“执行错了”

所以 planning 必须是结构化步骤，而不是一次函数内的临时字符串。

## 12. 一句话总结

你可以这样记：

1. `step` 是 Agent 运行过程中的结构化阶段
2. `MemoryStep` 是所有可进入记忆系统的步骤基类
3. `PlanningStep` 负责计划
4. `ActionStep` 负责行动闭环
5. `FinalAnswerStep` 负责收尾和最终答案

如果把整个 Agent 看成一个有状态的推理机，那么：

1. `memory` 是历史
2. `step` 是历史中的单元
3. 不同 step 类型描述不同运行语义

## 13. 自测问题

如果下面 4 个问题你都能顺畅回答，说明这部分已经基本读懂：

1. 为什么这个项目不把历史只保存成纯文本？
2. 为什么 `ActionStep` 既是记录，也是下一轮上下文来源？
3. 为什么 `FinalAnswerStep` 要单独存在？
4. 为什么 planning 要进入 memory，而不是只放在局部变量？
