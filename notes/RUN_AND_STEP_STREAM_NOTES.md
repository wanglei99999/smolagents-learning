# run、_run_stream、_step_stream 三层关系笔记

这份笔记专门用来理清 `run()`、`_run_stream()`、`_step_stream()` 三层方法之间的职责边界。

这是理解 `smolagents` 执行框架最重要的一组关系。

## 1. 先给出一句话结论

可以把这三层理解为：

1. `run()`：准备一次运行
2. `_run_stream()`：驱动一次运行
3. `_step_stream()`：完成运行中的某一步

这是整个 Agent 执行结构的主干。

## 2. run() 是什么

代码位置：

1. [agents.py](e:\Coding\smolagents\src\smolagents\agents.py:557)

`run()` 的职责不是直接做推理，也不是直接执行工具。

它更准确的职责是：

1. 启动一次新的任务运行
2. 处理输入参数
3. 初始化本次运行需要的上下文
4. 决定使用流式还是非流式模式
5. 调用 `_run_stream()` 进入真正的执行循环
6. 把内部执行结果整理成最终返回值

你可以把它理解成“对外入口”。

它做的典型事情包括：

1. 设置 `self.task`
2. 处理 `additional_args`
3. 重置 `memory` 和 `monitor`
4. 写入 `TaskStep`
5. 给 `CodeAgent` 的执行器注入变量和工具
6. 根据 `stream` 决定返回生成器还是最终结果

所以 `run()` 关心的是“一次运行怎么开始，以及最后怎么返回”。

## 3. _run_stream() 是什么

代码位置：

1. [agents.py](e:\Coding\smolagents\src\smolagents\agents.py:763)

`_run_stream()` 是这套系统的执行主循环。

它不负责决定某一步到底怎么执行，但它负责管理整个运行过程。

它的职责包括：

1. 维护 ReAct 循环
2. 控制 `step_number`
3. 判断是否需要插入 `PlanningStep`
4. 为每一轮创建 `ActionStep`
5. 调用 `_step_stream()` 执行当前这一步
6. 处理异常
7. 判断是否已经得到最终答案
8. 到达 `max_steps` 后执行兜底逻辑
9. 最后统一产出 `FinalAnswerStep`

所以 `_run_stream()` 关心的是“一次运行如何被一步一步推进”。

你可以把它理解成“执行调度器”。

## 4. _step_stream() 是什么

代码位置：

1. [agents.py](e:\Coding\smolagents\src\smolagents\agents.py:1038)
2. [agents.py](e:\Coding\smolagents\src\smolagents\agents.py:1651)
3. [agents.py](e:\Coding\smolagents\src\smolagents\agents.py:2244)

`_step_stream()` 是“单步执行策略接口”。

真正这一轮怎么做，不在 `run()`，也不在 `_run_stream()`，而是在这里。

父类只规定：

1. 当前有一个 `ActionStep`
2. 你要完成这一轮动作
3. 你要把结果逐步写回 `ActionStep`
4. 如果拿到最终答案，要通过 `ActionOutput` 告诉上层

至于具体怎么做，由子类实现：

1. `ToolCallingAgent._step_stream()`
   模型生成工具调用，然后执行工具
2. `CodeAgent._step_stream()`
   模型生成 Python 代码，然后执行代码

所以 `_step_stream()` 关心的是“这一轮 action 到底做什么”。

## 5. 三层之间的关系

一次完整执行可以抽象成下面这样：

1. 用户调用 `run(task)`
2. `run()` 准备上下文并进入 `_run_stream()`
3. `_run_stream()` 开始主循环
4. `_run_stream()` 创建当前轮次的 `ActionStep`
5. `_run_stream()` 调用 `_step_stream(action_step)`
6. `_step_stream()` 真正执行这一步
7. `_run_stream()` 接收输出，决定是否继续循环
8. `run()` 整理最终返回值

所以职责链条是：

1. `run()` 管 run 的入口和出口
2. `_run_stream()` 管 run 的推进过程
3. `_step_stream()` 管某一步的实际执行

## 6. 为什么要这样分层

这套分层有很强的工程价值。

### 6.1 避免职责混乱

如果把参数处理、主循环控制、单步执行全塞进一个方法，会很难维护。

拆开后：

1. `run()` 负责 API 层
2. `_run_stream()` 负责流程层
3. `_step_stream()` 负责策略层

### 6.2 父类和子类职责清晰

父类 `MultiStepAgent` 只负责公共框架：

1. 记忆系统
2. 调度循环
3. 异常收尾
4. 生命周期管理

子类只负责自己的执行风格：

1. 是工具调用
2. 还是代码执行

### 6.3 易扩展

如果以后新增一种 Agent 类型，通常不需要重写 `run()` 和 `_run_stream()`。
只要实现新的 `_step_stream()` 即可接入整个框架。

这就是典型的模板方法模式。

## 7. 你当前的理解怎么表述更准确

你原本的理解是：

1. `run` 相当于启动任务，添加参数
2. `_run_stream` 的任务是调度任务
3. 真正执行某个任务在 `_step_stream` 中

这个理解整体是对的。

更准确的版本是：

1. `run()`：启动一次 run，准备上下文，并整理最终返回值
2. `_run_stream()`：调度整次 run 的执行循环
3. `_step_stream()`：执行 run 中当前这一个 action step

这里特别要注意：

1. `_step_stream()` 执行的不是“整个任务”
2. `_step_stream()` 执行的是“当前这一轮 step”

整个任务通常由多轮 `step` 共同完成。

## 8. 一句话压缩记忆

可以直接背这一句：

1. `run()` 管开始和结束
2. `_run_stream()` 管循环和调度
3. `_step_stream()` 管单步怎么做

## 9. 自测问题

如果这 4 个问题你能回答顺畅，说明这部分已经读懂：

1. 为什么 `run()` 不直接负责执行工具？
2. 为什么 `_run_stream()` 不直接写死工具调用逻辑？
3. 为什么 `_step_stream()` 只负责“一步”，而不是整个任务？
4. 如果新增一个新 Agent 类型，为什么通常只需要实现 `_step_stream()`？
