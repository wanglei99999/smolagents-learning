# Smolagents 源码跟读助手

这不是阅读清单，而是跟读模板。
目标不是“知道有哪些文件”，而是“知道这段代码为什么这样写、状态怎么流动、哪里容易出错”。

## 1. 跟读方法

每次只跟一个核心方法，固定回答下面 7 个问题：

1. 这个方法在整个系统里的职责是什么？
2. 它的输入是什么？哪些参数会改变行为？
3. 它会读写哪些对象状态？
4. 它调用了哪些下游方法？
5. 它依赖哪些上游约束？
6. 它的异常路径是什么？
7. 它的输出为什么设计成现在这样？

如果这 7 个问题说不清，就说明还没有真正读懂。

## 2. 跟读顺序

建议严格按下面顺序：

1. `MultiStepAgent.run()`
2. `MultiStepAgent._run_stream()`
3. `ToolCallingAgent._step_stream()`
4. `ToolCallingAgent.process_tool_calls()`
5. `ToolCallingAgent.execute_tool_call()`
6. `CodeAgent._step_stream()`
7. `ActionStep` / `PlanningStep` / `FinalAnswerStep`
8. `Model.generate()` 的输入输出协议
9. `Tool` 的定义与参数校验
10. `LocalPythonExecutor`

原因很简单：先抓主流程，再抓分支，再抓底层契约。

## 3. 示例：如何跟读 `run()`

文件位置：

1. `src/smolagents/agents.py:557`

### 3.1 职责

`run()` 不是“执行智能体”的全部逻辑。
它更准确的职责是：

1. 准备本次运行的上下文。
2. 决定执行模式。
3. 调用真正的执行循环 `_run_stream()`。
4. 把内部执行结果整理成对外 API 的返回值。

所以它是“入口编排层”，不是“推理层”。

### 3.2 输入

关键参数有：

1. `task`
   本次任务文本，是最核心输入。
2. `stream`
   决定返回生成器还是最终结果。
3. `reset`
   决定是否延续之前的 memory 和 monitor 状态。
4. `images`
   提供多模态输入。
5. `additional_args`
   把外部变量注入到 agent state。
6. `max_steps`
   控制 ReAct 循环上限。
7. `return_full_result`
   决定返回裸结果还是 `RunResult`。

### 3.3 状态写入

`run()` 主要会改这些状态：

1. `self.task`
2. `self.interrupt_switch`
3. `self.state`
4. `self.memory.system_prompt`
5. `self.memory.steps`
6. `self.monitor`
7. `self.python_executor` 内部上下文

这一步很关键。读懂一个方法，首先看它改了谁。

### 3.4 关键设计

#### 设计 1：`additional_args` 同时写入状态和任务文本

它不是只做 `self.state.update(additional_args)`。
它还把变量名拼进任务文本。

原因是：

1. Python 执行器需要真实变量。
2. LLM 也需要知道这些变量存在。

只写状态不告诉模型，模型不知道能不能用。
只告诉模型不写状态，执行时又拿不到变量。

这就是典型的“双通道同步”设计。

#### 设计 2：`run()` 始终把任务写进 memory

`TaskStep` 会被 append 到 `self.memory.steps`。

原因是后续 `_run_stream()` 在构造 LLM 输入时，不是直接拿 `task` 参数，而是依赖 memory 序列化。
也就是说，这个项目把“任务”视为 memory 的第一步，而不是独立变量。

#### 设计 3：CodeAgent 在入口阶段就要同步变量和工具

如果当前 agent 带 `python_executor`，这里就会调用：

1. `send_variables()`
2. `send_tools()`

这意味着代码执行上下文不是在“代码生成后”才初始化，而是在 run 入口就准备好。
好处是执行器上下文和本次 run 强绑定，不会等到中途才发现缺工具或缺变量。

#### 设计 4：统一靠 `_run_stream()` 驱动

无论 `stream=True` 还是 `stream=False`，底层都走 `_run_stream()`。

区别只是：

1. 流式：直接把生成器交给调用方。
2. 非流式：内部把生成器消费完。

这能避免维护两套执行逻辑。

### 3.5 异常与边界

`run()` 自己不处理太多业务异常。
它假设：

1. 真正的执行错误在 `_run_stream()` 内部发生。
2. 最后一项必须是 `FinalAnswerStep`。

所以这里有一个关键断言：

1. `assert isinstance(steps[-1], FinalAnswerStep)`

这其实是在保护内部协议：
“只要一次 run 正常结束，最后一步就必须是最终答案。”

### 3.6 返回值设计

默认返回最终答案。
只有在 `return_full_result=True` 时，才返回 `RunResult`。

这是一个很实际的 API 取舍：

1. 普通用户只关心答案。
2. 调试、评估、可观测场景才需要完整步骤和 token 统计。

### 3.7 读懂 `run()` 的标准

你读完后，至少要能回答：

1. 为什么 `run()` 不直接做推理？
2. 为什么 `additional_args` 要同时影响 `state` 和 `task`？
3. 为什么 `TaskStep` 要尽早写入 memory？
4. 为什么流式和非流式都复用 `_run_stream()`？

## 4. 你接下来该怎么读

下一步直接读：

1. `src/smolagents/agents.py:763` 的 `_run_stream()`

读它时重点只看 4 件事：

1. 循环何时开始、何时停止。
2. `PlanningStep` 何时插入。
3. `ActionStep` 如何生成并落库。
4. 最大步数到达后如何兜底生成最终答案。

## 5. 高质量跟读的标准

如果你只是能复述“这里调用了某个函数”，那不算读懂。

你至少要能说出：

1. 为什么这里非得这样设计。
2. 如果删掉这一步会坏什么。
3. 这个方法和上下游方法之间的契约是什么。
4. 这个方法主要保护了什么不变量。

只有达到这个层次，才算真正理解项目。
