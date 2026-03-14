# Smolagents 学习手册

这份手册用于你在当前仓库里做源码级学习。
重点是阅读顺序、动手练习和测试验收。

## 0. 学习目标

完成后你应该能做到：

1. 讲清楚从 `agent.run()` 到 `final_answer` 的完整执行链路。
2. 用具体例子比较 `ToolCallingAgent` 和 `CodeAgent` 的取舍。
3. 安全地改一个小功能，并用测试验证。
4. 熟悉 model、tool、executor、memory 四个核心模块。

## 1. 高价值代码地图

建议按顺序阅读：

1. Agent 核心与主循环：
   - `src/smolagents/agents.py`
   - `class MultiStepAgent` 第 315 行
   - `run()` 第 557 行
   - `_run_stream()` 第 763 行
   - `class ToolCallingAgent` 第 1494 行
   - `ToolCallingAgent._step_stream()` 第 1651 行
   - `process_tool_calls()` 第 1768 行
   - `execute_tool_call()` 第 1904 行
   - `class CodeAgent` 第 2009 行
   - `CodeAgent._step_stream()` 第 2244 行

2. Model 抽象层：
   - `src/smolagents/models.py`
   - `class MessageRole` 第 168 行
   - `class ChatMessage` 第 185 行
   - `class Model` 第 519 行
   - `class LiteLLMModel` 第 1272 行
   - `class InferenceClientModel` 第 1523 行
   - `class OpenAIModel` 第 1713 行

3. Tool 协议与参数校验：
   - `src/smolagents/tools.py`
   - `class Tool` 第 154 行
   - `validate_tool_arguments()` 第 1461 行
   - `src/smolagents/default_tools.py`
   - `class FinalAnswerTool` 第 145 行

4. 代码执行器：
   - `src/smolagents/local_python_executor.py`
   - `class LocalPythonExecutor` 第 1767 行

5. 状态与可观测：
   - `src/smolagents/memory.py`
   - `src/smolagents/monitoring.py`

## 2. 7 天学习冲刺

每天 60-90 分钟。

### Day 1：项目骨架 + 入口链路

1. 阅读 `README.md`。
2. 跟读 `agents.py` 里的 `run()`。
3. 画一条序列：输入任务 -> 模型调用 -> 行动执行 -> memory 更新 -> 终止条件。

完成标准：

1. 能解释 `stream=True` 和 `stream=False` 的区别。
2. 能指出 max-steps 防护在哪里触发。

### Day 2：ReAct 主循环细节

1. 阅读 `_run_stream()`。
2. 梳理 planning、action、error handling 的连接方式。
3. 找出所有可能的终止路径。

完成标准：

1. 能说出至少 3 条终止路径。

### Day 3：ToolCallingAgent

1. 阅读 `ToolCallingAgent._step_stream()`。
2. 阅读 `process_tool_calls()` 和 `execute_tool_call()`。
3. 跟踪一次工具调用从 LLM 输出到 observation 日志的全过程。

完成标准：

1. 能解释何时并行执行工具，以及为什么这么设计。

### Day 4：CodeAgent

1. 阅读 `CodeAgent._step_stream()`。
2. 找到 parse -> 修正/清理 -> execute 三阶段。
3. 观察 `final_answer` 如何被检测并返回。

完成标准：

1. 能解释至少一个代码执行安全机制（如 import 限制等）。

### Day 5：Model 层

1. 阅读 `ChatMessage`、`MessageRole`、`Model`。
2. 对比 `LiteLLMModel`、`InferenceClientModel`、`OpenAIModel`。
3. 追踪 tool calls 在 model 输出中的表示方式。

完成标准：

1. 能描述统一的模型消息格式。

### Day 6：Tools + Validation + Memory

1. 阅读 `Tool` 和 `validate_tool_arguments()`。
2. 阅读 `FinalAnswerTool`。
3. 阅读 `memory.py` 的核心结构。

完成标准：

1. 能解释无效工具参数是如何被拦截的。

### Day 7：集成与微改动

1. 选择一个很小的改进点（日志、报错信息、注释等）。
2. 在一个模块里实现。
3. 新增或更新测试。

完成标准：

1. 相关测试全部通过。
2. 能说明改动为何安全。

## 3. 动手练习（可直接执行）

### Drill A：跟踪一次完整运行

1. 打开 `tests/test_agents.py`。
2. 从下面两个测试入手：
   - `test_fake_toolcalling_agent`（第 418 行）
   - `test_fake_code_agent`（第 466 行）
3. 为每个测试画出命中的 `agents.py` 方法链路。

预期结果：

1. 每个测试都能对应一条简短调用图。

### Drill B：工具参数校验边界

1. 打开 `tests/test_tools.py`。
2. 重点看：
   - `test_validate_tool_arguments`（第 954 行）
   - `test_validate_tool_arguments_nullable`（第 1035 行）
3. 用你自己的最小测试函数复现这些场景。

预期结果：

1. 你能说明 nullable/default/type mismatch 的行为差异。

### Drill C：流式输出

1. 打开 `tests/test_models.py`。
2. 阅读 `test_streaming_tool_calls`（第 544 行）。
3. 解释 stream delta 如何聚合成最终 tool calls。

预期结果：

1. 你能讲清楚流式聚合路径。

## 4. 验收命令

在仓库根目录执行：

```bash
pytest tests/test_agents.py -k "toolcalling_agent or code_agent" -sv
pytest tests/test_tools.py -k "validate_tool_arguments" -sv
pytest tests/test_models.py -k "streaming_tool_calls" -sv
```

更大范围的可选验收：

```bash
pytest -sv
```

## 5. 每日笔记模板

```text
日期：
今天目标：
我跟踪了什么：
我还不理解什么：
明天要验证的一个具体问题：
```

## 6. 完成标准（Definition of Done）

当你能做到以下三点，就算达到源码级熟悉：

1. 改一个 agent 行为并且不破坏现有测试。
2. 新增一个 tool 并正确通过参数校验。
3. 解释核心流程时能给出具体文件和方法，而不是泛泛描述。
