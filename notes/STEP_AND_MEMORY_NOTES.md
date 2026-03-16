# Step 和 Memory 跟读笔记

这份笔记专门解释 `step`、`MemoryStep`、`PlanningStep`、`ActionStep`、`FinalAnswerStep` 这些概念。

适用场景：

1. 第一次读 `smolagents` 源码时建立整体心智模型
2. 看 `agents.py` 和 `memory.py` 时理清对象关系
3. 后续读 `ToolCallingAgent` 和 `CodeAgent` 时不再混淆“过程记录”和“真实执行”

## 0. 核心理解：ReAct 框架是整个系统的灵魂

在深入 step 和 memory 之前，必须先理解 **ReAct 框架**。

### 0.1 什么是 ReAct

ReAct = **Re**asoning（推理）+ **Act**ing（行动）

完整循环：**Think（思考）→ Act（行动）→ Observe（观察）→ 重复**

这不是理论概念，而是 smolagents 的核心执行模式。

### 0.2 ReAct 在代码中的体现

**循环框架**：在 `_run_stream()` 方法中

```python
def _run_stream(self, task, max_steps):
    while not done and step <= max_steps:
        # === Think（思考）===
        # 准备上下文（包含历史观察）
        messages = self.write_memory_to_messages()
        
        # === Act（行动）===
        action_step = ActionStep()
        for output in self._step_stream(action_step):
            # _step_stream 内部：
            # 1. 调用 LLM 思考
            # 2. 解析输出
            # 3. 执行工具/代码
            yield output
        
        # === Observe（观察）===
        # 将执行结果写入记忆
        self.memory.steps.append(action_step)
        # 下一轮循环时，LLM 会看到这次的观察结果
```

**关键点**：
- Think 发生在 `_step_stream()` 调用 LLM 时
- Act 发生在 `_step_stream()` 执行工具/代码时
- Observe 发生在 `memory.append()` 时

### 0.3 为什么 Observe 只需要 memory.append()？

这是最容易被误解的地方。

**Observe 的本质**：不是"做什么"，而是"记录什么"，为下一轮推理提供信息。

```python
# === 第1轮 ReAct ===
# Think: LLM 决定搜索天气
# Act: 执行 web_search("Paris weather")
# Observe: memory.append(observations="Paris: 20°C")  # 👈 只是保存

# === 第2轮 ReAct ===
# Think: 构建上下文
messages = memory.to_messages()
# messages 现在包含：
# - 第1轮的 observations: "Paris: 20°C"  # 👈 从记忆读取
# LLM 看到第1轮的结果，基于此继续推理
```

**Observe 的两个核心作用**：
1. **保存**：`memory.append(observations=result)`
2. **构建上下文**：下一轮通过 `memory.to_messages()` 读取，转换为 LLM 的输入

这就是 ReAct 的闭环：每轮的观察成为下一轮的输入。

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

## 12. ReAct 框架的完整实现总结

### 12.1 ReAct 三阶段在代码中的映射

| ReAct 阶段 | 代码位置 | 具体实现 |
|-----------|---------|---------|
| **循环框架** | `_run_stream()` | `while not done:` 循环 |
| **Think（思考）** | `_step_stream()` 开始 | `model.generate_stream(messages)` |
| **Act（行动）** | `_step_stream()` 中间 | `execute(code)` 或 `tool(**args)` |
| **Observe（观察）** | `_run_stream()` 结尾 | `memory.steps.append(action_step)` |
| **记忆传递** | `write_memory_to_messages()` | 将观察转为下一轮的输入 |

### 12.2 关键设计理念

1. **Think 阶段**：LLM 看到的是完整的历史（包含之前的 Observation），基于此进行推理
2. **Act 阶段**：根据 LLM 的推理结果，实际执行工具调用
3. **Observe 阶段**：只是保存结果，真正的"观察"和"理解"发生在下一轮 LLM 的 Think 中

### 12.3 为什么 Agent 能自我纠错？

```python
# === 第1轮：错误的尝试 ===
# Think: "我用 calculator 计算"
# Act: calculator("2+2")
# Observe: memory.append(error="calculator 需要两个参数")

# === 第2轮：LLM 看到错误，自我纠正 ===
messages = memory.to_messages()
# messages 包含：
# {"role": "tool_response", "content": "Error: calculator 需要两个参数"}

# Think: "哦，我需要分开传参数"
# Act: calculator(2, 2)
# Observe: "Result: 4"
```

因为 **Observe 将错误信息写入 memory，下一轮 Think 时 LLM 能看到错误并调整策略**。

### 12.4 一句话总结

你可以这样记：

1. **ReAct 框架**：Think（LLM 推理）→ Act（执行工具）→ Observe（保存结果）→ 重复
2. **Step**：ReAct 循环中的结构化阶段（`ActionStep` 包含完整的 Think-Act-Observe）
3. **Memory**：连接各轮循环的桥梁（保存 Observation，供下一轮 Think 使用）
4. **Observation**：不是"做什么"，而是"记录什么"（保存 + 构建下一轮上下文）

如果把整个 Agent 看成一个有状态的推理机，那么：

1. `memory` 是历史（保存所有 Observation）
2. `step` 是历史中的单元（每个 ActionStep 是一轮完整的 ReAct）
3. `to_messages()` 是记忆的读取接口（将 Observation 转换为 LLM 的输入）

**核心洞察**：ReAct 框架通过 memory 系统实现了"反馈闭环"，让 LLM 能够像人类一样，基于经验（Observation）不断调整策略。

## 13. 自测问题

如果下面这些问题你都能顺畅回答，说明这部分已经基本读懂：

### 13.1 ReAct 框架相关

1. ReAct 的 Think-Act-Observe 三个阶段分别在代码的哪里体现？
2. 为什么 Observe 阶段只需要 `memory.append()`？
3. 为什么 Agent 能够基于错误信息自我纠错？
4. 下一轮 LLM 如何"看到"上一轮的执行结果？

### 13.2 Step 和 Memory 相关

1. 为什么这个项目不把历史只保存成纯文本？
2. 为什么 `ActionStep` 既是记录，也是下一轮上下文来源？
3. 为什么 `FinalAnswerStep` 要单独存在？
4. 为什么 planning 要进入 memory，而不是只放在局部变量？

### 13.3 核心理解验证

1. 如果去掉 `memory.append(action_step)`，Agent 会发生什么？
   - 答案：下一轮 LLM 看不到上一轮的结果，无法基于反馈调整策略
2. `ActionStep.to_messages()` 的作用是什么？
   - 答案：将 ActionStep（包含 Observation）转换为 LLM 的输入消息
3. 为什么说 Observe 是 ReAct 框架的"闭环"关键？
   - 答案：Observe 将执行结果保存到 memory，下一轮 Think 时通过 `to_messages()` 读取，形成反馈循环

## 14. 完整的 ReAct 循环示例

```python
# 任务：搜索巴黎和伦敦天气，比较温度

# === 第1轮 ReAct ===
# Think
messages = [{"role": "user", "content": "比较巴黎和伦敦的温度"}]
llm_output = model.generate(messages)
# "先搜索巴黎天气"

# Act
result = web_search("Paris weather")  # "Paris: 20°C"

# Observe
memory.append(ActionStep(observations="Paris: 20°C"))

# === 第2轮 ReAct ===
# Think - 构建上下文
messages = memory.to_messages()
# messages = [
#     {"role": "user", "content": "比较巴黎和伦敦的温度"},
#     {"role": "assistant", "content": "先搜索巴黎天气"},
#     {"role": "tool_response", "content": "Observation: Paris: 20°C"}  # 👈
# ]
llm_output = model.generate(messages)
# "巴黎是20°C，现在搜索伦敦"

# Act
result = web_search("London weather")  # "London: 15°C"

# Observe
memory.append(ActionStep(observations="London: 15°C"))

# === 第3轮 ReAct ===
# Think - 构建上下文
messages = memory.to_messages()
# messages = [
#     {"role": "user", "content": "比较巴黎和伦敦的温度"},
#     {"role": "assistant", "content": "先搜索巴黎天气"},
#     {"role": "tool_response", "content": "Observation: Paris: 20°C"},
#     {"role": "assistant", "content": "现在搜索伦敦"},
#     {"role": "tool_response", "content": "Observation: London: 15°C"}  # 👈
# ]
llm_output = model.generate(messages)
# "巴黎20°C，伦敦15°C，温度差是5°C"

# Act
final_answer("温度差是5°C")

# Done!
```

这个例子完整展示了：
1. 每轮的 Observe 如何保存到 memory
2. 下一轮的 Think 如何通过 `to_messages()` 读取历史
3. LLM 如何基于累积的 Observation 逐步完成任务

这就是 ReAct 框架在 smolagents 中的完整实现！


## 15. 深入理解：LLM 调用和工具执行的关系

### 15.1 LLM 只在 Think 阶段被调用

这是一个关键理解：**LLM 只在每个 ReAct 步骤的开始被调用一次**。

```python
def _step_stream(self, memory_step):
    # === 阶段1：Think - 需要 LLM ✅ ===
    messages = self.write_memory_to_messages()
    
    # 👇 这里是 LLM 调用！
    output_stream = self.model.generate_stream(
        messages,
        stop_sequences=["```", "Observation:"]
    )
    
    for token in output_stream:
        yield token  # 流式输出 LLM 的思考过程
    
    # === 阶段2：Parse - 不需要 LLM ❌ ===
    code = parse_code_blobs(output_text)
    
    # === 阶段3：Act - 不需要 LLM ❌ ===
    result = self.python_executor(code)
    # 工具调用是普通的函数调用，不需要 LLM
    
    yield ActionOutput(result)
```

### 15.2 工具调用不需要 LLM

```python
# LLM 的输出（CodeAgent）：
"""
Let me search for weather:
```python
result = web_search("Paris weather")
final_answer(result)
```
"""

# 解析代码（不需要 LLM）
code = "result = web_search('Paris weather')\nfinal_answer(result)"

# 执行代码（不需要 LLM）
# web_search 是提前注入到执行环境的函数
result = execute(code)
# 内部会调用：web_search("Paris weather") → "Paris: 20°C"
```

### 15.3 为什么 CodeAgent 更高效？

**ToolCallingAgent**：
```python
# 第1次 LLM 调用
llm_output = model.generate(messages)
# 输出：{"tool": "web_search", "args": {"query": "Paris weather"}}

# 执行工具
result1 = web_search("Paris weather")

# 第2次 LLM 调用（如果需要更多工具）
llm_output = model.generate(messages + [result1])
# 输出：{"tool": "web_search", "args": {"query": "London weather"}}

# 执行工具
result2 = web_search("London weather")

# 总共：2次 LLM 调用
```

**CodeAgent**：
```python
# 只需1次 LLM 调用
llm_output = model.generate(messages)
# 输出：
"""
```python
cities = ["Paris", "London"]
results = []
for city in cities:
    results.append(web_search(f"{city} weather"))
final_answer(results)
```
"""

# 执行代码（包含多次工具调用）
result = execute(code)
# 内部会调用：
#   web_search("Paris weather")
#   web_search("London weather")

# 总共：1次 LLM 调用！
```

这就是为什么 CodeAgent 能减少 30% 的 LLM 调用次数。

## 16. 深入理解：两层 Stream 的区别

### 16.1 两个 Stream 的层次

```
用户界面
  ↑ yield 各种事件
_step_stream()  ← Agent 层的流式处理
  ↑ yield ChatMessageStreamDelta (透传)
  ↑ yield ToolCall/ActionOutput (Agent 生成)
model.generate_stream()  ← LLM 层的流式处理
  ↑ yield ChatMessageStreamDelta (LLM 生成)
```

### 16.2 model.generate_stream() - LLM 层

**作用**：流式返回 LLM 生成的 token

```python
def generate_stream(self, messages):
    # 调用 OpenAI/Anthropic/HF API
    for chunk in api_call_stream(messages):
        yield ChatMessageStreamDelta(content=chunk)
        # 单个 token 或几个字符
```

**返回的是**：LLM 生成的"文本流"（只有一种类型）

### 16.3 _step_stream() - Agent 层

**作用**：流式返回 Agent 执行过程中的各种事件

```python
def _step_stream(self, memory_step):
    # 1. 透传 LLM 的 token
    for token in self.model.generate_stream(messages):
        yield token  # ChatMessageStreamDelta
    
    # 2. Agent 层的事件
    yield ToolCall(...)      # 工具调用请求
    yield ToolOutput(...)    # 工具执行结果
    yield ActionOutput(...)  # 最终输出
```

**返回的是**：Agent 执行的"事件流"（多种类型的对象）

### 16.4 完整的流式传递链

```python
# 用户调用
for event in agent.run("计算 2+2", stream=True):
    print(type(event), event)

# 输出示例：
<ChatMessageStreamDelta> "Let"           # 来自 LLM
<ChatMessageStreamDelta> " me"           # 来自 LLM
<ChatMessageStreamDelta> " calculate"    # 来自 LLM
<ChatMessageStreamDelta> "..."           # 来自 LLM
<ToolCall> python_interpreter            # 来自 Agent
<ActionOutput> 4                         # 来自 Agent
<FinalAnswerStep> 4                      # 来自 Agent
```

### 16.5 为什么需要两层 Stream？

**LLM 层的 `generate_stream()`**：
- 实时显示 LLM 的"思考过程"
- 用户体验更好（打字效果）
- 可以提前中断

**Agent 层的 `_step_stream()`**：
- 实时显示 Agent 的"执行过程"
- 透明度：看到 Agent 在做什么
- 调试：发现哪一步出错
- 进度反馈：长任务不会让用户焦虑

## 17. 常见误解澄清

### 17.1 误解：Observe 需要"处理"观察结果

**错误理解**：
```python
# Observe 阶段需要分析结果、提取信息、做判断
def observe(result):
    if "error" in result:
        return handle_error(result)
    else:
        return extract_info(result)
```

**正确理解**：
```python
# Observe 只是保存，不做任何处理
def observe(result):
    memory.append(observations=result)  # 只是保存
    # 真正的"观察"和"理解"发生在下一轮 LLM 的 Think 中
```

### 17.2 误解：每次工具调用都需要 LLM

**错误理解**：
```python
# 每次调用工具前都要问 LLM
for tool in tools:
    should_call = llm.decide(tool)  # ❌
    if should_call:
        result = tool()
```

**正确理解**：
```python
# LLM 一次性决定所有工具调用
llm_output = llm.generate(messages)  # 只调用一次
code = parse(llm_output)
result = execute(code)  # 执行所有工具调用
```

### 17.3 误解：ActionStep 只是日志

**错误理解**：
```python
# ActionStep 只是用来记录，方便调试
action_step = ActionStep(...)
memory.append(action_step)  # 只是存档
```

**正确理解**：
```python
# ActionStep 是下一轮推理的输入来源
action_step = ActionStep(observations="Paris: 20°C")
memory.append(action_step)

# 下一轮
messages = memory.to_messages()  # 👈 读取 ActionStep
# messages 包含 observations，LLM 能看到
llm_output = model.generate(messages)
```

### 17.4 误解：Memory 只是存储

**错误理解**：
```python
# Memory 只是一个列表，用来存储历史
memory = []
memory.append(step)
```

**正确理解**：
```python
# Memory 是 ReAct 循环的连接器
memory.append(step)  # 保存观察

# 下一轮
messages = memory.to_messages()  # 转换为 LLM 输入
# Memory 实现了"反馈闭环"
```

## 18. 核心设计模式总结

### 18.1 模板方法模式

```python
# 父类定义框架
class MultiStepAgent:
    def _run_stream(self):
        while not done:
            # 框架代码
            for output in self._step_stream(action_step):
                yield output
            memory.append(action_step)
    
    @abstractmethod
    def _step_stream(self):
        # 子类实现
        pass

# 子类实现具体策略
class CodeAgent(MultiStepAgent):
    def _step_stream(self):
        # 生成并执行代码
        pass

class ToolCallingAgent(MultiStepAgent):
    def _step_stream(self):
        # 生成并执行工具调用
        pass
```

### 18.2 策略模式

不同的 Agent 类型 = 不同的执行策略：
- `CodeAgent`：Python 代码执行策略
- `ToolCallingAgent`：JSON 工具调用策略

### 18.3 观察者模式

```python
# 回调系统监听 Agent 执行步骤
self.step_callbacks.register(ActionStep, callback)

# 每个步骤完成后触发回调
self._finalize_step(action_step)
# 内部会调用：self.step_callbacks.callback(action_step)
```

### 18.4 装饰器模式

```python
# AgentType 为原始数据类型添加额外功能
class AgentImage(AgentType):
    def __init__(self, raw_image):
        self._raw = raw_image  # 原始 PIL.Image
    
    def to_string(self):
        # 添加序列化能力
        pass
    
    def _ipython_display_(self):
        # 添加 Jupyter 显示能力
        pass
```

## 19. 最后的总结：一张图理解全部

```
┌─────────────────────────────────────────────────────────┐
│                    ReAct 循环                            │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │ Think（思考）                                   │    │
│  │ - 位置：_step_stream() 开始                     │    │
│  │ - 调用：model.generate_stream(messages)        │    │
│  │ - 输入：memory.to_messages() 👈 读取历史观察    │    │
│  │ - 输出：LLM 生成的代码或工具调用                │    │
│  └────────────────────────────────────────────────┘    │
│                      ↓                                   │
│  ┌────────────────────────────────────────────────┐    │
│  │ Act（行动）                                     │    │
│  │ - 位置：_step_stream() 中间                     │    │
│  │ - 执行：execute(code) 或 tool(**args)          │    │
│  │ - 不需要 LLM，只是普通函数调用                  │    │
│  └────────────────────────────────────────────────┘    │
│                      ↓                                   │
│  ┌────────────────────────────────────────────────┐    │
│  │ Observe（观察）                                 │    │
│  │ - 位置：_run_stream() 结尾                      │    │
│  │ - 保存：memory.append(action_step)             │    │
│  │ - action_step.observations = result            │    │
│  │ - 不做任何处理，只是保存                        │    │
│  └────────────────────────────────────────────────┘    │
│                      ↓                                   │
│              下一轮循环（带着新观察）                     │
│                      ↑                                   │
│                      └───────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

关键点：
1. Think 时 LLM 能看到之前的 Observation（通过 memory.to_messages()）
2. Act 不需要 LLM，只是执行工具
3. Observe 只是保存，真正的"理解"在下一轮 Think
4. Memory 是连接各轮的桥梁，实现反馈闭环
```

这就是 smolagents 的完整执行机制！
