# smolagents 核心数据格式指南

> 理解这些格式是读懂 Agent 代码的基础。Agent 运行的每一步都在这些格式之间转换。

## 完整链路概览

```
定义工具(Tool.inputs) → 描述给LLM(JSON Schema) → LLM决定调用(Tool Call)
    → 消息传递(ChatMessage) → 代码执行(Code Response) → 保存分享(Serialization)
```

---

## 1. 消息格式（ChatMessage）

**位置**: `src/smolagents/models.py`
**作用**: Agent 和 Model 之间通信的基本单位，整个对话历史就是一个 ChatMessage 列表。

```python
# 用户消息
{"role": "user", "content": "帮我搜索 Python 教程"}

# 助手消息（纯文本回复）
{"role": "assistant", "content": "好的，我来帮你搜索。"}

# 助手消息（调用工具）
{
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": {"query": "Python tutorial"}
            }
        }
    ]
}

# 工具返回结果
{"role": "tool-response", "content": "搜索结果：..."}
```

**角色类型（MessageRole）**:
- `user` — 用户输入
- `assistant` — 模型输出（可能包含 tool_calls）
- `system` — 系统提示词
- `tool-call` — 工具调用记录
- `tool-response` — 工具执行结果

---

## 2. 工具描述格式（JSON Schema）

**位置**: `get_json_schema()` 生成，传给 LLM API
**作用**: 告诉 LLM "你有哪些工具可以用、每个工具接受什么参数"
**规范来源**: OpenAI function calling 规范

```python
{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "搜索网页内容。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                }
            },
            "required": ["query"]
        },
        # 以下是 smolagents 的扩展字段（OpenAI 原始规范没有）
        "return": {
            "type": "string"
        }
    }
}
```

**为什么外面套一层 "function"**:
OpenAI 的设计里工具不只有函数一种类型，所以用 `"type": "function"` 做区分。
smolagents 内部用时取 `["function"]` 拿里面的内容。

---

## 3. 工具调用格式（Tool Call）

**位置**: LLM 返回的 `ChatMessage.tool_calls` 中
**作用**: LLM 决定调用工具时的输出指令，Agent 解析后去执行

```python
{
    "id": "call_abc123",           # 唯一标识，用于匹配调用和返回
    "type": "function",
    "function": {
        "name": "web_search",      # 要调用哪个工具
        "arguments": {              # 传什么参数
            "query": "Python tutorial"
        }
    }
}
```

**流转过程**:
1. LLM 返回 tool_call → Agent 解析出 name 和 arguments
2. Agent 找到对应的 Tool，调用 `tool(**arguments)`
3. 把结果包装成 `tool-response` 消息，追加到对话历史
4. 把更新后的对话历史再发给 LLM，继续推理

---

## 4. 工具输入定义格式（Tool.inputs）

**位置**: `Tool` 类的 `inputs` 属性
**作用**: 定义工具接受什么参数，会被转换成 JSON Schema 传给 LLM

```python
{
    "query": {
        "type": "string",                # 参数类型（必填）
        "description": "搜索关键词",      # 参数描述（必填，LLM 靠这个理解参数含义）
        "nullable": True                  # 是否可为 None（可选）
    },
    "top_k": {
        "type": "integer",
        "description": "返回结果数量",
        "nullable": True
    }
}
```

**允许的类型（AUTHORIZED_TYPES）**:
`string`, `boolean`, `integer`, `number`, `image`, `audio`, `array`, `object`, `any`, `null`

**与 forward() 的关系**:
inputs 的 key 必须和 forward() 的参数名完全一致：
```python
inputs = {"query": {...}, "top_k": {...}}
def forward(self, query: str, top_k: int):  # 参数名必须匹配
```

---

## 5. CodeAgent 代码响应格式

**位置**: CodeAgent 从 LLM 输出中提取
**作用**: CodeAgent 不用 JSON 调用工具，而是让 LLM 生成 Python 代码

LLM 看到的工具描述（由 `to_code_prompt()` 生成）：
```python
def web_search(query: string) -> string:
    """搜索网页内容。
    Args:
        query: 搜索关键词
    """
```

LLM 生成的代码：
```python
result = web_search(query="Python tutorial")
print(result)
```

框架提取代码块后用 exec 执行。

**与 ToolCallingAgent 的对比**:
| | CodeAgent | ToolCallingAgent |
|---|---|---|
| 工具描述 | Python 函数签名（`to_code_prompt()`） | 纯文本（`to_tool_calling_prompt()`） |
| 调用方式 | 生成 Python 代码 | 生成 JSON tool_call |
| 执行方式 | exec 执行代码 | 解析 JSON 后调用 |

---

## 6. 序列化格式（to_dict / from_dict）

**位置**: `Tool.to_dict()` / `Tool.from_dict()`
**作用**: 保存、分享、还原工具

```python
{
    "name": "web_search",
    "code": "from smolagents import Tool\n\nclass SimpleTool(Tool):\n    ...",  # 完整类源码
    "requirements": ["requests", "smolagents"],   # 第三方依赖
    "output_schema": {...}                         # 可选，结构化输出的 JSON Schema
}
```

**序列化路径**:
- `@tool` 装饰器创建的 → 从原函数反向拼出完整类代码
- 继承 `Tool` 创建的 → 直接提取类源码

**还原路径**:
`from_dict()` → `from_code()` → `exec()` 执行代码字符串 → 找到 Tool 子类 → 实例化

---

## 格式之间的转换关系

```
用户定义 Tool
    │
    ├── Tool.inputs (格式4)
    │       │
    │       ├── to_code_prompt() ──→ Python 函数签名 ──→ CodeAgent 用 (格式5)
    │       │
    │       └── get_tool_json_schema() ──→ JSON Schema (格式2) ──→ API 传给 LLM
    │
    ├── LLM 返回 ──→ Tool Call (格式3) 或 代码块 (格式5)
    │       │
    │       └── Agent 执行工具 ──→ 结果包装成 ChatMessage (格式1)
    │
    └── to_dict() ──→ 序列化字典 (格式6) ──→ save() / push_to_hub()
```
