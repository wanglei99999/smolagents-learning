# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# models.py —— 模型适配层（LLM 接入接口）
#
# smolagents 通过统一的 Model 抽象基类对接不同的 LLM 后端，
# 所有模型实现类都继承自 Model，并实现 generate() 方法。
#
# 内置模型类（每种对应一种接入方式）：
#
#   InferenceClientModel  ←→  HuggingFace Inference API（推荐入门）
#       - 直接调用 HF Hub 上托管的模型
#       - 需要 HF token（免费或 PRO）
#       - 示例：InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct")
#
#   LiteLLMModel  ←→  多后端（OpenAI、Anthropic、Ollama、Together AI 等）
#       - 底层依赖 litellm 库，支持 100+ 个 LLM 提供商
#       - 示例：LiteLLMModel(model_id="gpt-4o")
#               LiteLLMModel(model_id="ollama_chat/llama3")
#
#   LiteLLMRouterModel  ←→  多模型负载均衡
#       - 在多个 LLM 之间按策略（轮询、最少繁忙等）路由请求
#       - 适合高并发或多模型 A/B 测试场景
#
#   OpenAIServerModel  ←→  兼容 OpenAI API 协议的模型服务
#       - 可对接 vLLM、Ollama 等本地服务或第三方 API
#       - 示例：OpenAIServerModel(model_id="...", api_base="http://localhost:8000/v1")
#
#   TransformersModel  ←→  本地 HuggingFace transformers 模型
#       - 在本地 GPU/CPU 上运行模型，无需网络
#       - 推理速度取决于本地硬件
#
#   AzureOpenAIServerModel  ←→  Azure OpenAI 服务
#
#   VLLMModel  ←→  vLLM 高性能推理服务
#
# 核心数据结构（消息流转）：
#   ChatMessage → Model.generate() → ChatMessage（含 tool_calls / content）
#   MessageRole：USER / ASSISTANT / SYSTEM / TOOL_CALL / TOOL_RESPONSE
#
# 重试机制：
#   默认对 API 调用失败进行最多 3 次重试，等待 60 秒，指数退避
# =============================================================================

import json
import logging
import os
import re
import uuid
import warnings
from collections.abc import Generator
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from threading import Thread
from typing import TYPE_CHECKING, Any

from .monitoring import TokenUsage
from .tools import Tool
from .utils import RateLimiter, Retrying, _is_package_available, encode_image_base64, make_image_url, parse_json_blob


if TYPE_CHECKING:
    from transformers import StoppingCriteriaList


logger = logging.getLogger(__name__)

# API 重试配置：网络抖动或限流时自动重试
RETRY_WAIT = 60              # 首次重试前等待 60 秒
RETRY_MAX_ATTEMPTS = 3       # 最多重试 3 次
RETRY_EXPONENTIAL_BASE = 2   # 指数退避基数（下次等待 = 等待时间 * 2）
RETRY_JITTER = True          # 添加随机抖动，避免多实例同时重试导致雪崩

# 支持结构化生成（JSON Schema 约束输出）的推理提供商列表
STRUCTURED_GENERATION_PROVIDERS = ["cerebras", "fireworks-ai"]

# CodeAgent 使用结构化输出模式时的 JSON Schema 约束
# LLM 必须严格输出 {"thought": "...", "code": "..."} 格式
CODEAGENT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "schema": {
            "additionalProperties": False,
            "properties": {
                "thought": {
                    "description": "A free form text description of the thought process.",
                    "title": "Thought",
                    "type": "string",
                },
                "code": {
                    "description": "Valid Python code snippet implementing the thought.",
                    "title": "Code",
                    "type": "string",
                },
            },
            "required": ["thought", "code"],
            "title": "ThoughtAndCodeAnswer",
            "type": "object",
        },
        "name": "ThoughtAndCodeAnswer",
        "strict": True,
    },
}


def get_dict_from_nested_dataclasses(obj, ignore_key=None):
    """把嵌套的 dataclass 对象递归转成普通 dict。

    ChatMessage 里嵌套了 ChatMessageToolCall，里面又嵌套了 ChatMessageToolCallFunction。
    这个函数会递归地把它们全部转成 dict。

    ignore_key 用于跳过某个字段，比如 ignore_key="raw" 可以在序列化时
    跳过原始 API 响应（太大，不需要存）。

    被 ChatMessage.dict() 和 ChatMessage.model_dump_json() 调用。
    """
    def convert(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items() if k != ignore_key}
        return obj

    return convert(obj)


def remove_content_after_stop_sequences(content: str | None, stop_sequences: list[str] | None) -> str | None:
    """在停止词处截断文本。

    有些模型不支持 stop 参数（supports_stop_parameter 返回 False），
    生成的文本可能包含停止词之后的多余内容。这个函数做事后截断：
      remove_content_after_stop_sequences("答案是42。END多余内容", ["END"])
      → "答案是42。"

    每个模型的 generate() 末尾都会调用：
      if stop_sequences is not None and not self.supports_stop_parameter:
          content = remove_content_after_stop_sequences(content, stop_sequences)
    """
    if content is None or not stop_sequences:
        return content

    for stop_seq in stop_sequences:
        split = content.split(stop_seq)
        content = split[0]
    return content


# -----------------------------------------------------------------------------
# 消息数据结构：Agent ↔ LLM 之间的消息格式定义
# -----------------------------------------------------------------------------

@dataclass
class ChatMessageToolCallFunction:
    """工具调用中的函数部分（函数名 + 参数）。"""
    arguments: Any   # dict 或 JSON 字符串（取决于模型返回格式）
    name: str        # 工具名称（必须与注册的工具名一致）
    description: str | None = None


@dataclass
class ChatMessageToolCall:
    """LLM 发出的单次工具调用请求（一次 LLM 输出可能包含多个）。
    id：工具调用 ID，在多个并行工具调用时用于关联请求和结果
    """
    '''
        这样写的原因是，openAi Api的工具调用JSON长这样：
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"city\": \"Beijing\"}"
            }
        }

    '''
    function: ChatMessageToolCallFunction
    id: str
    type: str  # 固定为 "function"

    def __str__(self) -> str:
        return f"Call: {self.id}: Calling {str(self.function.name)} with arguments: {str(self.function.arguments)}"


class MessageRole(str, Enum):
    """消息角色枚举。
    TOOL_CALL：LLM 发出工具调用请求（对应 ToolCallingAgent 的 assistant 消息中含 tool_calls）
    TOOL_RESPONSE：工具调用结果（框架将执行结果写回历史的角色）
    """
    USER = "user"                    #用户输入
    ASSISTANT = "assistant"          #LLM回复内容
    SYSTEM = "system"                #系统提示词
    TOOL_CALL = "tool-call"          #LLM请求调用工具
    TOOL_RESPONSE = "tool-response"  #工具调用返回

    #classmethod 类方法标记，cls表示类本身
    @classmethod
    def roles(cls):
        return [r.value for r in cls]


@dataclass
class ChatMessage:
    """Agent 与 LLM 之间传递的核心消息体。
    整个框架的"血液"——Agent 把用户输入包装成 ChatMessage 列表发给 Model，
    Model 返回一个 ChatMessage（可能带 tool_calls），Agent 再决定下一步。

    生命周期：
        创建：API 返回 dict → from_dict() → ChatMessage，或直接构造
        使用：Agent 读取 .role, .content, .tool_calls 决定下一步动作
        序列化：.dict() → Python dict，.model_dump_json() → JSON 字符串
        展示：.render_as_markdown() → 人类可读文本
    """
    role: MessageRole                                       # 谁发的（user/assistant/system/tool-call/tool-response）
    content: str | list[dict[str, Any]] | None = None       # 消息内容：
                                                            #   str → 纯文本，如 "今天天气怎么样"
                                                            #   list[dict] → 多模态内容（文字+图片混合）
                                                            #   None → 没有文本，比如 LLM 只返回了工具调用
    tool_calls: list[ChatMessageToolCall] | None = None     # LLM 请求调用的工具列表（可一次调多个）
    raw: Any | None = None                                  # API 原始返回对象，调试用，序列化时会被忽略
    token_usage: TokenUsage | None = None                   # 本次调用消耗的 token 数（输入+输出），用于监控成本

    def __post_init__(self) -> None:
        """dataclass 的 __init__ 执行完后自动调用。
        作用：将 tool_calls 中各种格式的工具调用统一转换为 ChatMessageToolCall 对象。
        因为 tool_calls 可能来自不同 API（OpenAI 返回 pydantic 对象、dict、或已经是目标类型），
        通过 _coerce_tool_call 适配器统一格式。
        """
        if self.tool_calls is None:
            return
        self.tool_calls = [_coerce_tool_call(tool_call) for tool_call in self.tool_calls]

    def model_dump_json(self):
        """序列化为 JSON 字符串，忽略 raw 字段。
        raw 是 API 原始返回，可能包含不可序列化的对象且体积大，所以跳过。
        """
        return json.dumps(get_dict_from_nested_dataclasses(self, ignore_key="raw"))

    @classmethod
    def from_dict(cls, data: dict, raw: Any | None = None, token_usage: TokenUsage | None = None) -> "ChatMessage":
        """从字典创建 ChatMessage（类方法，用 ChatMessage.from_dict({...}) 调用）。
        使用场景：从 JSON 反序列化、从历史记录加载消息时。

        处理逻辑：
        1. 如果 dict 中有 tool_calls，先把每个 tool call 从 dict 转成 ChatMessageToolCall 对象
           （**tc["function"] 是字典解包，等价于 name=..., arguments=...）
        2. 用 cls(...) 构造 ChatMessage 实例（cls 就是 ChatMessage 本身）
        """
        if data.get("tool_calls"):
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(**tc["function"]), id=tc["id"], type=tc["type"]
                )
                for tc in data["tool_calls"]
            ]
            data["tool_calls"] = tool_calls
        return cls(
            role=MessageRole(data["role"]),
            content=data.get("content"),
            tool_calls=data.get("tool_calls"),
            raw=raw,
            token_usage=token_usage,
        )

    def dict(self):
        """转换为 Python dict（递归处理嵌套的 dataclass）。"""
        return get_dict_from_nested_dataclasses(self)

    def render_as_markdown(self) -> str:
        """渲染为人类可读的文本，用于日志和调试界面。
        如果有工具调用，会把工具名和参数以 JSON 格式追加到内容后面。
        """
        rendered = str(self.content) or ""
        if self.tool_calls:
            rendered += "\n".join(
                [
                    json.dumps({"tool": tool.function.name, "arguments": tool.function.arguments})
                    for tool in self.tool_calls
                ]
            )
        return rendered


def _coerce_tool_call(tool_call: Any) -> ChatMessageToolCall:
    """工具调用格式适配器：不管传入什么格式，都统一转成 ChatMessageToolCall。

    为什么需要这个适配器？
    因为 smolagents 支持多种 LLM 后端，每个后端返回的 tool_call 格式不同：

    - OpenAI SDK 新版本（pydantic v2）返回的 tool_call 对象：
        有 .model_dump() 方法，如：
        ChoiceDeltaToolCall(id="call_abc", type="function",
            function=Function(name="search", arguments='{"q":"天气"}'))

    - OpenAI SDK 老版本（pydantic v1）返回的 tool_call 对象：
        有 .dict() 方法，如：
        ChatCompletionMessageToolCall(id="call_xyz", type="function", function=...)

    - HuggingFace / 其他后端返回的可能就是普通 dict：
        {"id": "call_123", "type": "function",
         "function": {"name": "search", "arguments": {"q": "天气"}}}

    - 或者已经被转换过了，就是 ChatMessageToolCall 本身

    适配逻辑：不管你给我什么 → 先转成 dict → 再从 dict 构造出统一的 ChatMessageToolCall。
    这样 Agent 后续处理 tool_calls 时，永远只需要面对一种类型。
    """
    if isinstance(tool_call, ChatMessageToolCall):
        return tool_call

    # 将各种格式统一转成 dict
    if isinstance(tool_call, dict):
        tool_call_dict = tool_call
    elif hasattr(tool_call, "model_dump"):
        tool_call_dict = tool_call.model_dump()       # pydantic v2 对象（如 OpenAI SDK 新版）
    elif hasattr(tool_call, "dict") and callable(tool_call.dict):
        tool_call_dict = tool_call.dict()             # pydantic v1 对象（如 OpenAI SDK 老版）

    # 从 dict 构造统一的 ChatMessageToolCall
    return ChatMessageToolCall(
        function=ChatMessageToolCallFunction(
            arguments=tool_call_dict["function"]["arguments"],
            name=tool_call_dict["function"]["name"],
        ),
        id=tool_call_dict["id"],
        type=tool_call_dict["type"],
    )


def parse_json_if_needed(arguments: str | dict) -> str | dict:
    """如果 arguments 是 JSON 字符串就解析成 dict，已经是 dict 就直接返回。
    有些 API 返回的 tool call arguments 是字符串形式的 JSON（如 '{"city": "北京"}' ），
    有些直接返回 dict，这个函数统一处理两种情况。
    """
    if isinstance(arguments, dict):
        return arguments
    else:
        try:
            return json.loads(arguments)
        except Exception:
            return arguments


@dataclass
class ChatMessageToolCallStreamDelta:
    """流式输出时，工具调用的一个碎片（delta）。

    LLM 流式返回工具调用时，信息是一点一点到达的：
    - 碎片1：id="call_abc", name="web_search", arguments=""     （告诉你要调什么工具）
    - 碎片2：id=None, name="", arguments='{"q":'                （参数的前半部分）
    - 碎片3：id=None, name="", arguments=' "天气"}'              （参数的后半部分）

    index 字段标记这是第几个工具调用（LLM 可能同时调多个工具，碎片交错到达时需要区分）。
    后续碎片中 id/type/name 可能为 None，表示"跟前面同一个调用，只是补充 arguments"。
    """

    index: int | None = None       # 第几个工具调用（用于区分并行的多个调用）
    id: str | None = None          # 调用 ID（通常只在第一个碎片中出现）
    type: str | None = None        # 固定 "function"（通常只在第一个碎片中出现）
    function: ChatMessageToolCallFunction | None = None  # 函数名和参数的增量


@dataclass
class ChatMessageStreamDelta:
    """流式输出时，LLM 返回的一个碎片（delta）。

    流式模式下，LLM 每生成几个 token 就返回一个 delta，用户看到的是文字逐渐蹦出来的效果。
    所有 delta 最终通过 agglomerate_stream_deltas() 拼成一个完整的 ChatMessage。

    与 ChatMessage 的关系：
        非流式：Model.generate()        → 直接返回 ChatMessage（完整的）
        流式：  Model.generate_stream() → 逐个返回 ChatMessageStreamDelta（碎片）
                                            → agglomerate_stream_deltas() 拼成 ChatMessage
    """
    content: str | None = None                                      # 文本内容的增量（几个字）
    tool_calls: list[ChatMessageToolCallStreamDelta] | None = None  # 工具调用的增量
    token_usage: TokenUsage | None = None                           # token 使用量（通常在最后一个碎片中）


def agglomerate_stream_deltas(
    stream_deltas: list[ChatMessageStreamDelta], role: MessageRole = MessageRole.ASSISTANT
) -> ChatMessage:
    """将所有流式碎片（delta）拼接成一个完整的 ChatMessage。

    流式输出结束后调用此函数，把收集到的所有碎片合并：
    - content：所有碎片的 content 拼接成完整文本
    - tool_calls：按 index 分组，把同一个工具调用的碎片合并（name 取第一个非空的，arguments 逐段拼接）
    - token_usage：所有碎片的 token 数累加

    示例：
        delta1: content="北京"
        delta2: content="今天晴"
        delta3: content="，25°C"
        → ChatMessage(content="北京今天晴，25°C")
    """
    accumulated_tool_calls: dict[int, ChatMessageToolCallStreamDelta] = {}
    accumulated_content = ""
    total_input_tokens = 0
    total_output_tokens = 0
    for stream_delta in stream_deltas:
        # 累加 token 使用量
        if stream_delta.token_usage:
            total_input_tokens += stream_delta.token_usage.input_tokens
            total_output_tokens += stream_delta.token_usage.output_tokens
        # 拼接文本内容
        if stream_delta.content:
            accumulated_content += stream_delta.content
        # 合并工具调用碎片（按 index 分组）
        if stream_delta.tool_calls:
            for tool_call_delta in stream_delta.tool_calls:  # 通常每个碎片只有一个 tool_call_delta
                if tool_call_delta.index is not None:
                    # 如果是新的 index，创建一个空的工具调用容器
                    if tool_call_delta.index not in accumulated_tool_calls:
                        accumulated_tool_calls[tool_call_delta.index] = ChatMessageToolCallStreamDelta(
                            id=tool_call_delta.id,
                            type=tool_call_delta.type,
                            function=ChatMessageToolCallFunction(name="", arguments=""),
                        )
                    # 把碎片中的信息合并到对应 index 的工具调用上
                    tool_call = accumulated_tool_calls[tool_call_delta.index]
                    if tool_call_delta.id:
                        tool_call.id = tool_call_delta.id
                    if tool_call_delta.type:
                        tool_call.type = tool_call_delta.type
                    if tool_call_delta.function:
                        # name 只取第一个非空的（后续碎片的 name 通常为空）
                        if tool_call_delta.function.name and len(tool_call_delta.function.name) > 0:
                            tool_call.function.name = tool_call_delta.function.name
                        # arguments 逐段拼接（如 '{"q":' + ' "天气"}' → '{"q": "天气"}'）
                        if tool_call_delta.function.arguments:
                            tool_call.function.arguments += tool_call_delta.function.arguments
                else:
                    raise ValueError(f"Tool call index is not provided in tool delta: {tool_call_delta}")

    return ChatMessage(
        role=role,
        content=accumulated_content,
        tool_calls=[
            ChatMessageToolCall(
                function=ChatMessageToolCallFunction(
                    name=tool_call_stream_delta.function.name,
                    arguments=tool_call_stream_delta.function.arguments,
                ),
                id=tool_call_stream_delta.id or "",
                type="function",
            )
            for tool_call_stream_delta in accumulated_tool_calls.values()
            if tool_call_stream_delta.function
        ],
        token_usage=TokenUsage(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        ),
    )


# 角色转换映射：将 smolagents 内部角色名转换为标准 OpenAI 角色名
# 原因：大多数 LLM API 只支持 user/assistant/system 三种角色
# TOOL_CALL → ASSISTANT（工具调用是 LLM 发出的，属于 assistant 的行为）
# TOOL_RESPONSE → USER（工具结果从"外部"传回，属于 user 角色）
tool_role_conversions = {
    MessageRole.TOOL_CALL: MessageRole.ASSISTANT,
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


def get_tool_json_schema(tool: Tool) -> dict:
    """将 smolagents 的 Tool 对象转换为 OpenAI 格式的 function calling JSON Schema。

    LLM 需要知道"有哪些工具可以调用、每个工具接受什么参数"，这个函数就是做这个转换的。

    转换示例：
        Tool(name="web_search", description="搜索网页",
             inputs={"query": {"type": "string"}, "max_results": {"type": "integer", "nullable": True}})
        →
        {"type": "function", "function": {"name": "web_search", "description": "搜索网页",
         "parameters": {"type": "object", "properties": {...}, "required": ["query"]}}}

    类型兼容处理：
        - "any" → "string"（JSON Schema 没有 any 类型）
        - "anyOf" 联合类型 → 展开成具体类型列表
        - nullable 的参数不会出现在 required 列表中
    """
    properties = deepcopy(tool.inputs)
    required = []
    for key, value in properties.items():
        # JSON Schema 不支持 "any" 类型，统一转成 "string"
        if value["type"] == "any":
            value["type"] = "string"
        # 非 nullable 的参数加入 required 列表
        if not ("nullable" in value and value["nullable"]):
            required.append(key)

        # 处理 anyOf 联合类型（如：参数可以是 string 或 integer 或 null）
        if "anyOf" in value:
            types = []
            enum = None
            for t in value["anyOf"]:
                if t["type"] == "null":
                    value["nullable"] = True
                    continue
                if t["type"] == "any":
                    types.append("string")
                else:
                    types.append(t["type"])
                if "enum" in t:  # 假设 anyOf 中最多只有一个 enum
                    enum = t["enum"]

            value["type"] = types if len(types) > 1 else types[0]
            if enum is not None:
                value["enum"] = enum

            value.pop("anyOf")

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def get_clean_message_list(
    message_list: list[ChatMessage | dict],
    role_conversions: dict[MessageRole, MessageRole] | dict[str, str] = {},
    convert_images_to_image_urls: bool = False,
    flatten_messages_as_text: bool = False,
) -> list[dict[str, Any]]:
    """将 Agent 内部的 ChatMessage 列表转换为 LLM API 可接受的 dict 格式。
    这是发给 LLM 之前的最后一道处理，做四件事：

    1. 统一格式：输入可能混着 ChatMessage 对象和 dict，全部统一成 dict
    2. 角色转换：用 role_conversions 把 TOOL_CALL → ASSISTANT，TOOL_RESPONSE → USER
       （因为大多数 LLM API 只认 user/assistant/system 三种角色）
    3. 图片编码：把 PIL Image 对象转成 base64 字符串或 image_url 格式
    4. 合并相邻同角色消息：某些模型（如 Claude）要求消息必须 user/assistant 交替出现

    合并示例（角色转换后）：
        [assistant: "我来搜索", assistant: "Calling tools: ..."]  → [assistant: "我来搜索\nCalling tools: ..."]
        [user: "Observation: 25°C", user: "请继续"]              → [user: "Observation: 25°C\n请继续"]

    Args:
        message_list: 消息列表，可以混合 ChatMessage 对象和 dict
        role_conversions: 角色转换映射，如 {TOOL_CALL: ASSISTANT, TOOL_RESPONSE: USER}
        convert_images_to_image_urls: 是否把图片转成 URL 格式（API 模型用 True，本地模型用 False）
        flatten_messages_as_text: 是否把 content 压成纯文本字符串
            - False（默认）：content 保持 list[dict] 格式，适合支持多模态的 API
            - True：content 压成纯文本，适合只支持文本的模型（如某些本地模型）
    """
    output_message_list: list[dict[str, Any]] = []
    message_list = deepcopy(message_list)  # 深拷贝，避免修改原始列表
    for message in message_list:
        # === 第1步：统一格式 ===
        if isinstance(message, dict):
            message = ChatMessage.from_dict(message)
        role = message.role
        if role not in MessageRole.roles():
            raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported for now.")

        # === 第2步：角色转换 ===
        if role in role_conversions:
            message.role = role_conversions[role]  # type: ignore

        # === 第3步：图片编码 ===
        if isinstance(message.content, list):
            for element in message.content:
                assert isinstance(element, dict), "Error: this element should be a dict:" + str(element)
                if element["type"] == "image":
                    assert not flatten_messages_as_text, f"Cannot use images with {flatten_messages_as_text=}"
                    if convert_images_to_image_urls:
                        # API 模型：PIL Image → base64 → data:image/png;base64,xxx 格式的 URL
                        element.update(
                            {
                                "type": "image_url",
                                "image_url": {"url": make_image_url(encode_image_base64(element.pop("image")))},
                            }
                        )
                    else:
                        # 本地模型：PIL Image → base64 字符串
                        element["image"] = encode_image_base64(element["image"])

        # === 第4步：合并相邻同角色消息 ===
        if len(output_message_list) > 0 and message.role == output_message_list[-1]["role"]:
            # 当前消息的 role 跟上一条一样 → 合并到上一条里
            assert isinstance(message.content, list), "Error: wrong content:" + str(message.content)
            if flatten_messages_as_text:
                # 纯文本模式：直接拼接字符串
                output_message_list[-1]["content"] += "\n" + message.content[0]["text"]
            else:
                # 多模态模式：逐个元素追加，相邻的 text 元素合并
                for el in message.content:
                    if el["type"] == "text" and output_message_list[-1]["content"][-1]["type"] == "text":
                        # 两个相邻的 text 元素 → 合并成一个（避免冗余）
                        output_message_list[-1]["content"][-1]["text"] += "\n" + el["text"]
                    else:
                        # 不同类型（如 text 后面跟 image）→ 直接追加
                        output_message_list[-1]["content"].append(el)
        else:
            # 新角色 → 创建新消息
            if flatten_messages_as_text:
                content = message.content[0]["text"]
            else:
                content = message.content
            output_message_list.append(
                {
                    "role": message.role,
                    "content": content,
                }
            )
    return output_message_list


def get_tool_call_from_text(text: str, tool_name_key: str, tool_arguments_key: str) -> ChatMessageToolCall:
    """从 LLM 的纯文本输出中解析出工具调用。

    有些老模型不支持原生的 function calling，而是在文本中输出 JSON 格式的工具调用，如：
        '{"name": "web_search", "arguments": {"query": "天气"}}'
    这个函数从文本中提取 JSON，解析出工具名和参数，构造成 ChatMessageToolCall。

    Args:
        text: LLM 的原始文本输出
        tool_name_key: JSON 中工具名的 key（通常是 "name"）
        tool_arguments_key: JSON 中参数的 key（通常是 "arguments"）
    """
    tool_call_dictionary, _ = parse_json_blob(text)
    try:
        tool_name = tool_call_dictionary[tool_name_key]
    except Exception as e:
        raise ValueError(
            f"Tool call needs to have a key '{tool_name_key}'. Got keys: {list(tool_call_dictionary.keys())} instead"
        ) from e
    tool_arguments = tool_call_dictionary.get(tool_arguments_key, None)
    if isinstance(tool_arguments, str):
        tool_arguments = parse_json_if_needed(tool_arguments)
    return ChatMessageToolCall(
        id=str(uuid.uuid4()),  # 生成随机 UUID 作为调用 ID（因为文本中没有 id）
        type="function",
        function=ChatMessageToolCallFunction(name=tool_name, arguments=tool_arguments),
    )


def supports_stop_parameter(model_id: str) -> bool:
    """
    Check if the model supports the `stop` parameter.

    Not supported with reasoning models openai/o3, openai/o4-mini, and the openai/gpt-5 series (and their versioned variants).

    Args:
        model_id (`str`): Model identifier (e.g. "openai/o3", "o4-mini-2025-04-16")

    Returns:
        bool: True if the model supports the stop parameter, False otherwise
    """
    model_name = model_id.split("/")[-1]
    if model_name == "o3-mini":
        return True
    # o3* (except mini), o4*, all grok-* models, and the gpt-5* family (including versioned variants) don't support stop parameter
    openai_model_pattern = r"(o3(?:$|[-.].*)|o4(?:$|[-.].*)|gpt-5.*)"
    grok_model_pattern = r"([A-Za-z][A-Za-z0-9_-]*\.)?grok-[A-Za-z0-9][A-Za-z0-9_.-]*"
    pattern = rf"^({openai_model_pattern}|{grok_model_pattern})$"

    return not re.match(pattern, model_name)


class _ParameterRemove:
    """Sentinel value to indicate a parameter should be removed."""

    def __repr__(self):
        return "REMOVE_PARAMETER"


# 全局单例，所有地方都用这一个实例
REMOVE_PARAMETER = _ParameterRemove()


class Model:
    """所有模型实现的基类（抽象类）。

    定义了 Agent 与 LLM 交互的统一接口。所有模型子类都必须实现 generate() 方法。
    基类提供公共逻辑（参数组装、工具调用解析、序列化），子类只需关注"怎么调自己的 API"。

    继承关系：
        Model（本类）
        ├── VLLMModel          → vLLM 本地推理
        ├── MLXModel           → Apple Silicon 本地推理
        ├── TransformersModel  → HuggingFace transformers 本地推理
        └── ApiModel（中间层，增加限流+重试）
            ├── LiteLLMModel / LiteLLMRouterModel
            ├── InferenceClientModel
            ├── OpenAIModel / AzureOpenAIModel
            └── AmazonBedrockModel

    Parameters:
        flatten_messages_as_text: 是否把消息压成纯文本（本地文本模型用 True，API 模型用 False）
        tool_name_key: 从文本解析工具调用时，工具名的 key（默认 "name"）
        tool_arguments_key: 从文本解析工具调用时，参数的 key（默认 "arguments"）
        model_id: 模型标识符，如 "gpt-4" 或 "Qwen/Qwen2.5-72B-Instruct"
        **kwargs: 额外参数，存到 self.kwargs，会以最高优先级覆盖所有其他参数
    """

    def __init__(
        self,
        flatten_messages_as_text: bool = False,
        tool_name_key: str = "name",
        tool_arguments_key: str = "arguments",
        model_id: str | None = None,
        **kwargs,
    ):
        self.flatten_messages_as_text = flatten_messages_as_text
        self.tool_name_key = tool_name_key      # 给 get_tool_call_from_text 用的
        self.tool_arguments_key = tool_arguments_key
        self.kwargs = kwargs                     # 最高优先级参数，会覆盖一切
        self.model_id: str | None = model_id

    @property
    def supports_stop_parameter(self) -> bool:
        """判断当前模型是否支持 stop 参数（o3/o4/gpt-5/grok 系列不支持）。"""
        return supports_stop_parameter(self.model_id or "")

    def _prepare_completion_kwargs(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        convert_images_to_image_urls: bool = False,
        tool_choice: str | dict | None = "required",
        **kwargs,
    ) -> dict[str, Any]:
        """组装调用 LLM API 所需的全部参数。所有子类在调 API 之前都先调这个方法。

        参数优先级（从低到高）：
        1. 具体参数（stop_sequences, response_format, tools 等）
        2. kwargs（调用 generate 时传的额外参数）
        3. self.kwargs（构造模型时传的参数，最高优先级，可"锁定"某些参数）

        如果 self.kwargs 中某个值是 REMOVE_PARAMETER，该参数会被从请求中完全删除。
        """
        # === 第1步：清洗消息列表 ===
        flatten_messages_as_text = kwargs.pop("flatten_messages_as_text", self.flatten_messages_as_text)
        messages_as_dicts = get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=flatten_messages_as_text,
        )
        # === 第2步：以消息为基础，逐步添加参数 ===
        completion_kwargs = {
            "messages": messages_as_dicts,
        }
        # 添加具体参数（最低优先级）
        if stop_sequences is not None and self.supports_stop_parameter:
            completion_kwargs["stop"] = stop_sequences
        if response_format is not None:
            completion_kwargs["response_format"] = response_format
        if tools_to_call_from:
            # 把 Tool 对象列表转成 JSON Schema 格式
            completion_kwargs["tools"] = [get_tool_json_schema(tool) for tool in tools_to_call_from]
            if tool_choice is not None:
                completion_kwargs["tool_choice"] = tool_choice
        # === 第3步：kwargs 覆盖 ===
        completion_kwargs.update(kwargs)
        # === 第4步：self.kwargs 覆盖（最高优先级） ===
        for kwarg_name, kwarg_value in self.kwargs.items():
            if kwarg_value is REMOVE_PARAMETER:
                completion_kwargs.pop(kwarg_name, None)  # 哨兵值 → 删除参数
            else:
                completion_kwargs[kwarg_name] = kwarg_value  # 正常值 → 覆盖参数
        return completion_kwargs

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """调用 LLM 生成响应（抽象方法，子类必须实现）。

        这是整个框架跟 LLM 交互的唯一入口。
        接收 ChatMessage 列表（对话历史），返回一个 ChatMessage（LLM 的响应）。

        Args:
            messages: 对话历史（ChatMessage 列表）
            stop_sequences: 停止序列，LLM 遇到这些字符串就停止生成
            response_format: 响应格式约束（如 JSON Schema）
            tools_to_call_from: 可调用的工具列表
        """
        raise NotImplementedError("This method must be implemented in child classes")

    def __call__(self, *args, **kwargs):
        """语法糖：model(messages) 等价于 model.generate(messages)。"""
        return self.generate(*args, **kwargs)

    def parse_tool_calls(self, message: ChatMessage) -> ChatMessage:
        """从 LLM 的文本输出中解析工具调用（兜底方案）。

        当 LLM 不支持原生 function calling 时（返回的 ChatMessage 没有 tool_calls），
        就从 content 文本中解析出工具调用。

        处理逻辑：
        1. 如果 message.tool_calls 为空，调用 get_tool_call_from_text 从文本中解析
        2. 确保所有 tool_call 的 arguments 是 dict 而不是 JSON 字符串
        """
        message.role = MessageRole.ASSISTANT
        if not message.tool_calls:
            assert message.content is not None, "Message contains no content and no tool calls"
            message.tool_calls = [
                get_tool_call_from_text(message.content, self.tool_name_key, self.tool_arguments_key)
            ]
        assert len(message.tool_calls) > 0, "No tool call was found in the model output"
        for tool_call in message.tool_calls:
            tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
        return message

    def to_dict(self) -> dict:
        """将模型配置导出为 dict（用于保存和传输）。

        安全设计：故意跳过 token 和 api_key，防止敏感信息泄露。
        """
        model_dictionary = {
            **self.kwargs,
            "model_id": self.model_id,
        }
        for attribute in [
            "custom_role_conversion",
            "temperature",
            "max_tokens",
            "provider",
            "timeout",
            "api_base",
            "torch_dtype",
            "device_map",
            "organization",
            "project",
            "azure_endpoint",
        ]:
            if hasattr(self, attribute):
                model_dictionary[attribute] = getattr(self, attribute)

        # 安全：敏感属性不导出
        dangerous_attributes = ["token", "api_key"]
        for attribute_name in dangerous_attributes:
            if hasattr(self, attribute_name):
                print(
                    f"For security reasons, we do not export the `{attribute_name}` attribute of your model. Please export it manually."
                )
        return model_dictionary

    @classmethod
    def from_dict(cls, model_dictionary: dict[str, Any]) -> "Model":
        """从 dict 创建模型实例（反序列化）。"""
        return cls(**{k: v for k, v in model_dictionary.items()})


class VLLMModel(Model):
    """vLLM 本地推理模型 —— 在本地 GPU 上跑 LLM 推理。

    vLLM 是什么？
        一个高性能的 LLM 推理引擎，专门优化了 GPU 显存管理（PagedAttention），
        能在本地 GPU 上高效运行 HuggingFace 上的开源模型。

    和 API 模型的本质区别：
        API 模型：代码 → HTTP 请求 → 远程服务器推理 → 返回结果
        VLLMModel：代码 → 直接在本地 GPU 推理 → 拿到结果

    继承关系：
        直接继承 Model，跳过 ApiModel。
        因为本地推理不需要 client、rate_limiter、retryer 那三件套。

    __init__ 很重（和 API 模型的轻量 init 不同）：
        要把整个模型加载到 GPU 显存，可能需要几十秒甚至几分钟。
        所以还提供了 cleanup() 方法来释放 GPU 显存。

    generate() 流程和 API 模型不同：
        API 模型：prepare_kwargs → rate_limit → retry(api_call) → ChatMessage
        VLLMModel：prepare_kwargs → 拆参数 → apply_chat_template → SamplingParams → model.generate → ChatMessage

    没有 generate_stream()（不支持流式输出）。

    Parameters:
           (`str`):
            HuggingFace 模型 ID 或本地路径，如：
              - "meta-llama/Llama-3-8B-Instruct"
              - "/path/to/local/model"
        model_kwargs (`dict[str, Any]`, *optional*):
            传给 vLLM LLM() 构造函数的额外参数，控制模型加载方式：
              - revision: 模型版本
              - max_model_len: 最大上下文长度
              - tensor_parallel_size: 多 GPU 并行数
              - gpu_memory_utilization: GPU 显存使用比例（默认 0.9）
        apply_chat_template_kwargs (`dict`, *optional*):
            传给 tokenizer.apply_chat_template() 的额外参数。
            不同模型的 chat template 可能接受不同的参数。
        **kwargs:
            传给 vLLM model.generate() 的额外参数。
    """

    def __init__(
        self,
        model_id,
        model_kwargs: dict[str, Any] | None = None,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        if not _is_package_available("vllm"):
            raise ModuleNotFoundError("Please install 'vllm' extra to use VLLMModel: `pip install 'smolagents[vllm]'`")

        from vllm import LLM  # type: ignore
        from vllm.transformers_utils.tokenizer import get_tokenizer  # type: ignore

        self.model_kwargs = model_kwargs or {}
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        super().__init__(**kwargs)
        self.model_id = model_id
        # 加载模型到 GPU（这一步很慢，可能几十秒到几分钟）
        self.model = LLM(model=model_id, **self.model_kwargs)
        assert self.model is not None
        # 加载分词器（用于 apply_chat_template 把消息列表转成 prompt 文本）
        self.tokenizer = get_tokenizer(model_id)
        self._is_vlm = False  # vLLM 目前不支持视觉模型，所以固定为 False

    def cleanup(self):
        """释放 GPU 显存和分布式环境资源。
        API 模型不需要这个方法（没有占用本地 GPU）。
        本地模型用完后应该调用此方法，否则 GPU 显存不会释放。
        """
        import gc

        import torch
        from vllm.distributed.parallel_state import (  # type: ignore
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        # 销毁多 GPU 并行环境
        destroy_model_parallel()
        if self.model is not None:
            # 删除模型的 worker 进程，释放 GPU 显存
            # 参考：https://github.com/vllm-project/vllm/issues/1908#issuecomment-2076870351
            del self.model.llm_engine.model_executor.driver_worker
        gc.collect()                          # Python 垃圾回收
        destroy_distributed_environment()     # 清理分布式通信
        torch.cuda.empty_cache()              # 释放 GPU 显存缓存

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """本地推理生成。流程和 API 模型不同：

        1. _prepare_completion_kwargs() → 组装参数（OpenAI 格式）
        2. 从 OpenAI 格式中拆出 vLLM 需要的参数
        3. apply_chat_template → 把消息列表转成 prompt 文本
        4. 构建 SamplingParams（vLLM 自己的采样参数格式）
        5. model.generate() → 本地 GPU 推理
        6. 包装为 ChatMessage → 统一返回格式

        没有 rate_limit 和 retryer（本地推理不需要）。
        """
        from vllm import SamplingParams  # type: ignore
        from vllm.sampling_params import StructuredOutputsParams  # type: ignore

        # === 第1步：组装参数（复用父类方法，出来的是 OpenAI 格式） ===
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            flatten_messages_as_text=(not self._is_vlm),  # 文本模型 → True，视觉模型 → False
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        # 处理结构化输出：把 OpenAI 的 response_format 转成 vLLM 的 StructuredOutputsParams
        structured_outputs = (
            StructuredOutputsParams(json=response_format["json_schema"]["schema"]) if response_format else None
        )

        # === 第2步：从 OpenAI 格式中拆出 vLLM 需要的参数 ===
        # _prepare_completion_kwargs 按 OpenAI 格式组装，但 vLLM 的接口不同
        # 所以要手动拆出来，分别传给不同的地方
        messages = completion_kwargs.pop("messages")              # 消息列表 → 给 apply_chat_template
        prepared_stop_sequences = completion_kwargs.pop("stop", [])  # stop → 给 SamplingParams
        tools = completion_kwargs.pop("tools", None)              # 工具列表 → 给 apply_chat_template
        completion_kwargs.pop("tool_choice", None)                # vLLM 不支持 tool_choice，丢弃

        # === 第3步：用 tokenizer 把消息列表转成 prompt 文本 ===
        # API 模型不需要这步（服务端自动做），本地模型必须手动做
        # 例如 Llama-3 的模板会把消息转成：
        #   <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        #   你好<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,                        # 工具描述也会被编入 prompt
            add_generation_prompt=True,          # 在末尾加上 assistant 的开头标记
            tokenize=False,                      # 返回字符串而不是 token ID 列表
            **self.apply_chat_template_kwargs,
        )

        # === 第4步：构建 vLLM 的采样参数 ===
        # vLLM 用 SamplingParams 对象，不是 OpenAI 的 dict
        sampling_params = SamplingParams(
            n=kwargs.get("n", 1),                    # 生成几个候选回答
            temperature=kwargs.get("temperature", 0.0),  # 温度（0 = 确定性输出）
            max_tokens=kwargs.get("max_tokens", 2048),   # 最大生成 token 数
            stop=prepared_stop_sequences,                 # 停止词
            structured_outputs=structured_outputs,        # 结构化输出约束
        )

        # === 第5步：本地 GPU 推理 ===
        out = self.model.generate(
            prompt,
            sampling_params=sampling_params,
            **completion_kwargs,                  # 剩余参数传给 vLLM
        )

        # === 第6步：包装为统一的 ChatMessage ===
        output_text = out[0].outputs[0].text
        if stop_sequences is not None and not self.supports_stop_parameter:
            output_text = remove_content_after_stop_sequences(output_text, stop_sequences)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=output_text,
            raw={"out": output_text, "completion_kwargs": completion_kwargs},
            token_usage=TokenUsage(
                input_tokens=len(out[0].prompt_token_ids),      # 输入 token 数
                output_tokens=len(out[0].outputs[0].token_ids), # 输出 token 数
            ),
        )


class MLXModel(Model):
    """Apple Silicon 本地推理模型 —— 在 Mac 的 M 系列芯片上跑 LLM。

    MLX 是什么？
        Apple 开发的机器学习框架，专门为 M1/M2/M3/M4 芯片优化。
        mlx-lm 是基于 MLX 的 LLM 推理库，类似于 vLLM 之于 NVIDIA GPU。

    和 VLLMModel 的对比：
        VLLMModel → NVIDIA GPU（CUDA），用 vLLM 引擎
        MLXModel  → Apple Silicon（统一内存），用 mlx-lm 引擎

    generate() 流程和 VLLMModel 几乎一样：
        prepare_kwargs → 拆参数 → apply_chat_template → 推理 → ChatMessage

    两个小区别：
        1. apply_chat_template 返回 token ID 列表（不是文本字符串），
           因为 mlx_lm.stream_generate 接受 token ID 作为输入
        2. 推理用 stream_generate（逐 token 流式生成），内部循环拼接文本，
           遇到 stop 词就 break。虽然内部是流式的，但对外返回完整 ChatMessage

    不支持：
        - 视觉模型（_is_vlm = False）
        - 结构化输出（response_format）
        - generate_stream()（没有对外的流式接口）

    Parameters:
        model_id (`str`):
            HuggingFace 模型 ID，通常用 MLX 社区的量化版本：
              - "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit"
              - "mlx-community/Llama-3-8B-Instruct-4bit"
        trust_remote_code (`bool`, default `False`):
            是否信任模型仓库中的远程代码。某些模型需要设为 True。
        load_kwargs (`dict[str, Any]`, *optional*):
            传给 mlx_lm.load() 的额外参数，控制模型加载方式。
        apply_chat_template_kwargs (`dict`, *optional*):
            传给 tokenizer.apply_chat_template() 的额外参数。
        **kwargs:
            传给 mlx_lm.stream_generate() 的额外参数，如 max_tokens。

    Example:
    ```python
    >>> engine = MLXModel(
    ...     model_id="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
    ...     max_tokens=10000,
    ... )
    >>> messages = [
    ...     {
    ...         "role": "user",
    ...         "content": "Explain quantum mechanics in simple terms."
    ...     }
    ... ]
    >>> response = engine(messages, stop_sequences=["END"])
    >>> print(response)
    "Quantum mechanics is the branch of physics that studies..."
    ```
    """

    def __init__(
        self,
        model_id: str,
        trust_remote_code: bool = False,
        load_kwargs: dict[str, Any] | None = None,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        if not _is_package_available("mlx_lm"):
            raise ModuleNotFoundError(
                "Please install 'mlx-lm' extra to use 'MLXModel': `pip install 'smolagents[mlx-lm]'`"
            )
        import mlx_lm

        self.load_kwargs = load_kwargs or {}
        # 把 trust_remote_code 塞进 tokenizer 配置
        self.load_kwargs.setdefault("tokenizer_config", {}).setdefault("trust_remote_code", trust_remote_code)
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        # 默认加上 assistant 开头标记（和 VLLMModel 的 add_generation_prompt=True 一样）
        self.apply_chat_template_kwargs.setdefault("add_generation_prompt", True)
        # mlx-lm 不支持视觉模型，所以固定 flatten_messages_as_text=True
        super().__init__(model_id=model_id, flatten_messages_as_text=True, **kwargs)

        # 加载模型和分词器到 Apple Silicon 的统一内存（和 VLLMModel 的 LLM() 对应）
        self.model, self.tokenizer = mlx_lm.load(self.model_id, **self.load_kwargs)
        self.stream_generate = mlx_lm.stream_generate
        self.is_vlm = False  # mlx-lm 不支持视觉模型

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """本地推理生成，流程和 VLLMModel 类似但有两个区别：
        1. apply_chat_template 返回 token ID（不是文本）
        2. 用 stream_generate 逐 token 生成，内部循环拼接
        """
        if response_format is not None:
            raise ValueError("MLX does not support structured outputs.")
        # === 第1步：组装参数（OpenAI 格式） ===
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        # === 第2步：拆参数（和 VLLMModel 一样） ===
        messages = completion_kwargs.pop("messages")
        stops = completion_kwargs.pop("stop", [])
        tools = completion_kwargs.pop("tools", None)
        completion_kwargs.pop("tool_choice", None)

        # === 第3步：apply_chat_template ===
        # 注意：这里返回的是 token ID 列表（不是文本字符串）
        # 因为 mlx_lm.stream_generate 接受 token ID 作为 prompt 输入
        prompt_ids = self.tokenizer.apply_chat_template(messages, tools=tools, **self.apply_chat_template_kwargs)

        # === 第4步 + 第5步：流式推理 + 逐 token 拼接 ===
        # 虽然内部是流式的，但对外返回完整结果（不是 yield）
        output_tokens = 0
        text = ""
        for response in self.stream_generate(self.model, self.tokenizer, prompt=prompt_ids, **completion_kwargs):
            output_tokens += 1
            text += response.text
            # 检查是否遇到了停止词，遇到就截断并停止
            if any((stop_index := text.rfind(stop)) != -1 for stop in stops):
                text = text[:stop_index]
                break
        if stop_sequences is not None and not self.supports_stop_parameter:
            text = remove_content_after_stop_sequences(text, stop_sequences)
        # === 第6步：包装为 ChatMessage ===
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=text,
            raw={"out": text, "completion_kwargs": completion_kwargs},
            token_usage=TokenUsage(
                input_tokens=len(prompt_ids),       # 输入 token 数
                output_tokens=output_tokens,        # 输出 token 数（循环计数）
            ),
        )


class TransformersModel(Model):
    """HuggingFace Transformers 本地推理模型 —— 最通用的本地模型。

    Transformers 是什么？
        HuggingFace 的核心库，几乎所有 HuggingFace Hub 上的模型都能用它加载和推理。
        支持 CPU 和 GPU，是最灵活的选择（但性能不如 vLLM 那么极致）。

    和其他本地模型的对比：
        VLLMModel        → vLLM 引擎，专为高吞吐量优化，只支持 NVIDIA GPU
        MLXModel         → mlx-lm 引擎，专为 Apple Silicon 优化
        TransformersModel → Transformers，通用，CPU/GPU 都行，最灵活

    三个独特之处：
        1. 自动检测视觉模型：先尝试加载为视觉模型（AutoModelForImageTextToText），
           失败了再当文本模型（AutoModelForCausalLM）加载
        2. 自定义停止条件：Transformers 不直接支持字符串 stop，
           需要用 StoppingCriteria 回调来实现
        3. 支持 generate_stream()：用多线程 + TextIteratorStreamer 实现流式输出

    Parameters:
        model_id (`str`):
            HuggingFace 模型 ID 或本地路径，如：
              - "Qwen/Qwen3-Next-80B-A3B-Thinking"
              - "meta-llama/Llama-3-8B-Instruct"
        device_map (`str`, *optional*):
            模型放在哪个设备上。默认自动检测：有 CUDA 用 GPU，没有用 CPU。
            也可以传 "auto" 让 Transformers 自动分配多 GPU。
        torch_dtype (`str`, *optional*):
            模型精度，如 "float16", "bfloat16"。低精度省显存但可能损失精度。
        trust_remote_code (`bool`, default `False`):
            是否信任模型仓库中的远程代码。某些模型需要设为 True。
        model_kwargs (`dict[str, Any]`, *optional*):
            传给 AutoModel.from_pretrained() 的额外参数。
        max_new_tokens (`int`, default `4096`):
            最大生成 token 数。
        max_tokens (`int`, *optional*):
            max_new_tokens 的别名，传了会覆盖 max_new_tokens。
        apply_chat_template_kwargs (`dict`, *optional*):
            传给 tokenizer.apply_chat_template() 的额外参数。
        **kwargs: 传给 model.generate() 的额外参数。

    Example:
    ```python
    >>> engine = TransformersModel(
    ...     model_id="Qwen/Qwen3-Next-80B-A3B-Thinking",
    ...     device="cuda",
    ...     max_new_tokens=5000,
    ... )
    >>> messages = [{"role": "user", "content": "Explain quantum mechanics in simple terms."}]
    >>> response = engine(messages, stop_sequences=["END"])
    >>> print(response)
    "Quantum mechanics is the branch of physics that studies..."
    ```
    """

    def __init__(
        self,
        model_id: str | None = None,
        device_map: str | None = None,
        torch_dtype: str | None = None,
        trust_remote_code: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        max_new_tokens: int = 4096,
        max_tokens: int | None = None,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoProcessor,
                AutoTokenizer,
                TextIteratorStreamer,
            )
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'transformers' extra to use 'TransformersModel': `pip install 'smolagents[transformers]'`"
            )

        # model_id 在 2.0 版本将变为必填参数
        if not model_id:
            warnings.warn(
                "The 'model_id' parameter will be required in version 2.0.0. "
                "Please update your code to pass this parameter to avoid future errors. "
                "For now, it defaults to 'HuggingFaceTB/SmolLM2-1.7B-Instruct'.",
                FutureWarning,
            )
            model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

        # max_tokens 是 max_new_tokens 的别名，传了就覆盖
        max_new_tokens = max_tokens if max_tokens is not None else max_new_tokens

        # 自动检测设备：有 CUDA 用 GPU，没有用 CPU
        if device_map is None:
            device_map = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device_map}")
        self._is_vlm = False
        self.model_kwargs = model_kwargs or {}
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        # === 自动检测视觉模型 ===
        # 先尝试加载为视觉模型（能处理图片+文字）
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                **self.model_kwargs,
            )
            # 视觉模型用 processor（能同时处理图片和文字）
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            self._is_vlm = True    # 标记为视觉模型
            # 流式输出用的 streamer（推理线程写入，主线程读取）
            self.streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore

        except ValueError as e:
            # 加载视觉模型失败 → 当普通文本模型加载
            if "Unrecognized configuration class" in str(e):
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                    **self.model_kwargs,
                )
                # 文本模型用 tokenizer（只处理文字）
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
                self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore
            else:
                raise e
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer and model for {model_id=}: {e}") from e
        super().__init__(
            flatten_messages_as_text=not self._is_vlm, model_id=model_id, max_new_tokens=max_new_tokens, **kwargs
        )

    def make_stopping_criteria(self, stop_sequences: list[str], tokenizer) -> "StoppingCriteriaList":
        """创建自定义停止条件。

        Transformers 的 model.generate() 不直接支持字符串 stop 参数
        （不像 OpenAI/vLLM 那样传 stop=["END"] 就行）。
        需要用 StoppingCriteria 回调：每生成一个 token 就检查"要不要停"。

        原理：把每个新 token 解码成文字，拼到累积字符串上，
        检查是否以某个停止词结尾。如果是，返回 True 停止生成。
        """
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopOnStrings(StoppingCriteria):
            def __init__(self, stop_strings: list[str], tokenizer):
                self.stop_strings = stop_strings
                self.tokenizer = tokenizer
                self.stream = ""  # 累积已生成的文字

            def reset(self):
                self.stream = ""

            def __call__(self, input_ids, scores, **kwargs):
                # 把最新生成的 token 解码成文字
                generated = self.tokenizer.decode(input_ids[0][-1], skip_special_tokens=True)
                self.stream += generated
                # 检查累积文字是否以停止词结尾
                if any([self.stream.endswith(stop_string) for stop_string in self.stop_strings]):
                    return True   # 停止生成
                return False      # 继续生成

        return StoppingCriteriaList([StopOnStrings(stop_sequences, tokenizer)])

    def _prepare_completion_args(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """把 OpenAI 格式参数翻译成 Transformers model.generate() 的参数。

        和 VLLMModel 的"拆参数"逻辑类似，但多了几步：
        1. 调父类 _prepare_completion_kwargs 拿到 OpenAI 格式
        2. 拆出 messages、stop、tools
        3. apply_chat_template → 返回 PyTorch tensor（不是文本）
        4. 把 tensor 移到模型所在设备（GPU/CPU）
        5. 构建 StoppingCriteria
        6. 打包返回
        """
        # 第1步：父类方法组装 OpenAI 格式参数
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            tool_choice=None,  # Transformers 不支持 tool_choice
            **kwargs,
        )

        # 第2步：拆参数
        messages = completion_kwargs.pop("messages")
        stop_sequences = completion_kwargs.pop("stop", None)
        tools = completion_kwargs.pop("tools", None)

        # 解析 max_new_tokens：从多个来源中取值（优先级从高到低）
        max_new_tokens = (
            kwargs.get("max_new_tokens")
            or kwargs.get("max_tokens")
            or self.kwargs.get("max_new_tokens")
            or self.kwargs.get("max_tokens")
            or 1024
        )
        # 第3步：apply_chat_template → PyTorch tensor
        # 注意和 VLLMModel 的区别：
        #   VLLMModel: tokenize=False → 返回文本字符串
        #   TransformersModel: return_tensors="pt" → 返回 PyTorch tensor
        prompt_tensor = (self.processor if hasattr(self, "processor") else self.tokenizer).apply_chat_template(
            messages,
            tools=tools,
            return_tensors="pt",             # 返回 PyTorch tensor
            add_generation_prompt=True,
            tokenize=True,                   # 直接 tokenize
            return_dict=True,
            **self.apply_chat_template_kwargs,
        )
        # 第4步：把 tensor 移到模型所在设备（GPU/CPU）
        prompt_tensor = prompt_tensor.to(self.model.device)  # type: ignore
        if hasattr(prompt_tensor, "input_ids"):
            prompt_tensor = prompt_tensor["input_ids"]

        # 第5步：构建停止条件，通过hassttr(self,"processor")判断是不是文本模型
        model_tokenizer = self.processor.tokenizer if hasattr(self, "processor") else self.tokenizer
        stopping_criteria = (
            self.make_stopping_criteria(stop_sequences, tokenizer=model_tokenizer) if stop_sequences else None
        )
        completion_kwargs["max_new_tokens"] = max_new_tokens
        # 第6步：打包返回 Transformers model.generate() 需要的参数
        return dict(
            inputs=prompt_tensor,                    # 输入 token tensor
            use_cache=True,                          # 启用 KV Cache 加速
            stopping_criteria=stopping_criteria,     # 自定义停止条件
            **completion_kwargs,
        )

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """非流式生成。流程：
        1. _prepare_completion_args() → 组装 Transformers 格式参数
        2. model.generate() → 本地推理，返回完整的 token ID 序列
        3. decode → 把 token ID 转回文字
        4. 包装为 ChatMessage
        """
        if response_format is not None:
            raise ValueError("Transformers does not support structured outputs, use VLLMModel for this.")
        # === 第1步：组装参数 ===
        generation_kwargs = self._prepare_completion_args(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        # 记录输入 token 数（用于统计）
        count_prompt_tokens = generation_kwargs["inputs"].shape[1]  # type: ignore
        # === 第2步：本地推理 ===
        out = self.model.generate(
            **generation_kwargs,
        )
        # out 包含输入+输出的完整 token 序列，需要截掉输入部分
        generated_tokens = out[0, count_prompt_tokens:]
        # === 第3步：decode → 文字 ===
        # 视觉模型用 processor.decode，文本模型用 tokenizer.decode
        if hasattr(self, "processor"):
            output_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
        else:
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if stop_sequences is not None:
            output_text = remove_content_after_stop_sequences(output_text, stop_sequences)
        # === 第4步：包装为 ChatMessage ===
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=output_text,
            raw={
                "out": output_text,
                "completion_kwargs": {key: value for key, value in generation_kwargs.items() if key != "inputs"},
            },
            token_usage=TokenUsage(
                input_tokens=count_prompt_tokens,
                output_tokens=len(generated_tokens),
            ),
        )

    def generate_stream(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        """流式生成 —— 用多线程 + TextIteratorStreamer 实现。

        原理：
        - 推理在子线程中运行（model.generate 是阻塞的）
        - 子线程每生成一个 token 就写入 streamer
        - 主线程从 streamer 中逐 token 读取并 yield
        - streamer 就像一个管道，连接推理线程和主线程
        """
        if response_format is not None:
            raise ValueError("Transformers does not support structured outputs, use VLLMModel for this.")
        generation_kwargs = self._prepare_completion_args(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )

        # 记录输入 token 数
        count_prompt_tokens = generation_kwargs["inputs"].shape[1]  # type: ignore

        # 在子线程中启动推理（因为 model.generate 是阻塞的）
        # streamer 参数让推理过程把每个 token 写入 streamer
        thread = Thread(target=self.model.generate, kwargs={"streamer": self.streamer, **generation_kwargs})
        thread.start()

        # 主线程从 streamer 中逐 token 读取并 yield
        is_first_token = True
        count_generated_tokens = 0
        for new_text in self.streamer:
            count_generated_tokens += 1
            # 只在第一个 token 时报告输入 token 数
            input_tokens = count_prompt_tokens if is_first_token else 0
            is_first_token = False
            yield ChatMessageStreamDelta(
                content=new_text,
                tool_calls=None,
                token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=1),
            )
            count_prompt_tokens = 0
        # 等待推理线程结束
        thread.join()

        # 记录总输出 token 数（供外部统计用）
        self._last_output_token_count = count_generated_tokens


class ApiModel(Model):
    """API 模型的基类，在 Model 基础上增加了 3 个关键能力：

    1. client —— API 客户端（如 openai.OpenAI()），由子类的 create_client() 创建
    2. rate_limiter —— 速率限制器，控制每分钟请求数，防止被 API 封禁
    3. retryer —— 重试器，遇到速率限制错误时自动重试（指数退避 + 抖动）

    继承关系：
        Model（纯逻辑骨架）
          └── ApiModel（+ client + rate_limiter + retryer）
                ├── OpenAIModel（OpenAI / 兼容 API）
                ├── LiteLLMModel（通过 LiteLLM 统一调各家 API）
                ├── InferenceClientModel（HuggingFace Inference API）
                └── AmazonBedrockModel（AWS Bedrock）

    与本地模型（VLLMModel, TransformersModel）的区别：
        本地模型直接继承 Model，不需要 client/rate_limiter/retryer，
        因为推理在本地 GPU 上跑，没有 API 调用。

    Parameters:
        model_id (`str`):
            模型标识符，告诉 API "我要用哪个模型"。
            例如：
              - OpenAI: "gpt-4o", "gpt-4o-mini", "o3"
              - DeepSeek: "deepseek-chat"
              - Claude: "claude-3-opus"（通过兼容 API 调用时）
        custom_role_conversions (`dict[str, str]`, *optional*):
            角色名映射表。有些模型不支持某些角色，需要转换。
            例如：Google Gemini 不支持 "system" 角色，可以传：
              custom_role_conversions={"system": "user"}
            这样所有 system 消息会被当作 user 消息发送。
            默认为空 dict（不做任何转换）。
        client (`Any`, *optional*):
            预配置的 API 客户端实例。通常不需要传，ApiModel 会自动调
            子类的 create_client() 创建。但如果你需要自定义客户端配置
            （比如自定义 HTTP 代理、超时时间），可以自己创建后传进来：
              import openai
              my_client = openai.OpenAI(api_key="sk-xxx", timeout=60)
              model = OpenAIModel("gpt-4o", client=my_client)
        requests_per_minute (`float`, *optional*):
            每分钟最大请求数。用于主动控制调用频率，避免触发 API 的
            429 Too Many Requests 限制。例如：
              model = OpenAIModel("gpt-4o", requests_per_minute=30)
            不传则不限速（完全靠 API 端的限制 + retryer 兜底）。
        retry (`bool`, *optional*):
            是否在遇到速率限制错误（429）时自动重试。默认 True。
            设为 False 时遇到 429 直接抛异常，适合你想自己处理错误的场景。
            重试策略：指数退避 + 随机抖动，最多重试 RETRY_MAX_ATTEMPTS 次。
        **kwargs:
            传给底层 API 调用的额外参数，会存到 self.kwargs 中，
            在 _prepare_completion_kwargs 中以最高优先级覆盖。
            例如：
              model = OpenAIModel("gpt-4o", temperature=0.7, max_tokens=4096)
              # 之后每次 generate() 都会带上 temperature=0.7, max_tokens=4096
    """

    def __init__(
        self,
        model_id: str,
        custom_role_conversions: dict[str, str] | None = None,
        client: Any | None = None,
        requests_per_minute: float | None = None,
        retry: bool = True,
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.custom_role_conversions = custom_role_conversions or {}
        # 如果用户传了 client 就用用户的，否则调子类的 create_client() 创建
        self.client = client or self.create_client()
        # 速率限制器：控制请求频率，避免触发 API 的 429 Too Many Requests
        self.rate_limiter = RateLimiter(requests_per_minute)
        # 重试器：遇到速率限制错误时，等待一段时间后自动重试
        # 使用指数退避策略：每次等待时间翻倍（+ 随机抖动避免多个请求同时重试）
        self.retryer = Retrying(
            max_attempts=RETRY_MAX_ATTEMPTS if retry else 1,  # retry=False 时只试 1 次
            wait_seconds=RETRY_WAIT,           # 初始等待秒数
            exponential_base=RETRY_EXPONENTIAL_BASE,  # 指数底数（每次等待时间 × 这个数）
            jitter=RETRY_JITTER,               # 随机抖动范围
            retry_predicate=is_rate_limit_error,  # 判断函数：只有速率限制错误才重试
            reraise=True,                      # 重试耗尽后重新抛出原始异常
            before_sleep_logger=(logger, logging.INFO),
            after_logger=(logger, logging.INFO),
        )

    def create_client(self):
        """创建 API 客户端。子类必须实现此方法。
        例如 OpenAIModel 会返回 openai.OpenAI(api_key=..., base_url=...)
        """
        raise NotImplementedError("Subclasses must implement this method to create a client")

    def _apply_rate_limit(self):
        """在调 API 之前调用，确保不超过速率限制。"""
        self.rate_limiter.throttle()


def is_rate_limit_error(exception: BaseException) -> bool:
    """判断一个异常是否是速率限制错误（429 Too Many Requests）。

    被 ApiModel 的 retryer 用作 retry_predicate：
      - 返回 True → 值得重试（等一下就好了）
      - 返回 False → 不重试（比如认证错误，重试也没用）

    做法：把异常信息转成字符串，看里面有没有关键词。
    不管是哪家 API 的错误对象，只要包含这些词就算速率限制错误。
    """
    error_str = str(exception).lower()
    return (
        "429" in error_str
        or "rate limit" in error_str
        or "too many requests" in error_str
        or "rate_limit" in error_str
    )


class LiteLLMModel(ApiModel):
    """LiteLLM 模型 —— 通过 LiteLLM 库统一调用各家 API 的"万能适配器"。

    LiteLLM 是什么？
        一个 Python 库，把 OpenAI、Anthropic、Google、AWS Bedrock、Ollama 等
        几百个 LLM 的 API 统一成一个接口。你只需要换 model_id 就能切换模型：
          - "gpt-4o"                    → OpenAI
          - "anthropic/claude-3-opus"   → Anthropic
          - "ollama/llama3"             → 本地 Ollama
          - "bedrock/claude-3"          → AWS Bedrock

    和 OpenAIModel 的区别：
        - OpenAIModel 直接用 openai 库，只能调 OpenAI 兼容的 API
        - LiteLLMModel 用 litellm 库，能调几乎所有 LLM API
        - generate() 流程几乎一样，只是把 client.chat.completions.create
          换成了 litellm.completion

    特殊处理：
        - create_client() 返回的是 litellm 模块本身（不是一个实例），
          因为 litellm 的调用方式是 litellm.completion()，不需要创建客户端对象
        - 某些本地/轻量 API（ollama, groq, cerebras）默认开启 flatten_messages_as_text

    Parameters:
        model_id (`str`):
            LiteLLM 格式的模型标识符。格式通常是 "provider/model_name"：
              - "gpt-4o"                    （OpenAI 可以省略 provider）
              - "anthropic/claude-3-opus"
              - "ollama/llama3"
              - "bedrock/anthropic.claude-3"
        api_base (`str`, *optional*):
            API 地址。大多数情况不需要传（LiteLLM 自动处理），
            但如果用自建服务（如本地 Ollama），需要指定：
              api_base="http://localhost:11434"
        api_key (`str`, *optional*):
            API 密钥。也可以通过环境变量设置（如 OPENAI_API_KEY）。
        custom_role_conversions (`dict[str, str]`, *optional*):
            角色名映射，同 OpenAIModel。
        flatten_messages_as_text (`bool`, *optional*):
            是否把消息压成纯文本。默认行为：
              - model_id 以 "ollama"/"groq"/"cerebras" 开头 → True
              - 其他 → False
        **kwargs: 传给 litellm.completion() 的额外参数。
    """

    def __init__(
        self,
        model_id: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool | None = None,
        **kwargs,
    ):
        # model_id 在 2.0 版本将变为必填参数
        if not model_id:
            warnings.warn(
                "The 'model_id' parameter will be required in version 2.0.0. "
                "Please update your code to pass this parameter to avoid future errors. "
                "For now, it defaults to 'anthropic/claude-3-5-sonnet-20240620'.",
                FutureWarning,
            )
            model_id = "anthropic/claude-3-5-sonnet-20240620"
        # api_base 和 api_key 存起来，generate() 时传给 litellm.completion()
        self.api_base = api_base
        self.api_key = api_key
        # 自动判断是否需要 flatten：ollama/groq/cerebras 这些轻量 API 需要纯文本
        flatten_messages_as_text = (
            flatten_messages_as_text
            if flatten_messages_as_text is not None
            else model_id.startswith(("ollama", "groq", "cerebras"))
        )
        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        """返回 litellm 模块本身作为"客户端"。
        因为 litellm 的调用方式是 litellm.completion()（模块级函数），
        不像 openai 需要先创建 openai.OpenAI() 实例。
        """
        try:
            import litellm
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'litellm' extra to use LiteLLMModel: `pip install 'smolagents[litellm]'`"
            ) from e

        return litellm

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """标准 4 步流程，和 OpenAIModel.generate() 几乎一样。
        区别：
          1. 调的是 litellm.completion() 而不是 client.chat.completions.create()
          2. 额外传了 api_base 和 api_key（LiteLLM 需要这些来路由到正确的 provider）
          3. 多了一个 response.choices 为空的检查（某些 provider 可能返回空结果）
        """
        # === 第1步：组装参数 ===
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            api_base=self.api_base,       # LiteLLM 特有：告诉它 API 地址
            api_key=self.api_key,         # LiteLLM 特有：告诉它用哪个密钥
            convert_images_to_image_urls=True,
            custom_role_conversions=self.custom_role_conversions,
            **kwargs,
        )
        # === 第2步：速率限制 ===
        self._apply_rate_limit()
        # === 第3步：带重试的 API 调用 ===
        # litellm.completion() 是模块级函数，等价于 self.client.completion()
        response = self.retryer(self.client.completion, **completion_kwargs)

        # LiteLLM 特有的防御性检查：某些 provider 可能返回空 choices
        if not response.choices:
            raise RuntimeError(
                f"Unexpected API response: model '{self.model_id}' returned no choices. "
                " This may indicate a possible API or upstream issue. "
                f"Response details: {response.model_dump()}"
            )
        # === 第4步：包装为 ChatMessage ===
        content = response.choices[0].message.content
        if stop_sequences is not None and not self.supports_stop_parameter:
            content = remove_content_after_stop_sequences(content, stop_sequences)
        return ChatMessage(
            role=response.choices[0].message.role,
            content=content,
            tool_calls=response.choices[0].message.tool_calls,
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

    def generate_stream(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        """流式版本，和 OpenAIModel.generate_stream() 几乎一样。"""
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        self._apply_rate_limit()
        for event in self.retryer(
            self.client.completion, **completion_kwargs, stream=True, stream_options={"include_usage": True}
        ):
            if getattr(event, "usage", None):
                yield ChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )
            if event.choices:
                choice = event.choices[0]
                if choice.delta:
                    yield ChatMessageStreamDelta(
                        content=choice.delta.content,
                        tool_calls=[
                            ChatMessageToolCallStreamDelta(
                                index=delta.index,
                                id=delta.id,
                                type=delta.type,
                                function=delta.function,
                            )
                            for delta in choice.delta.tool_calls
                        ]
                        if choice.delta.tool_calls
                        else None,
                    )
                else:
                    if not getattr(choice, "finish_reason", None):
                        raise ValueError(f"No content or tool calls in event: {event}")


class LiteLLMRouterModel(LiteLLMModel):
    """LiteLLM 路由模型 —— LiteLLMModel 的"负载均衡版本"。

    解决的问题：
        普通 LiteLLMModel 每次都打同一个模型。但生产环境中你可能想：
        - 把请求随机分配给 GPT-4o 或 Claude-3（分散压力）
        - GPT-4o 挂了自动切到 Claude-3（故障转移）
        - 给便宜的模型分配更多流量（成本控制）

    和 LiteLLMModel 的区别（只有两处）：
        1. __init__ 多了 model_list 参数（配置多个模型）
        2. create_client() 返回 litellm.Router 而不是 litellm 模块
        generate() 和 generate_stream() 完全继承，一行没改。
        因为 Router 和 litellm 模块有相同的 .completion() 接口。

    model_list 的理解方式：
        model_id 就像"前台电话号码"，model_list 就像"后面坐着的几个客服"，
        Router 根据策略决定这次电话转给谁接。
        多个配置用相同的 model_name → 属于同一个组 → Router 从组里选一个调用。

    Parameters:
        model_id (`str`):
            模型组名（逻辑名），对应 model_list 中的 model_name。
            例如 "model-group-1"，Router 会从这个组里选一个实际模型来调。
        model_list (`list[dict[str, Any]]`):
            模型配置列表，每个元素定义一个可用的实际模型。格式：
              [
                {"model_name": "组名", "litellm_params": {"model": "gpt-4o", "api_key": "..."}},
                {"model_name": "组名", "litellm_params": {"model": "claude-3", "api_key": "..."}},
              ]
            同一个 model_name 的多个配置 = 同一个组里的多个候选模型。
        client_kwargs (`dict[str, Any]`, *optional*):
            Router 的额外配置，如路由策略：
              - "simple-shuffle"：随机选
              - "least-busy"：选最空闲的
              - "latency-based-routing"：选延迟最低的
        custom_role_conversions (`dict[str, str]`, *optional*):
            角色名映射，同 LiteLLMModel。
        flatten_messages_as_text (`bool`, *optional*):
            是否压成纯文本，同 LiteLLMModel。
        **kwargs: 传给底层 completion 调用的额外参数。

    Example:
    ```python
    >>> import os
    >>> from smolagents import CodeAgent, WebSearchTool, LiteLLMRouterModel
    >>> os.environ["OPENAI_API_KEY"] = ""
    >>> os.environ["AWS_ACCESS_KEY_ID"] = ""
    >>> os.environ["AWS_SECRET_ACCESS_KEY"] = ""
    >>> os.environ["AWS_REGION"] = ""
    >>> llm_loadbalancer_model_list = [
    ...     {
    ...         "model_name": "model-group-1",
    ...         "litellm_params": {
    ...             "model": "gpt-4o-mini",
    ...             "api_key": os.getenv("OPENAI_API_KEY"),
    ...         },
    ...     },
    ...     {
    ...         "model_name": "model-group-1",
    ...         "litellm_params": {
    ...             "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    ...             "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
    ...             "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
    ...             "aws_region_name": os.getenv("AWS_REGION"),
    ...         },
    ...     },
    >>> ]
    >>> model = LiteLLMRouterModel(
    ...    model_id="model-group-1",
    ...    model_list=llm_loadbalancer_model_list,
    ...    client_kwargs={
    ...        "routing_strategy":"simple-shuffle"
    ...    }
    >>> )
    >>> agent = CodeAgent(tools=[WebSearchTool()], model=model)
    >>> agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
    ```
    """

    def __init__(
        self,
        model_id: str,
        model_list: list[dict[str, Any]],
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool | None = None,
        **kwargs,
    ):
        # 把 model_list 和额外配置合并，后面 create_client() 会用
        self.client_kwargs = {
            "model_list": model_list,       # Router 需要知道有哪些候选模型
            **(client_kwargs or {}),        # 路由策略等额外配置
        }
        # 其他全交给 LiteLLMModel（→ ApiModel → Model）
        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        """创建 LiteLLM Router 客户端。
        和 LiteLLMModel 的区别：返回 Router 实例而不是 litellm 模块。
        Router 有和 litellm 相同的 .completion() 接口，
        但内部会根据路由策略从 model_list 中选一个实际模型来调用。
        """
        try:
            from litellm.router import Router
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'litellm' extra to use LiteLLMRouterModel: `pip install 'smolagents[litellm]'`"
            ) from e
        return Router(**self.client_kwargs)


class InferenceClientModel(ApiModel):
    """HuggingFace 推理模型 —— 通过 HuggingFace Inference API 调用模型。

    HuggingFace Inference API 是什么？
        HuggingFace 提供的云端推理服务，你不需要自己部署模型，
        直接调 API 就能用 HuggingFace Hub 上的模型。
        支持多个推理提供商：Cerebras, Fireworks, Together, Nebius 等。

    和 OpenAIModel 的区别：
        - 客户端：用 huggingface_hub.InferenceClient 而不是 openai.OpenAI
        - API 方法：调 client.chat_completion() 而不是 client.chat.completions.create()
        - 认证：用 HuggingFace token 而不是 OpenAI API key
        - 额外功能：支持选择推理提供商（provider）、账单归属（bill_to）

    generate() 流程和 OpenAIModel 一样是标准 4 步，只是：
        1. 多了 response_format 的 provider 兼容性检查
        2. 调的是 client.chat_completion() 而不是 client.chat.completions.create()

    Parameters:
        model_id (`str`, *optional*, default `"Qwen/Qwen3-Next-80B-A3B-Thinking"`):
            HuggingFace 模型 ID，如 "meta-llama/Llama-3-70B-Instruct"。
            也可以是部署好的 Inference Endpoint 的 URL。
        provider (`str`, *optional*):
            推理提供商名称，如 "hyperbolic", "together", "fireworks" 等。
            默认 "auto"（自动选择第一个可用的提供商）。
        token (`str`, *optional*):
            HuggingFace API token。不传则从环境变量 HF_TOKEN 读取。
            如果模型是 gated 的（如 Llama-3），token 还需要有读取权限。
        timeout (`int`, *optional*, defaults to 120):
            API 请求超时时间（秒）。
        client_kwargs (`dict[str, Any]`, *optional*):
            传给 InferenceClient 的额外参数。
        custom_role_conversions (`dict[str, str]`, *optional*):
            角色名映射，同 OpenAIModel。
        api_key (`str`, *optional*):
            token 的别名，为了和 OpenAI 客户端的参数名保持一致。
            不能和 token 同时传。
        bill_to (`str`, *optional*):
            账单归属的组织名。默认计费到个人账户，
            也可以指定一个你所属的 Enterprise Hub 组织。
        base_url (`str`, *optional*):
            自定义 API 地址。如果传了这个，provider 参数会被忽略。
        **kwargs: 传给底层 chat_completion() 的额外参数。

    Example:
    ```python
    >>> engine = InferenceClientModel(
    ...     model_id="Qwen/Qwen3-Next-80B-A3B-Thinking",
    ...     provider="hyperbolic",
    ...     token="your_hf_token_here",
    ...     max_tokens=5000,
    ... )
    >>> messages = [{"role": "user", "content": "Explain quantum mechanics in simple terms."}]
    >>> response = engine(messages, stop_sequences=["END"])
    >>> print(response)
    "Quantum mechanics is the branch of physics that studies..."
    ```
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-Next-80B-A3B-Thinking",
        provider: str | None = None,
        token: str | None = None,
        timeout: int = 120,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        api_key: str | None = None,
        bill_to: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        # token 和 api_key 是同一个东西，不能同时传
        if token is not None and api_key is not None:
            raise ValueError(
                "Received both `token` and `api_key` arguments. Please provide only one of them."
                " `api_key` is an alias for `token` to make the API compatible with OpenAI's client."
                " It has the exact same behavior as `token`."
            )
        token = token if token is not None else api_key
        # 都没传就从环境变量读
        if token is None:
            token = os.getenv("HF_TOKEN")
        # 收集所有客户端参数，create_client() 会用
        self.client_kwargs = {
            **(client_kwargs or {}),
            "model": model_id,
            "provider": provider,
            "token": token,
            "timeout": timeout,
            "bill_to": bill_to,
            "base_url": base_url,
        }
        super().__init__(model_id=model_id, custom_role_conversions=custom_role_conversions, **kwargs)

    def create_client(self):
        """创建 HuggingFace InferenceClient。
        和 OpenAIModel 的 openai.OpenAI() 对应。
        """
        from huggingface_hub import InferenceClient

        return InferenceClient(**self.client_kwargs)

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """标准 4 步流程，和 OpenAIModel.generate() 几乎一样。
        区别：
          1. 多了 response_format 的 provider 兼容性检查（不是所有 provider 都支持结构化输出）
          2. 调的是 client.chat_completion() 而不是 client.chat.completions.create()
        """
        # HuggingFace 特有：检查 provider 是否支持结构化输出
        if response_format is not None and self.client_kwargs["provider"] not in STRUCTURED_GENERATION_PROVIDERS:
            raise ValueError(
                "InferenceClientModel only supports structured outputs with these providers:"
                + ", ".join(STRUCTURED_GENERATION_PROVIDERS)
            )
        # === 第1步：组装参数 ===
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            # response_format=response_format,  # 暂时注释掉了，可能还在开发中
            convert_images_to_image_urls=True,
            custom_role_conversions=self.custom_role_conversions,
            **kwargs,
        )
        # === 第2步：速率限制 ===
        self._apply_rate_limit()
        # === 第3步：带重试的 API 调用 ===
        # 注意：是 chat_completion 不是 chat.completions.create
        response = self.retryer(self.client.chat_completion, **completion_kwargs)
        # === 第4步：包装为 ChatMessage ===
        content = response.choices[0].message.content
        if stop_sequences is not None and not self.supports_stop_parameter:
            content = remove_content_after_stop_sequences(content, stop_sequences)
        return ChatMessage(
            role=response.choices[0].message.role,
            content=content,
            tool_calls=response.choices[0].message.tool_calls,
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

    def generate_stream(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        """流式版本，和 OpenAIModel.generate_stream() 一样。"""
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        self._apply_rate_limit()
        for event in self.retryer(
            self.client.chat.completions.create,
            **completion_kwargs,
            stream=True,
            stream_options={"include_usage": True},
        ):
            if getattr(event, "usage", None):
                yield ChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )
            if event.choices:
                choice = event.choices[0]
                if choice.delta:
                    yield ChatMessageStreamDelta(
                        content=choice.delta.content,
                        tool_calls=[
                            ChatMessageToolCallStreamDelta(
                                index=delta.index,
                                id=delta.id,
                                type=delta.type,
                                function=delta.function,
                            )
                            for delta in choice.delta.tool_calls
                        ]
                        if choice.delta.tool_calls
                        else None,
                    )
                else:
                    if not getattr(choice, "finish_reason", None):
                        raise ValueError(f"No content or tool calls in event: {event}")


class OpenAIModel(ApiModel):
    """OpenAI 兼容 API 模型 —— 所有 API 模型子类的"标准模板"。

    这是最典型的 API 模型实现，其 generate() 方法展示了标准的 4 步流程：
        1. _prepare_completion_kwargs() → 组装参数
        2. _apply_rate_limit()          → 速率限制
        3. retryer(api_call)            → 带重试的 API 调用
        4. 包装为 ChatMessage            → 统一返回格式

    不仅支持 OpenAI 官方 API，也支持任何兼容 OpenAI 接口的服务
    （如 vLLM serve、Ollama、LocalAI 等），只需设置 api_base 即可。

    Parameters:
        model_id (`str`): 模型标识符（如 "gpt-4o", "deepseek-chat"）
        api_base (`str`, *optional*): API 地址，默认为 OpenAI 官方地址
        api_key (`str`, *optional*): API 密钥
        organization (`str`, *optional*): OpenAI 组织 ID
        project (`str`, *optional*): OpenAI 项目 ID
        client_kwargs (`dict[str, Any]`, *optional*): 传给 openai.OpenAI() 的额外参数
        custom_role_conversions (`dict[str, str]`, *optional*): 角色名映射
        flatten_messages_as_text (`bool`, default `False`): 是否压成纯文本
        **kwargs: 传给 chat.completions.create() 的额外参数（如 temperature）
    """

    def __init__(
        self,
        model_id: str,
        api_base: str | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool = False,
        **kwargs,
    ):
        # 先把所有客户端参数收集到一个 dict 里，后面 create_client() 会用
        self.client_kwargs = {
            **(client_kwargs or {}),
            "api_key": api_key,
            "base_url": api_base,
            "organization": organization,
            "project": project,
        }
        # 注意：super().__init__() 内部会调 self.create_client()
        # 所以 self.client_kwargs 必须在 super() 之前设置好
        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        """创建 OpenAI 客户端。被 ApiModel.__init__() 自动调用。"""
        try:
            import openai
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use OpenAIModel: `pip install 'smolagents[openai]'`"
            ) from e

        return openai.OpenAI(**self.client_kwargs)

    def generate_stream(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        """流式生成：逐块返回 LLM 的输出（用于实时显示打字效果）。

        与 generate() 的区别：
        - generate() 等 LLM 全部生成完才返回一个完整的 ChatMessage
        - generate_stream() 边生成边返回 ChatMessageStreamDelta 片段
          调用方用 agglomerate_stream_deltas() 把片段拼成完整消息
        """
        # === 第1步：组装参数（和 generate 一样） ===
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        # === 第2步：速率限制 ===
        self._apply_rate_limit()
        # === 第3步：带重试的流式 API 调用 ===
        # stream=True 让 API 返回一个事件流（Server-Sent Events）
        for event in self.retryer(
            self.client.chat.completions.create,
            **completion_kwargs,
            stream=True,
            stream_options={"include_usage": True},  # 最后一个事件包含 token 用量
        ):
            # 最后一个事件包含 usage 信息（token 用量统计）
            if event.usage:
                yield ChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )
            # 正常的内容/工具调用片段
            if event.choices:
                choice = event.choices[0]
                if choice.delta:
                    yield ChatMessageStreamDelta(
                        content=choice.delta.content,
                        tool_calls=[
                            ChatMessageToolCallStreamDelta(
                                index=delta.index,
                                id=delta.id,
                                type=delta.type,
                                function=delta.function,
                            )
                            for delta in choice.delta.tool_calls
                        ]
                        if choice.delta.tool_calls
                        else None,
                    )
                else:
                    if not getattr(choice, "finish_reason", None):
                        raise ValueError(f"No content or tool calls in event: {event}")

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """非流式生成 —— API 模型的标准 4 步流程：

        1. _prepare_completion_kwargs() → 组装所有参数
        2. _apply_rate_limit()          → 检查速率限制
        3. retryer(api_call)            → 调 API（自动重试）
        4. 包装为 ChatMessage            → 统一返回格式
        """
        # === 第1步：组装参数 ===
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,                    # OpenAI API 需要 model 参数
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,       # OpenAI 用 URL 格式传图片
            **kwargs,
        )
        # === 第2步：速率限制 ===
        self._apply_rate_limit()
        # === 第3步：带重试的 API 调用 ===
        # retryer 包装了 client.chat.completions.create()
        # 如果遇到 429 错误，会自动等待并重试
        response = self.retryer(self.client.chat.completions.create, **completion_kwargs)
        # === 第4步：包装为统一的 ChatMessage ===
        content = response.choices[0].message.content
        if stop_sequences is not None and not self.supports_stop_parameter:
            content = remove_content_after_stop_sequences(content, stop_sequences)
        return ChatMessage(
            role=response.choices[0].message.role,
            content=content,
            tool_calls=response.choices[0].message.tool_calls,  # 可能为 None
            raw=response,                           # 保留原始响应，方便调试
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )


OpenAIServerModel = OpenAIModel


class AzureOpenAIModel(OpenAIModel):
    """Azure OpenAI 模型 —— OpenAIModel 的"换皮"版本。

    和 OpenAIModel 的唯一区别：
        1. __init__ 多了 azure_endpoint 和 api_version 两个参数
        2. create_client() 返回 openai.AzureOpenAI() 而不是 openai.OpenAI()

    generate() 和 generate_stream() 完全继承自 OpenAIModel，一行都没改。
    因为 Azure OpenAI 的 API 接口和 OpenAI 官方完全一样，只是入口地址不同。

    用法示例：
        model = AzureOpenAIModel(
            model_id="gpt-4o-mini",                    # Azure 上的部署名称
            azure_endpoint="https://xxx.openai.azure.com/",
            api_key="your-azure-key",
            api_version="2024-02-01",
        )

    也可以不传 api_key / azure_endpoint，它们会自动从环境变量读取：
        AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, OPENAI_API_VERSION

    Parameters:
        model_id (`str`):
            Azure 上的模型部署名称（不是模型名，是你在 Azure 控制台创建的部署名）。
            例如你部署了一个 gpt-4o-mini，部署名叫 "my-gpt4o"，就传 "my-gpt4o"。
        azure_endpoint (`str`, *optional*):
            Azure 端点地址，如 `https://example-resource.openai.azure.com/`。
            不传则从环境变量 AZURE_OPENAI_ENDPOINT 读取。
        api_key (`str`, *optional*):
            API 密钥。不传则从环境变量 AZURE_OPENAI_API_KEY 读取。
        api_version (`str`, *optional*):
            API 版本号，如 "2024-02-01"。不传则从环境变量 OPENAI_API_VERSION 读取。
        client_kwargs (`dict[str, Any]`, *optional*):
            传给 AzureOpenAI() 客户端的额外参数。
        custom_role_conversions (`dict[str, str]`, *optional*):
            角色名映射，同 OpenAIModel。
        **kwargs:
            传给底层 API 调用的额外参数（如 temperature）。
    """

    def __init__(
        self,
        model_id: str,
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        **kwargs,
    ):
        # 把 Azure 特有的参数塞进 client_kwargs
        client_kwargs = client_kwargs or {}
        client_kwargs.update(
            {
                "api_version": api_version,
                "azure_endpoint": azure_endpoint,
            }
        )
        # 剩下的全交给 OpenAIModel.__init__()
        # → OpenAIModel 会把 client_kwargs 存到 self.client_kwargs
        # → 然后 ApiModel.__init__() 会调 self.create_client()
        # → 走到下面的 create_client()，用 AzureOpenAI 而不是 OpenAI
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            client_kwargs=client_kwargs,
            custom_role_conversions=custom_role_conversions,
            **kwargs,
        )

    def create_client(self):
        """创建 Azure OpenAI 客户端。
        和 OpenAIModel.create_client() 唯一的区别：openai.AzureOpenAI vs openai.OpenAI
        """
        try:
            import openai
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use AzureOpenAIModel: `pip install 'smolagents[openai]'`"
            ) from e

        return openai.AzureOpenAI(**self.client_kwargs)


AzureOpenAIServerModel = AzureOpenAIModel


class AmazonBedrockModel(ApiModel):
    """AWS Bedrock 模型 —— 通过 AWS Bedrock API 调用各种模型。

    AWS Bedrock 是什么？
        AWS 提供的托管 LLM 服务，可以调用 Amazon Nova、Claude、Llama 等模型，
        不需要自己部署，按用量计费。类似 OpenAI API，但在 AWS 生态内。

    和 OpenAIModel 的主要区别：
        1. 客户端：用 boto3.client("bedrock-runtime") 而不是 openai.OpenAI()
        2. API 方法：调 client.converse() 而不是 client.chat.completions.create()
        3. 响应格式不同：Bedrock 返回的是嵌套 dict，不是 OpenAI 风格的对象
           - OpenAI: response.choices[0].message.content
           - Bedrock: response["output"]["message"]["content"][-1]["text"]
        4. 角色限制：Bedrock 只支持 "user" 和 "assistant" 两个角色，
           所以默认把所有角色都映射成 "user"
        5. 重写了 _prepare_completion_kwargs()：需要做 Bedrock 特有的参数适配
        6. 不支持 response_format（结构化输出）
        7. 不支持 generate_stream()（没实现流式）

    认证方式：
        - 默认 AWS 凭证链（IAM 角色、IAM 用户等）
        - API Key 认证（需要 boto3 >= 1.39.0，通过 AWS_BEARER_TOKEN_BEDROCK 环境变量）

    Parameters:
        model_id (`str`):
            Bedrock 模型标识符，如 "us.amazon.nova-pro-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0"。
        client (`boto3.client`, *optional*):
            预配置的 boto3 客户端。不传则自动创建。
        client_kwargs (`dict[str, Any]`, *optional*):
            创建 boto3 客户端时的参数，如：
              client_kwargs={"region_name": "us-west-2"}
        custom_role_conversions (`dict[str, str]`, *optional*):
            角色名映射。默认把所有角色都映射成 "user"，
            因为 Bedrock 大多数模型只支持 user/assistant 两个角色。
        **kwargs:
            传给 client.converse() 的额外参数，如：
              inferenceConfig={"maxTokens": 3000}
              guardrailConfig={"guardrailIdentifier": "id1", "guardrailVersion": "v1"}

    Examples:
        Creating a model instance with default settings:
        ```python
        >>> bedrock_model = AmazonBedrockModel(
        ...     model_id='us.amazon.nova-pro-v1:0'
        ... )
        ```

        Creating a model instance with a custom boto3 client:
        ```python
        >>> import boto3
        >>> client = boto3.client('bedrock-runtime', region_name='us-west-2')
        >>> bedrock_model = AmazonBedrockModel(
        ...     model_id='us.amazon.nova-pro-v1:0',
        ...     client=client
        ... )
        ```

        Creating a model instance with client_kwargs for internal client creation:
        ```python
        >>> bedrock_model = AmazonBedrockModel(
        ...     model_id='us.amazon.nova-pro-v1:0',
        ...     client_kwargs={'region_name': 'us-west-2', 'endpoint_url': 'https://custom-endpoint.com'}
        ... )
        ```

        Creating a model instance with inference and guardrail configurations:
        ```python
        >>> additional_api_config = {
        ...     "inferenceConfig": {
        ...         "maxTokens": 3000
        ...     },
        ...     "guardrailConfig": {
        ...         "guardrailIdentifier": "identify1",
        ...         "guardrailVersion": 'v1'
        ...     },
        ... }
        >>> bedrock_model = AmazonBedrockModel(
        ...     model_id='anthropic.claude-3-haiku-20240307-v1:0',
        ...     **additional_api_config
        ... )
        ```
    """

    def __init__(
        self,
        model_id: str,
        client=None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        **kwargs,
    ):
        self.client_kwargs = client_kwargs or {}

        # Bedrock 只支持 "user" 和 "assistant" 两个角色
        # 而且很多模型不允许对话以 assistant 开头
        # 所以默认把所有角色都映射成 user（最安全的做法）
        custom_role_conversions = custom_role_conversions or {
            MessageRole.SYSTEM: MessageRole.USER,
            MessageRole.ASSISTANT: MessageRole.USER,
            MessageRole.TOOL_CALL: MessageRole.USER,
            MessageRole.TOOL_RESPONSE: MessageRole.USER,
        }

        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=False,  # Bedrock API 要求消息是列表格式，不能压成纯文本
            client=client,
            **kwargs,
        )

    def _prepare_completion_kwargs(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        convert_images_to_image_urls: bool = False,
        tool_choice: str | dict[Any, Any] | None = None,
        **kwargs,
    ) -> dict:
        """重写父类方法，做 Bedrock 特有的参数适配。

        Bedrock 的 API 格式和 OpenAI 不同，需要额外处理：
        1. stop_sequences 不能直接传，要通过 inferenceConfig 传
        2. toolConfig 要移除（smolagents 已经在 prompt 里包含了工具信息）
        3. 消息内容中的 "type" 字段要删除（Bedrock API 不认识）
        4. 要加上 modelId 参数
        """
        # 先调父类方法组装基础参数
        completion_kwargs = super()._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=None,  # Bedrock 用 inferenceConfig 传 stop，不是顶层参数
            tools_to_call_from=tools_to_call_from,
            custom_role_conversions=custom_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            **kwargs,
        )
        # smolagents 已经在 prompt 中描述了工具，不需要 Bedrock 的 toolConfig
        # 同时传两种工具描述可能导致冲突
        completion_kwargs.pop("toolConfig", None)

        # Bedrock API 不支持消息内容中的 "type" 字段
        # OpenAI 格式：{"type": "text", "text": "hello"}
        # Bedrock 格式：{"text": "hello"}（没有 type）
        for message in completion_kwargs.get("messages", []):
            for content in message.get("content", []):
                if "type" in content:
                    del content["type"]

        # Bedrock 用 modelId 而不是 model
        return {
            "modelId": self.model_id,
            **completion_kwargs,
        }

    def create_client(self):
        """创建 AWS Bedrock 客户端。
        用 boto3（AWS 的 Python SDK）创建 bedrock-runtime 服务客户端。
        """
        try:
            import boto3  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'bedrock' extra to use AmazonBedrockServerModel: `pip install 'smolagents[bedrock]'`"
            ) from e

        return boto3.client("bedrock-runtime", **self.client_kwargs)

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """标准 4 步流程，但响应解析和 OpenAI 不同。

        Bedrock 返回的是嵌套 dict（不是 OpenAI 风格的对象）：
          response["output"]["message"]["content"] → 内容块列表
          response["usage"]["inputTokens"]         → token 用量

        另外 Bedrock 可能返回"思考块"（thinking blocks），
        需要过滤掉，只取包含 "text" 的内容块。
        """
        if response_format is not None:
            raise ValueError("Amazon Bedrock does not support response_format")
        # === 第1步：组装参数（用重写后的 _prepare_completion_kwargs） ===
        completion_kwargs: dict = self._prepare_completion_kwargs(
            messages=messages,
            tools_to_call_from=tools_to_call_from,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        # === 第2步：速率限制 ===
        self._apply_rate_limit()
        # === 第3步：带重试的 API 调用 ===
        # Bedrock 用 converse() 而不是 chat.completions.create()
        response = self.retryer(self.client.converse, **completion_kwargs)

        # === 第4步：解析响应，包装为 ChatMessage ===
        # Bedrock 的响应可能包含"思考块"（thinking blocks），只取有 "text" 的块
        message_content_blocks_with_text = [
            block for block in response["output"]["message"]["content"] if "text" in block
        ]
        if not message_content_blocks_with_text:
            raise KeyError("No message content blocks with 'text' key found in response")
        # 取最后一个文本块（思考块在前，最终回答在后）
        content = message_content_blocks_with_text[-1]["text"]
        if stop_sequences is not None and not self.supports_stop_parameter:
            content = remove_content_after_stop_sequences(content, stop_sequences)
        return ChatMessage(
            role=response["output"]["message"]["role"],
            content=content,
            tool_calls=response["output"]["message"]["tool_calls"],
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response["usage"]["inputTokens"],    # Bedrock 用驼峰命名
                output_tokens=response["usage"]["outputTokens"],  # 不是 prompt_tokens
            ),
        )


AmazonBedrockServerModel = AmazonBedrockModel


# Model Registry for secure deserialization
# This registry maps model class names to their actual classes.
# 模型注册表：用于安全的反序列化（从 JSON 恢复模型对象）
# 序列化时：OpenAIModel 类 → "OpenAIModel" 字符串 → 存进 JSON
# 反序列化时：JSON 读出 "OpenAIModel" → 查此表 → 拿到 OpenAIModel 类 → 创建实例
# 只有此表中列出的类才允许被创建，防止恶意 JSON 执行任意代码
MODEL_REGISTRY = {
    "VLLMModel": VLLMModel,
    "MLXModel": MLXModel,
    "TransformersModel": TransformersModel,
    "LiteLLMModel": LiteLLMModel,
    "LiteLLMRouterModel": LiteLLMRouterModel,
    "InferenceClientModel": InferenceClientModel,
    "OpenAIModel": OpenAIModel,
    "AzureOpenAIModel": AzureOpenAIModel,
    "AmazonBedrockModel": AmazonBedrockModel,
}

__all__ = [
    "REMOVE_PARAMETER",
    "MessageRole",
    "tool_role_conversions",
    "get_clean_message_list",
    "Model",
    "MLXModel",
    "TransformersModel",
    "ApiModel",
    "InferenceClientModel",
    "LiteLLMModel",
    "LiteLLMRouterModel",
    "OpenAIServerModel",
    "OpenAIModel",
    "VLLMModel",
    "AzureOpenAIServerModel",
    "AzureOpenAIModel",
    "AmazonBedrockServerModel",
    "AmazonBedrockModel",
    "ChatMessage",
]
