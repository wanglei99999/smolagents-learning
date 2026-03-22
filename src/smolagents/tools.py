#!/usr/bin/env python
# coding=utf-8

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
# tools.py —— 工具系统的核心文件
#
# 本文件定义了 smolagents 中"工具"的抽象层，包含两种定义工具的方式：
#
# 方式一：@tool 装饰器（快速方式）
#   - 最简单的定义方式，用于包装普通 Python 函数
#   - 必须提供类型注解 (type hints) 和 Args: 文档字符串
#   - 框架通过这些信息自动生成工具的 name / description / inputs / output_type
#   - 示例：
#       @tool
#       def get_weather(location: str, celsius: bool = True) -> str:
#           """获取某地天气。
#           Args:
#               location: 城市名称
#               celsius: 是否使用摄氏度
#           """
#           return f"{location}: 晴，25°C"
#
# 方式二：继承 Tool 类（完整方式）
#   - 提供完全控制，适合需要延迟初始化（如加载模型）的工具
#   - 必须定义 name, description, inputs, output_type 类属性
#   - 核心逻辑写在 forward() 方法中
#   - 重写 setup() 可实现延迟加载（第一次调用时才执行）
#
# 其他重要类：
#   - ToolCollection：工具集合，支持从 HuggingFace Hub 或 MCP 服务器批量加载工具
#   - validate_tool_arguments：检查工具参数有效性（由 Agent 在调用前使用）
#
# 工具的 inputs 格式：
#   inputs = {
#       "param_name": {
#           "type": "string",        # AUTHORIZED_TYPES 中的类型名
#           "description": "说明",   # LLM 理解此参数用途的关键
#           "nullable": True,        # 可选参数需要设置（对应 Python 的 Optional）
#       }
#   }
# =============================================================================

from __future__ import annotations

import ast
import inspect
import json
import logging
import os
import sys
import tempfile
import textwrap
import types
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import (
    CommitOperationAdd,
    create_commit,
    create_repo,
    get_collection,
    hf_hub_download,
    metadata_update,
)

from ._function_type_hints_utils import (
    TypeHintParsingException,
    _convert_type_hints_to_json_schema,
    _get_json_schema_type,
    get_imports,
    get_json_schema,
)
from .agent_types import AgentAudio, AgentImage, handle_agent_input_types, handle_agent_output_types
from .tool_validation import MethodChecker, validate_tool_attributes
from .utils import (
    BASE_BUILTIN_MODULES,
    _is_package_available,
    get_source,
    instance_to_source,
    is_valid_name,
)


if TYPE_CHECKING:
    import mcp


logger = logging.getLogger(__name__)


def validate_after_init(cls):
    """类装饰器：在 __init__ 执行完成后自动调用 validate_arguments() 进行属性验证。
    通过 __init_subclass__ 钩子，所有 Tool 子类在实例化时都会自动触发验证。
    这样可以在工具定义有误时立即报错，而不是到运行时才发现问题。
    """
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.validate_arguments()

    cls.__init__ = new_init
    return cls


# 工具参数和返回值允许使用的类型名称（在 inputs["type"] 和 output_type 中使用）
# 注意：这些是字符串名称，非 Python 类型对象
AUTHORIZED_TYPES = [
    "string",
    "boolean",
    "integer",
    "number",
    "image",
    "audio",
    "array",
    "object",
    "any",
    "null",
]

# Python 原生类型名 → AUTHORIZED_TYPES 类型名 的转换映射
# 用于从 LangChain 工具等外部格式导入时的类型转换
CONVERSION_DICT = {"str": "string", "int": "integer", "float": "number"}


class BaseTool(ABC):
    name: str

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        pass


class Tool(BaseTool):
    """
    工具基类 - 所有工具的父类
    
    这是 smolagents 中定义工具的标准方式。继承这个类可以创建自定义工具。
    
    必须定义的类属性：
    - name (str): 工具名称，必须是有效的 Python 标识符（LLM 会用这个名字调用工具）
    - description (str): 工具描述，告诉 LLM 这个工具是做什么的
    - inputs (dict): 输入参数的 JSON Schema 格式定义
    - output_type (str): 输出类型（必须是 AUTHORIZED_TYPES 中的一个）
    
    可选的类属性：
    - output_schema (dict): 输出的 JSON Schema（用于结构化输出）
    
    必须实现的方法：
    - forward(*args, **kwargs): 工具的核心逻辑
    
    可选重写的方法：
    - setup(): 延迟初始化（如加载模型），第一次调用时执行
    
    A base class for the functions used by the agent. Subclass this and implement the `forward` method as well as the
    following class attributes:

    - **description** (`str`) -- A short description of what your tool does, the inputs it expects and the output(s) it
      will return. For instance 'This is a tool that downloads a file from a `url`. It takes the `url` as input, and
      returns the text contained in the file'.
    - **name** (`str`) -- A performative name that will be used for your tool in the prompt to the agent. For instance
      `"text-classifier"` or `"image_generator"`.
    - **inputs** (`Dict[str, Dict[str, Union[str, type, bool]]]`) -- The dict of modalities expected for the inputs.
      It has one `type`key and a `description`key.
      This is used by `launch_gradio_demo` or to make a nice space from your tool, and also can be used in the generated
      description for your tool.
    - **output_type** (`type`) -- The type of the tool output. This is used by `launch_gradio_demo`
      or to make a nice space from your tool, and also can be used in the generated description for your tool.
    - **output_schema** (`Dict[str, Any]`, *optional*) -- The JSON schema defining the expected structure of the tool output.
      This can be included in system prompts to help agents understand the expected output format. Note: This is currently
      used for informational purposes only and does not perform actual output validation.

    You can also override the method [`~Tool.setup`] if your tool has an expensive operation to perform before being
    usable (such as loading a model). [`~Tool.setup`] will be called the first time you use your tool, but not at
    instantiation.
    
    Example:
        ```python
        class WeatherTool(Tool):
            name = "get_weather"
            description = "获取指定城市的天气信息"
            inputs = {
                "location": {"type": "string", "description": "城市名称"},
                "celsius": {"type": "boolean", "description": "是否使用摄氏度", "nullable": True}
            }
            output_type = "string"
            
            def forward(self, location: str, celsius: bool = True):
                # 实际的天气查询逻辑
                return f"{location}: 晴天，25°C"
        ```
    """

    name: str
    description: str
    inputs: dict[str, dict[str, str | type | bool]]
    output_type: str
    output_schema: dict[str, Any] | None = None

    def __init__(self, *args, **kwargs):
        self.is_initialized = False

    def __init_subclass__(cls, **kwargs):
        """Python 类创建钩子：每当有新类继承 Tool 时，自动被 Python 解释器调用。

        作用：通过 validate_after_init() 将子类的 __init__ 替换为"带验证的版本"，
        使得子类实例化后自动执行 validate_arguments()，无需开发者手动调用。

        触发时机：类定义时（不是实例化时）。例如：
            class MyTool(Tool):  # ← 这一行就会触发 Tool.__init_subclass__(MyTool)
                ...

        继承链也会生效：
            class A(Tool): ...   # 触发
            class B(A): ...      # 也触发（沿继承链向上找到 Tool 的这个方法）

        Args:
            cls: 被创建的子类本身（由 Python 自动传入，不是实例）
            **kwargs: 传递给 super().__init_subclass__ 的关键字参数
        """
        super().__init_subclass__(**kwargs)
        validate_after_init(cls)

    def validate_arguments(self):
        """验证工具定义的完整性和正确性。

        本方法通过 validate_after_init 装饰器在每个 Tool 子类实例化后自动调用，
        确保工具在使用前就暴露定义错误，而不是等到 Agent 运行时才出问题。

        验证分为五个阶段：
        1. 必需类属性检查（name/description/inputs/output_type 是否存在且类型正确）
        2. 工具名合法性检查（必须是合法的 Python 标识符，不能是保留字）
        3. inputs 结构检查（每个参数必须有 type 和 description，type 必须在 AUTHORIZED_TYPES 中）
        4. output_type 检查（必须在 AUTHORIZED_TYPES 中）
        5. forward() 签名一致性检查（参数名必须和 inputs 的 key 完全一致，nullable 也要对应）

        Raises:
            TypeError: 属性类型不匹配，或 inputs 中 type 字段格式错误
            ValueError: inputs 中使用了不在 AUTHORIZED_TYPES 中的类型
            Exception: name 不是合法标识符，或 forward() 签名与 inputs 不一致
            AssertionError: inputs 结构缺少必需字段，或 output_type 不合法
        """
        # ===== 第一阶段：验证四个必需的类属性是否存在且类型正确 =====
        # 这四个属性是 Tool 的"身份证"，缺一不可
        required_attributes = {
            "description": str,
            "name": str,
            "inputs": dict,
            "output_type": str,
        }
        for attr, expected_type in required_attributes.items():
            attr_value = getattr(self, attr, None)
            if attr_value is None:
                raise TypeError(f"You must set an attribute {attr}.")
            if not isinstance(attr_value, expected_type):
                raise TypeError(
                    f"Attribute {attr} should have type {expected_type.__name__}, got {type(attr_value)} instead."
                )

        # 验证可选的 output_schema 属性（用于结构化输出，不是必须的）
        output_schema = getattr(self, "output_schema", None)
        if output_schema is not None and not isinstance(output_schema, dict):
            raise TypeError(f"Attribute output_schema should have type dict, got {type(output_schema)} instead.")

        # ===== 第二阶段：验证工具名是否是合法的 Python 标识符 =====
        # 因为 CodeAgent 会生成 `tool_name(args)` 这样的代码，name 必须是合法标识符
        # 例如 "get_weather" ✓, "get-weather" ✗, "class" ✗（保留字）
        if not is_valid_name(self.name):
            raise Exception(
                f"Invalid Tool name '{self.name}': must be a valid Python identifier and not a reserved keyword"
            )

        # ===== 第三阶段：验证 inputs 字典的结构和类型 =====
        # 每个参数必须是 {"type": "...", "description": "..."} 格式
        for input_name, input_content in self.inputs.items():
            assert isinstance(input_content, dict), f"Input '{input_name}' should be a dictionary."
            assert "type" in input_content and "description" in input_content, (
                f"Input '{input_name}' should have keys 'type' and 'description', has only {list(input_content.keys())}."
            )
            # type 可以是单个字符串 "string"，也可以是列表 ["string", "null"]（联合类型）
            if isinstance(input_content["type"], str):
                input_types = [input_content["type"]]
            elif isinstance(input_content["type"], list):
                input_types = input_content["type"]
                if not all(isinstance(t, str) for t in input_types):
                    raise TypeError(
                        f"Input '{input_name}': when type is a list, all elements must be strings, got {input_content['type']}"
                    )
            else:
                raise TypeError(
                    f"Input '{input_name}': type must be a string or list of strings, got {type(input_content['type']).__name__}"
                )
            # 确保所有类型都在白名单中（AUTHORIZED_TYPES: string, boolean, integer, number, image, audio, array, object, any, null）
            invalid_types = [t for t in input_types if t not in AUTHORIZED_TYPES]
            if invalid_types:
                raise ValueError(f"Input '{input_name}': types {invalid_types} must be one of {AUTHORIZED_TYPES}")

        # ===== 第四阶段：验证 output_type 是否在白名单中 =====
        assert getattr(self, "output_type", None) in AUTHORIZED_TYPES

        # ===== 第五阶段：验证 forward() 方法签名与 inputs 的一致性 =====
        # 某些特殊工具（PipelineTool, SpaceToolWrapper, LangChainToolWrapper）使用通用签名 *args/**kwargs，
        # 通过 skip_forward_signature_validation = True 跳过此检查
        if not (
            hasattr(self, "skip_forward_signature_validation")
            and getattr(self, "skip_forward_signature_validation") is True
        ):
            # 检查 forward() 的参数名是否和 inputs 的 key 完全一致
            # 例如 inputs={"query": ..., "top_k": ...} → forward(self, query, top_k)
            signature = inspect.signature(self.forward)
            actual_keys = set(key for key in signature.parameters.keys() if key != "self")
            expected_keys = set(self.inputs.keys())
            if actual_keys != expected_keys:
                raise Exception(
                    f"In tool '{self.name}', 'forward' method parameters were {actual_keys}, but expected {expected_keys}. "
                    f"It should take 'self' as its first argument, then its next arguments should match the keys of tool attribute 'inputs'."
                )

            # 从 forward() 的类型注解中提取 JSON Schema，用于交叉验证 nullable 属性
            # 确保 inputs 中标记为 nullable 的参数，在 forward() 签名中也是 Optional 类型，反之亦然
            json_schema = _convert_type_hints_to_json_schema(self.forward, error_on_missing_type_hints=False)[
                "properties"
            ]
            for key, value in self.inputs.items():
                assert key in json_schema, (
                    f"Input '{key}' should be present in function signature, found only {json_schema.keys()}"
                )
                # 双向检查：inputs 说 nullable → 签名也必须是 Optional；签名是 Optional → inputs 也必须说 nullable
                if "nullable" in value:
                    assert "nullable" in json_schema[key], (
                        f"Nullable argument '{key}' in inputs should have key 'nullable' set to True in function signature."
                    )
                if key in json_schema and "nullable" in json_schema[key]:
                    assert "nullable" in value, (
                        f"Nullable argument '{key}' in function signature should have key 'nullable' set to True in inputs."
                    )

    def forward(self, *args, **kwargs):
        """工具的核心逻辑方法，子类必须重写。

        与 __call__ 的分工类似于 PyTorch 的 nn.Module：
        - __call__() 处理框架层面的杂活（延迟初始化、参数兼容、类型转换）
        - forward()  只负责工具的实际功能（搜索、计算、文件处理等）

        子类实现时，参数名必须与 inputs 字典的 key 完全一致，例如：
            inputs = {"query": {"type": "string", ...}, "top_k": {"type": "integer", ...}}
            def forward(self, query: str, top_k: int):  # 参数名必须是 query 和 top_k
                ...

        不要直接调用 forward()，应该通过 tool(args) 即 __call__() 调用，
        这样才能享受延迟初始化和类型转换等框架功能。
        """
        raise NotImplementedError("Write this method in your subclass of `Tool`.")

    def __call__(self, *args, sanitize_inputs_outputs: bool = False, **kwargs):
        """工具调用入口。
        sanitize_inputs_outputs=True 时（由 Agent 调用），会对 AgentImage/AgentAudio 等类型进行
        自动转换，确保图像/音频在 Agent 工具链中以正确格式传递。
        """
        if not self.is_initialized:
            self.setup()  # 延迟初始化：第一次调用时才执行 setup（如加载模型）

        # 兼容处理：若参数以单个 dict 形式传入，自动展开为 kwargs
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            potential_kwargs = args[0]
            if all(key in self.inputs for key in potential_kwargs):
                args = ()
                kwargs = potential_kwargs

        if sanitize_inputs_outputs:
            args, kwargs = handle_agent_input_types(*args, **kwargs)
        outputs = self.forward(*args, **kwargs)
        if sanitize_inputs_outputs:
            outputs = handle_agent_output_types(outputs, self.output_type)
        return outputs

    def setup(self):
        """
        Overwrite this method here for any operation that is expensive and needs to be executed before you start using
        your tool. Such as loading a big model.

        懒操作用的函数，把比较重的操作放到这里去实现，而不是在类init中实现。
        """
        self.is_initialized = True

    def to_code_prompt(self) -> str:
        """为 CodeAgent 生成 Python 函数签名风格的工具描述，嵌入到系统提示词中。

        CodeAgent 通过生成 Python 代码来调用工具，所以这里把工具描述成一个 Python 函数，
        让 LLM 自然地生成 `result = tool_name(arg1, arg2)` 这样的调用代码。

        生成的格式示例（普通工具）：
            def web_search(query: string) -> string:
                \"\"\"Performs a web search...
                Args:
                    query: The search query to perform.
                \"\"\"

        生成的格式示例（有 output_schema 的结构化输出工具）：
            def get_info(query: string) -> dict:
                \"\"\"获取信息...
                Important: This tool returns structured output! ...
                Args:
                    query: ...
                Returns:
                    dict (structured output): ... JSON schema ...
                \"\"\"

        Returns:
            str: Python 伪代码格式的工具描述
        """
        # 拼接参数签名，如 "query: string, top_k: integer"
        args_signature = ", ".join(f"{arg_name}: {arg_schema['type']}" for arg_name, arg_schema in self.inputs.items())

        # 若工具有结构化输出 Schema，返回类型标注为 dict（提示 LLM 用字段访问方式）
        has_schema = hasattr(self, "output_schema") and self.output_schema is not None
        output_type = "dict" if has_schema else self.output_type
        tool_signature = f"({args_signature}) -> {output_type}"
        tool_doc = self.description

        # 对于结构化输出工具，追加提示语，帮助小模型正确处理 dict 返回值
        if has_schema:
            tool_doc += "\n\nImportant: This tool returns structured output! Use the JSON schema below to directly access fields like result['field_name']. NO print() statements needed to inspect the output!"

        # 拼接参数文档（Args 段落）
        if self.inputs:
            args_descriptions = "\n".join(
                f"{arg_name}: {arg_schema['description']}" for arg_name, arg_schema in self.inputs.items()
            )
            args_doc = f"Args:\n{textwrap.indent(args_descriptions, '    ')}"
            tool_doc += f"\n\n{args_doc}"

        # 若有结构化输出，追加 Returns 段落和完整的 JSON Schema
        if has_schema:
            formatted_schema = json.dumps(self.output_schema, indent=4)
            indented_schema = textwrap.indent(formatted_schema, "        ")
            returns_doc = f"\nReturns:\n    dict (structured output): This tool ALWAYS returns a dictionary that strictly adheres to the following JSON schema:\n{indented_schema}"
            tool_doc += f"\n{returns_doc}"

        # 包装成 Python docstring 格式
        tool_doc = f'"""{tool_doc}\n"""'
        return f"def {self.name}{tool_signature}:\n{textwrap.indent(tool_doc, '    ')}"

    def to_tool_calling_prompt(self) -> str:
        """为 ToolCallingAgent 生成纯文本风格的工具描述，嵌入到系统提示词中。

        ToolCallingAgent 通过生成 JSON 来调用工具（而非 Python 代码），
        所以这里用更简洁的纯文本格式描述工具，让 LLM 生成如下调用：
            {"tool": "web_search", "arguments": {"query": "Python tutorial"}}

        生成的格式示例：
            web_search: Performs a web search...
                Takes inputs: {"query": {"type": "string", "description": "..."}}
                Returns an output of type: string

        Returns:
            str: 纯文本格式的工具描述
        """
        return f"{self.name}: {self.description}\n    Takes inputs: {self.inputs}\n    Returns an output of type: {self.output_type}"

    def to_dict(self) -> dict:
        """将工具序列化为字典，包含完整源代码，用于 save()/push_to_hub()/from_dict() 等场景。

        根据工具的创建方式走两条路径：
        - @tool 装饰器创建的（SimpleTool）：从原函数源码反向生成完整的 Tool 子类代码
        - 继承 Tool 类创建的：直接用 instance_to_source() 提取整个类的源码

        Returns:
            dict: {"name": 工具名, "code": 完整类源码, "requirements": 依赖列表, "output_schema": 可选}
        """
        class_name = self.__class__.__name__
        if type(self).__name__ == "SimpleTool":
            # ===== 路径一：@tool 装饰器创建的工具 =====
            # 这种工具只有一个函数，需要反向生成完整的类代码

            # 第一步：拿到原函数源码，用 AST 检查是否"自包含"（不依赖外部变量）
            # 如果函数里引用了外部全局变量，序列化后别人那边没有这个变量就会报错
            source_code = get_source(self.forward).replace("@tool", "")
            forward_node = ast.parse(source_code)
            method_checker = MethodChecker(set())
            method_checker.visit(forward_node)

            if len(method_checker.errors) > 0:
                errors = [f"- {error}" for error in method_checker.errors]
                raise (ValueError(f"SimpleTool validation failed for {self.name}:\n" + "\n".join(errors)))

            # 第二步：手动拼出完整的 Tool 子类代码（类定义 + 类属性）
            forward_source_code = get_source(self.forward)
            tool_code = textwrap.dedent(
                f"""
            from smolagents import Tool
            from typing import Any, Optional

            class {class_name}(Tool):
                name = "{self.name}"
                description = {json.dumps(textwrap.dedent(self.description).strip())}
                inputs = {repr(self.inputs)}
                output_type = "{self.output_type}"
            """
            ).strip()

            if hasattr(self, "output_schema") and self.output_schema is not None:
                tool_code += f"\n                output_schema = {repr(self.output_schema)}"
            import re

            def add_self_argument(source_code: str) -> str:
                """Add 'self' as first argument to a function definition if not present."""
                pattern = r"def forward\(((?!self)[^)]*)\)"

                def replacement(match):
                    args = match.group(1).strip()
                    if args:  # If there are other arguments
                        return f"def forward(self, {args})"
                    return "def forward(self)"

                return re.sub(pattern, replacement, source_code)

            # 第三步：把原函数改造成类的 forward 方法
            # - 函数名从原名（如 get_weather）改成 forward
            # - 加上 self 参数（原函数是普通函数，没有 self）
            # - 去掉 @tool 装饰器
            # - 缩进 4 格放进类体内
            forward_source_code = forward_source_code.replace(self.name, "forward")
            forward_source_code = add_self_argument(forward_source_code)
            forward_source_code = forward_source_code.replace("@tool", "").strip()
            tool_code += "\n\n" + textwrap.indent(forward_source_code, "    ")

        else:  # ===== 路径二：继承 Tool 类创建的工具 =====
            # 三种包装类依赖外部运行时对象，无法序列化
            if type(self).__name__ in [
                "SpaceToolWrapper",
                "LangChainToolWrapper",
                "GradioToolWrapper",
            ]:
                raise ValueError(
                    "Cannot save objects created with from_space, from_langchain or from_gradio, as this would create errors."
                )

            validate_tool_attributes(self.__class__)
            # 直接提取整个类的源码
            tool_code = "from typing import Any, Optional\n" + instance_to_source(self, base_cls=Tool)

        # 从生成的代码中提取第三方依赖（排除标准库），smolagents 始终作为依赖
        requirements = {el for el in get_imports(tool_code) if el not in sys.stdlib_module_names} | {"smolagents"}

        tool_dict = {"name": self.name, "code": tool_code, "requirements": sorted(requirements)}

        if hasattr(self, "output_schema") and self.output_schema is not None:
            tool_dict["output_schema"] = self.output_schema

        return tool_dict

    @classmethod
    def from_dict(cls, tool_dict: dict[str, Any], **kwargs) -> "Tool":
        """从字典反序列化出 Tool 实例（to_dict 的逆操作）。

        内部调用 from_code()，用 exec 执行字典中的源代码字符串，还原出 Tool 对象。

        Args:
            tool_dict: to_dict() 生成的字典，必须包含 "code" 键
            **kwargs: 传递给 Tool 子类构造函数的额外参数

        Returns:
            还原出的 Tool 实例
        """
        if "code" not in tool_dict:
            raise ValueError("Tool dictionary must contain 'code' key with the tool source code")

        tool = cls.from_code(tool_dict["code"], **kwargs)

        # 补上 output_schema（如果字典中有的话）
        if "output_schema" in tool_dict:
            tool.output_schema = tool_dict["output_schema"]

        return tool

    def save(self, output_dir: str | Path, tool_file_name: str = "tool", make_gradio_app: bool = True):
        """将工具保存到本地文件夹，生成可直接上传到 HuggingFace Space 的文件。

        生成的文件结构：
            output_dir/
            ├── {tool_file_name}.py   # 工具的完整类代码（来自 to_dict()["code"]）
            ├── app.py                # Gradio UI 启动脚本（make_gradio_app=True 时生成）
            └── requirements.txt      # 第三方依赖列表（make_gradio_app=True 时生成）

        与 push_to_hub() 的关系：两者生成的文件内容完全一样，
        save() 写到本地磁盘，push_to_hub() 直接上传到 Hub。

        Args:
            output_dir: 保存目录路径
            tool_file_name: 工具代码文件名（不含 .py 后缀）
            make_gradio_app: 是否同时生成 app.py 和 requirements.txt
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        # 保存工具源码
        self._write_file(output_path / f"{tool_file_name}.py", self._get_tool_code())
        if make_gradio_app:
            # 保存 Gradio UI 启动脚本
            self._write_file(output_path / "app.py", self._get_gradio_app_code(tool_module_name=tool_file_name))
            # 保存依赖列表
            self._write_file(output_path / "requirements.txt", self._get_requirements())

    def _write_file(self, file_path: Path, content: str) -> None:
        """写文件的封装方法，统一使用 UTF-8 编码。

        封装的好处：save() 里写三个文件时复用，子类也可以重写来改变写文件行为（如加日志）。
        """
        file_path.write_text(content, encoding="utf-8")

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload tool",
        private: bool | None = None,
        token: bool | str | None = None,
        create_pr: bool = False,
    ) -> str:
        """将工具上传到 HuggingFace Hub，作为一个 Gradio Space。

        save() 的线上版：生成的文件内容一样（tool.py + app.py + requirements.txt），
        但不写本地磁盘，直接上传到 Hub。上传后别人可以通过 Tool.from_hub(repo_id) 加载。

        三步流程：
        1. _initialize_hub_repo(): 在 Hub 上创建 Space 仓库，打上 smolagents 标签
        2. _prepare_hub_files():   生成三个文件的内容（tool.py/app.py/requirements.txt）
        3. create_commit():        提交文件到仓库（相当于 git add + commit + push）

        Parameters:
            repo_id: Hub 上的仓库 ID（如 "your_name/your_tool"）
            commit_message: 提交信息
            private: 是否私有仓库
            token: HuggingFace 认证 token
            create_pr: 是否创建 PR 而非直接提交
        """
        # 第一步：创建/初始化 Hub 仓库
        repo_id = self._initialize_hub_repo(repo_id, token, private)
        # 第二步：准备要上传的文件
        additions = self._prepare_hub_files()
        # 第三步：提交到 Hub
        return create_commit(
            repo_id=repo_id,
            operations=additions,
            commit_message=commit_message,
            token=token,
            create_pr=create_pr,
            repo_type="space",
        )

    @staticmethod
    def _initialize_hub_repo(repo_id: str, token: bool | str | None, private: bool | None) -> str:
        """在 HuggingFace Hub 上创建/初始化 Space 仓库，并打上 smolagents 标签。"""
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,       # 仓库已存在时不报错
            repo_type="space",   # 创建为 Space（而非 model 或 dataset）
            space_sdk="gradio",  # 使用 Gradio SDK
        )
        metadata_update(repo_url.repo_id, {"tags": ["smolagents", "tool"]}, repo_type="space", token=token)
        return repo_url.repo_id

    def _prepare_hub_files(self) -> list:
        """准备要上传到 Hub 的三个文件，返回 CommitOperationAdd 列表。"""
        additions = [
            CommitOperationAdd(
                path_in_repo="tool.py",
                path_or_fileobj=self._get_tool_code().encode(),
            ),
            CommitOperationAdd(
                path_in_repo="app.py",
                path_or_fileobj=self._get_gradio_app_code().encode(),
            ),
            CommitOperationAdd(
                path_in_repo="requirements.txt",
                path_or_fileobj=self._get_requirements().encode(),
            ),
        ]
        return additions

    def _get_tool_code(self) -> str:
        """获取工具的完整类源码（来自 to_dict()["code"]）。"""
        return self.to_dict()["code"]

    def _get_gradio_app_code(self, tool_module_name: str = "tool") -> str:
        """生成 Gradio UI 启动脚本代码，导入工具类并启动演示页面。"""
        class_name = self.__class__.__name__
        return textwrap.dedent(
            f"""\
            from smolagents import launch_gradio_demo
            from {tool_module_name} import {class_name}

            tool = {class_name}()
            launch_gradio_demo(tool)
            """
        )

    def _get_requirements(self) -> str:
        """获取工具的第三方依赖列表（每行一个包名）。"""
        return "\n".join(self.to_dict()["requirements"])

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        token: str | None = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Loads a tool defined on the Hub.

        <Tip warning={true}>

        Loading a tool from the Hub means that you'll download the tool and execute it locally.
        ALWAYS inspect the tool you're downloading before loading it within your runtime, as you would do when
        installing a package using pip/npm/apt.

        </Tip>

        Args:
            repo_id (`str`):
                The name of the Space repo on the Hub where your tool is defined.
            token (`str`, *optional*):
                The token to identify you on hf.co. If unset, will use the token generated when running
                `huggingface-cli login` (stored in `~/.huggingface`).
            trust_remote_code(`str`, *optional*, defaults to False):
                This flags marks that you understand the risk of running remote code and that you trust this tool.
                If not setting this to True, loading the tool from Hub will fail.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
                `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your tool, and the
                others will be passed along to its init.
        """
        if not trust_remote_code:
            raise ValueError(
                "Loading a tool from Hub requires to acknowledge you trust its code: to do so, pass `trust_remote_code=True`."
            )

        # Get the tool's tool.py file.
        tool_file = hf_hub_download(
            repo_id,
            "tool.py",
            token=token,
            repo_type="space",
            cache_dir=kwargs.get("cache_dir"),
            force_download=kwargs.get("force_download"),
            proxies=kwargs.get("proxies"),
            revision=kwargs.get("revision"),
            subfolder=kwargs.get("subfolder"),
            local_files_only=kwargs.get("local_files_only"),
        )

        tool_code = Path(tool_file).read_text()
        return Tool.from_code(tool_code, **kwargs)

    @classmethod
    def from_code(cls, tool_code: str, **kwargs):
        module = types.ModuleType("dynamic_tool")

        exec(tool_code, module.__dict__)

        # Find the Tool subclass
        tool_class = next(
            (
                obj
                for _, obj in inspect.getmembers(module, inspect.isclass)
                if issubclass(obj, Tool) and obj is not Tool
            ),
            None,
        )

        if tool_class is None:
            raise ValueError("No Tool subclass found in the code.")

        if not isinstance(tool_class.inputs, dict):
            tool_class.inputs = ast.literal_eval(tool_class.inputs)

        # Handle output_schema if it exists and is a string representation
        if hasattr(tool_class, "output_schema") and isinstance(tool_class.output_schema, str):
            tool_class.output_schema = ast.literal_eval(tool_class.output_schema)

        return tool_class(**kwargs)

    @staticmethod
    def from_space(
        space_id: str,
        name: str,
        description: str = "",
        api_name: str | None = None,
        token: str | None = None,
    ):
        """
        Creates a [`Tool`] from a Space given its id on the Hub.

        Args:
            space_id (`str`):
                The id of the Space on the Hub.
            name (`str`):
                The name of the tool.
            description (`str`):
                The description of the tool.
            api_name (`str`, *optional*):
                The specific api_name to use, if the space has several tabs. If not precised, will default to the first available api.
            token (`str`, *optional*):
                Add your token to access private spaces or increase your GPU quotas.
        Returns:
            [`Tool`]:
                The Space, as a tool.

        Examples:
        ```py
        >>> image_generator = Tool.from_space(
        ...     space_id="black-forest-labs/FLUX.1-schnell",
        ...     name="image-generator",
        ...     description="Generate an image from a prompt"
        ... )
        >>> image = image_generator("Generate an image of a cool surfer in Tahiti")
        ```
        ```py
        >>> face_swapper = Tool.from_space(
        ...     "tuan2308/face-swap",
        ...     "face_swapper",
        ...     "Tool that puts the face shown on the first image on the second image. You can give it paths to images.",
        ... )
        >>> image = face_swapper('./aymeric.jpeg', './ruth.jpg')
        ```
        """
        from gradio_client import Client, handle_file

        class SpaceToolWrapper(Tool):
            skip_forward_signature_validation = True

            def __init__(
                self,
                space_id: str,
                name: str,
                description: str = "",
                api_name: str | None = None,
                token: str | None = None,
            ):
                self.name = name
                self.description = description
                self.client = Client(space_id, hf_token=token)
                space_api = self.client.view_api(return_format="dict", print_info=False)
                assert isinstance(space_api, dict)
                space_description = space_api["named_endpoints"]

                # If api_name is not defined, take the first of the available APIs for this space
                if api_name is None:
                    api_name = list(space_description.keys())[0]
                    warnings.warn(
                        f"Since `api_name` was not defined, it was automatically set to the first available API: `{api_name}`."
                    )
                self.api_name = api_name

                try:
                    space_description_api = space_description[api_name]
                except KeyError:
                    raise KeyError(f"Could not find specified {api_name=} among available api names.")
                self.inputs = {}
                for parameter in space_description_api["parameters"]:
                    parameter_type = parameter["type"]["type"]
                    if parameter_type == "object":
                        parameter_type = "any"
                    self.inputs[parameter["parameter_name"]] = {
                        "type": parameter_type,
                        "description": parameter["python_type"]["description"],
                        "nullable": parameter["parameter_has_default"],
                    }
                output_component = space_description_api["returns"][0]["component"]
                if output_component == "Image":
                    self.output_type = "image"
                elif output_component == "Audio":
                    self.output_type = "audio"
                else:
                    self.output_type = "any"
                self.is_initialized = True

            def sanitize_argument_for_prediction(self, arg):
                from gradio_client.utils import is_http_url_like
                from PIL.Image import Image

                if isinstance(arg, Image):
                    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    arg.save(temp_file.name)
                    arg = temp_file.name
                if (
                    (isinstance(arg, str) and os.path.isfile(arg))
                    or (isinstance(arg, Path) and arg.exists() and arg.is_file())
                    or is_http_url_like(arg)
                ):
                    arg = handle_file(arg)
                return arg

            def forward(self, *args, **kwargs):
                # Preprocess args and kwargs:
                args = list(args)
                for i, arg in enumerate(args):
                    args[i] = self.sanitize_argument_for_prediction(arg)
                for arg_name, arg in kwargs.items():
                    kwargs[arg_name] = self.sanitize_argument_for_prediction(arg)

                output = self.client.predict(*args, api_name=self.api_name, **kwargs)
                if isinstance(output, tuple) or isinstance(output, list):
                    if isinstance(output[1], str):
                        raise ValueError("The space returned this message: " + output[1])
                    output = output[
                        0
                    ]  # Sometime the space also returns the generation seed, in which case the result is at index 0
                IMAGE_EXTENTIONS = [".png", ".jpg", ".jpeg", ".gif", ".webp"]
                AUDIO_EXTENTIONS = [".mp3", ".wav", ".ogg", ".m4a", ".flac"]
                if isinstance(output, str) and any([output.endswith(ext) for ext in IMAGE_EXTENTIONS]):
                    output = AgentImage(output)
                elif isinstance(output, str) and any([output.endswith(ext) for ext in AUDIO_EXTENTIONS]):
                    output = AgentAudio(output)
                return output

        return SpaceToolWrapper(
            space_id=space_id,
            name=name,
            description=description,
            api_name=api_name,
            token=token,
        )

    @staticmethod
    def from_gradio(gradio_tool):
        """
        Creates a [`Tool`] from a gradio tool.
        """
        import inspect

        class GradioToolWrapper(Tool):
            def __init__(self, _gradio_tool):
                self.name = _gradio_tool.name
                self.description = _gradio_tool.description
                self.output_type = "string"
                self._gradio_tool = _gradio_tool
                func_args = list(inspect.signature(_gradio_tool.run).parameters.items())
                self.inputs = {
                    key: {"type": CONVERSION_DICT[value.annotation], "description": ""} for key, value in func_args
                }
                self.forward = self._gradio_tool.run

        return GradioToolWrapper(gradio_tool)

    @staticmethod
    def from_langchain(langchain_tool):
        """
        Creates a [`Tool`] from a langchain tool.
        """

        class LangChainToolWrapper(Tool):
            skip_forward_signature_validation = True

            def __init__(self, _langchain_tool):
                self.name = _langchain_tool.name.lower()
                self.description = _langchain_tool.description
                self.inputs = _langchain_tool.args.copy()
                for input_content in self.inputs.values():
                    if "title" in input_content:
                        input_content.pop("title")
                    input_content["description"] = ""
                self.output_type = "string"
                self.langchain_tool = _langchain_tool
                self.is_initialized = True

            def forward(self, *args, **kwargs):
                tool_input = kwargs.copy()
                for index, argument in enumerate(args):
                    if index < len(self.inputs):
                        input_key = next(iter(self.inputs))
                        tool_input[input_key] = argument
                return self.langchain_tool.run(tool_input)

        return LangChainToolWrapper(langchain_tool)


def launch_gradio_demo(tool: Tool):
    """
    Launches a gradio demo for a tool. The corresponding tool class needs to properly implement the class attributes
    `inputs` and `output_type`.

    Args:
        tool (`Tool`): The tool for which to launch the demo.
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio should be installed in order to launch a gradio demo.")

    TYPE_TO_COMPONENT_CLASS_MAPPING = {
        "boolean": gr.Checkbox,
        "image": gr.Image,
        "audio": gr.Audio,
        "string": gr.Textbox,
        "integer": gr.Number,
        "number": gr.Number,
    }

    def tool_forward(*args, **kwargs):
        return tool(*args, sanitize_inputs_outputs=True, **kwargs)

    tool_forward.__signature__ = inspect.signature(tool.forward)

    gradio_inputs = []
    for input_name, input_details in tool.inputs.items():
        input_gradio_component_class = TYPE_TO_COMPONENT_CLASS_MAPPING[input_details["type"]]
        new_component = input_gradio_component_class(label=input_name)
        gradio_inputs.append(new_component)

    output_gradio_component_class = TYPE_TO_COMPONENT_CLASS_MAPPING[tool.output_type]
    gradio_output = output_gradio_component_class(label="Output")

    gr.Interface(
        fn=tool_forward,
        inputs=gradio_inputs,
        outputs=gradio_output,
        title=tool.name,
        description=tool.description,
        api_name=tool.name,
    ).launch()


def load_tool(
    repo_id,
    model_repo_id: str | None = None,
    token: str | None = None,
    trust_remote_code: bool = False,
    **kwargs,
):
    """
    Main function to quickly load a tool from the Hub.

    <Tip warning={true}>

    Loading a tool means that you'll download the tool and execute it locally.
    ALWAYS inspect the tool you're downloading before loading it within your runtime, as you would do when
    installing a package using pip/npm/apt.

    </Tip>

    Args:
        repo_id (`str`):
            Space repo ID of a tool on the Hub.
        model_repo_id (`str`, *optional*):
            Use this argument to use a different model than the default one for the tool you selected.
        token (`str`, *optional*):
            The token to identify you on hf.co. If unset, will use the token generated when running `huggingface-cli
            login` (stored in `~/.huggingface`).
        trust_remote_code (`bool`, *optional*, defaults to False):
            This needs to be accepted in order to load a tool from Hub.
        kwargs (additional keyword arguments, *optional*):
            Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
            `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your tool, and the others
            will be passed along to its init.
    """
    return Tool.from_hub(
        repo_id,
        model_repo_id=model_repo_id,
        token=token,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )


def add_description(description):
    """
    A decorator that adds a description to a function.
    """

    def inner(func):
        func.description = description
        func.name = func.__name__
        return func

    return inner


class ToolCollection:
    """
    Tool collections enable loading a collection of tools in the agent's toolbox.

    Collections can be loaded from a collection in the Hub or from an MCP server, see:
    - [`ToolCollection.from_hub`]
    - [`ToolCollection.from_mcp`]

    For example and usage, see: [`ToolCollection.from_hub`] and [`ToolCollection.from_mcp`]
    """

    def __init__(self, tools: list[Tool]):
        self.tools = tools

    @classmethod
    def from_hub(
        cls,
        collection_slug: str,
        token: str | None = None,
        trust_remote_code: bool = False,
    ) -> "ToolCollection":
        """Loads a tool collection from the Hub.

        it adds a collection of tools from all Spaces in the collection to the agent's toolbox

        > [!NOTE]
        > Only Spaces will be fetched, so you can feel free to add models and datasets to your collection if you'd
        > like for this collection to showcase them.

        Args:
            collection_slug (str): The collection slug referencing the collection.
            token (str, *optional*): The authentication token if the collection is private.
            trust_remote_code (bool, *optional*, defaults to False): Whether to trust the remote code.

        Returns:
            ToolCollection: A tool collection instance loaded with the tools.

        Example:
        ```py
        >>> from smolagents import ToolCollection, CodeAgent

        >>> image_tool_collection = ToolCollection.from_hub("huggingface-tools/diffusion-tools-6630bb19a942c2306a2cdb6f")
        >>> agent = CodeAgent(tools=[*image_tool_collection.tools], add_base_tools=True)

        >>> agent.run("Please draw me a picture of rivers and lakes.")
        ```
        """
        _collection = get_collection(collection_slug, token=token)
        _hub_repo_ids = {item.item_id for item in _collection.items if item.item_type == "space"}

        tools = [Tool.from_hub(repo_id, token, trust_remote_code) for repo_id in _hub_repo_ids]

        return cls(tools)

    @classmethod
    @contextmanager
    def from_mcp(
        cls,
        server_parameters: "mcp.StdioServerParameters" | dict,
        trust_remote_code: bool = False,
        structured_output: bool | None = None,
    ) -> "ToolCollection":
        """Automatically load a tool collection from an MCP server.

        This method supports Stdio, Streamable HTTP, and legacy HTTP+SSE MCP servers. Look at the `server_parameters`
        argument for more details on how to connect to each MCP server.

        Note: a separate thread will be spawned to run an asyncio event loop handling
        the MCP server.

        Args:
            server_parameters (`mcp.StdioServerParameters` or `dict`):
                Configuration parameters to connect to the MCP server. This can be:

                - An instance of `mcp.StdioServerParameters` for connecting a Stdio MCP server via standard input/output using a subprocess.

                - A `dict` with at least:
                  - "url": URL of the server.
                  - "transport": Transport protocol to use, one of:
                    - "streamable-http": Streamable HTTP transport (default).
                    - "sse": Legacy HTTP+SSE transport (deprecated).
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether to trust the execution of code from tools defined on the MCP server.
                This option should only be set to `True` if you trust the MCP server,
                and undertand the risks associated with running remote code on your local machine.
                If set to `False`, loading tools from MCP will fail.
            structured_output (`bool`, *optional*, defaults to `False`):
                Whether to enable structured output features for MCP tools. If True, enables:
                - Support for outputSchema in MCP tools
                - Structured content handling (structuredContent from MCP responses)
                - JSON parsing fallback for structured data
                If False, uses the original simple text-only behavior for backwards compatibility.

        Returns:
            ToolCollection: A tool collection instance.

        Example with a Stdio MCP server:
        ```py
        >>> import os
        >>> from smolagents import ToolCollection, CodeAgent, InferenceClientModel
        >>> from mcp import StdioServerParameters

        >>> model = InferenceClientModel()

        >>> server_parameters = StdioServerParameters(
        >>>     command="uvx",
        >>>     args=["--quiet", "pubmedmcp@0.1.3"],
        >>>     env={"UV_PYTHON": "3.12", **os.environ},
        >>> )

        >>> with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
        >>>     agent = CodeAgent(tools=[*tool_collection.tools], add_base_tools=True, model=model)
        >>>     agent.run("Please find a remedy for hangover.")
        ```

        Example with structured output enabled:
        ```py
        >>> with ToolCollection.from_mcp(server_parameters, trust_remote_code=True, structured_output=True) as tool_collection:
        >>>     agent = CodeAgent(tools=[*tool_collection.tools], add_base_tools=True, model=model)
        >>>     agent.run("Please find a remedy for hangover.")
        ```

        Example with a Streamable HTTP MCP server:
        ```py
        >>> with ToolCollection.from_mcp({"url": "http://127.0.0.1:8000/mcp", "transport": "streamable-http"}, trust_remote_code=True) as tool_collection:
        >>>     agent = CodeAgent(tools=[*tool_collection.tools], add_base_tools=True, model=model)
        >>>     agent.run("Please find a remedy for hangover.")
        ```
        """
        # Handle future warning for structured_output default value change
        if structured_output is None:
            warnings.warn(
                "Parameter 'structured_output' was not specified. "
                "Currently it defaults to False, but in version 1.25, the default will change to True. "
                "To suppress this warning, explicitly set structured_output=True (new behavior) or structured_output=False (legacy behavior). "
                "See documentation at https://huggingface.co/docs/smolagents/tutorials/tools#structured-output-and-output-schema-support for more details.",
                FutureWarning,
                stacklevel=2,
            )
            structured_output = False

        try:
            from mcpadapt.core import MCPAdapt
            from mcpadapt.smolagents_adapter import SmolAgentsAdapter
        except ImportError:
            raise ImportError(
                """Please install 'mcp' extra to use ToolCollection.from_mcp: `pip install 'smolagents[mcp]'`."""
            )
        if isinstance(server_parameters, dict):
            transport = server_parameters.get("transport")
            if transport is None:
                transport = "streamable-http"
                server_parameters["transport"] = transport
            if transport not in {"sse", "streamable-http"}:
                raise ValueError(
                    f"Unsupported transport: {transport}. Supported transports are 'streamable-http' and 'sse'."
                )
        if not trust_remote_code:
            raise ValueError(
                "Loading tools from MCP requires you to acknowledge you trust the MCP server, "
                "as it will execute code on your local machine: pass `trust_remote_code=True`."
            )
        with MCPAdapt(server_parameters, SmolAgentsAdapter(structured_output=structured_output)) as tools:
            yield cls(tools)


def tool(tool_function: Callable) -> Tool:
    """@tool 装饰器：将一个普通 Python 函数转换为 Tool 实例。

    实现原理：
    1. 调用 get_json_schema() 从函数的类型注解和文档字符串中提取元信息
    2. 动态创建一个 SimpleTool 子类，将函数的 name/description/inputs/output_type 作为类属性
    3. 将原函数绑定为 forward 方法
    4. 保留原函数源码（用于序列化 / push_to_hub）

    使用要求（缺一不可）：
    - 每个参数必须有类型注解（type hint）
    - 必须有返回值类型注解
    - docstring 中必须包含 "Args:" 段落描述每个参数（LLM 依赖这些描述理解工具）

    Args:
        tool_function (`Callable`): Function to convert into a Tool subclass.
            Should have type hints for each input and a type hint for the output.
            Should also have a docstring including the description of the function
            and an 'Args:' part where each argument is described.
    """
    tool_json_schema = get_json_schema(tool_function)["function"]
    if "return" not in tool_json_schema:
        if len(tool_json_schema["parameters"]["properties"]) == 0:
            tool_json_schema["return"] = {"type": "null"}
        else:
            raise TypeHintParsingException(
                "Tool return type not found: make sure your function has a return type hint!"
            )

    class SimpleTool(Tool):
        def __init__(self):
            self.is_initialized = True

    # Set the class attributes
    SimpleTool.name = tool_json_schema["name"]
    SimpleTool.description = tool_json_schema["description"]
    SimpleTool.inputs = tool_json_schema["parameters"]["properties"]
    SimpleTool.output_type = tool_json_schema["return"]["type"]

    # Set output_schema if it exists in the JSON schema
    if "output_schema" in tool_json_schema:
        SimpleTool.output_schema = tool_json_schema["output_schema"]
    elif "return" in tool_json_schema and "schema" in tool_json_schema["return"]:
        SimpleTool.output_schema = tool_json_schema["return"]["schema"]

    @wraps(tool_function)
    def wrapped_function(*args, **kwargs):
        return tool_function(*args, **kwargs)

    # Bind the copied function to the forward method
    SimpleTool.forward = staticmethod(wrapped_function)

    # Get the signature parameters of the tool function
    sig = inspect.signature(tool_function)
    # - Add "self" as first parameter to tool_function signature
    new_sig = sig.replace(
        parameters=[inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(sig.parameters.values())
    )
    # - Set the signature of the forward method
    SimpleTool.forward.__signature__ = new_sig

    # Create and attach the source code of the dynamically created tool class and forward method
    # - Get the source code of tool_function
    tool_source = textwrap.dedent(inspect.getsource(tool_function))
    # - Remove the tool decorator and function definition line
    lines = tool_source.splitlines()
    tree = ast.parse(tool_source)
    #   - Find function definition
    func_node = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)), None)
    if not func_node:
        raise ValueError(
            f"No function definition found in the provided source of {tool_function.__name__}. "
            "Ensure the input is a standard function."
        )
    #   - Extract decorator lines
    decorator_lines = ""
    if func_node.decorator_list:
        tool_decorators = [d for d in func_node.decorator_list if isinstance(d, ast.Name) and d.id == "tool"]
        if len(tool_decorators) > 1:
            raise ValueError(
                f"Multiple @tool decorators found on function '{func_node.name}'. Only one @tool decorator is allowed."
            )
        if len(tool_decorators) < len(func_node.decorator_list):
            warnings.warn(
                f"Function '{func_node.name}' has decorators other than @tool. "
                "This may cause issues with serialization in the remote executor. See issue #1626."
            )
        decorator_start = tool_decorators[0].end_lineno if tool_decorators else 0
        decorator_end = func_node.decorator_list[-1].end_lineno
        decorator_lines = "\n".join(lines[decorator_start:decorator_end])
    #   - Extract tool source body
    body_start = func_node.body[0].lineno - 1  # AST lineno starts at 1
    tool_source_body = "\n".join(lines[body_start:])
    # - Create the forward method source, including def line and indentation
    forward_method_source = f"def forward{new_sig}:\n{tool_source_body}"
    # - Create the class source
    indent = " " * 4  # for class method
    class_source = (
        textwrap.dedent(f"""
        class SimpleTool(Tool):
            name: str = "{tool_json_schema["name"]}"
            description: str = {json.dumps(textwrap.dedent(tool_json_schema["description"]).strip())}
            inputs: dict[str, dict[str, str]] = {tool_json_schema["parameters"]["properties"]}
            output_type: str = "{tool_json_schema["return"]["type"]}"

            def __init__(self):
                self.is_initialized = True

        """)
        + textwrap.indent(decorator_lines, indent)
        + textwrap.indent(forward_method_source, indent)
    )
    # - Store the source code on both class and method for inspection
    SimpleTool.__source__ = class_source
    SimpleTool.forward.__source__ = forward_method_source

    simple_tool = SimpleTool()
    return simple_tool


class PipelineTool(Tool):
    """
    A [`Tool`] tailored towards Transformer models. On top of the class attributes of the base class [`Tool`], you will
    need to specify:

    - **model_class** (`type`) -- The class to use to load the model in this tool.
    - **default_checkpoint** (`str`) -- The default checkpoint that should be used when the user doesn't specify one.
    - **pre_processor_class** (`type`, *optional*, defaults to [`transformers.AutoProcessor`]) -- The class to use to load the
      pre-processor
    - **post_processor_class** (`type`, *optional*, defaults to [`transformers.AutoProcessor`]) -- The class to use to load the
      post-processor (when different from the pre-processor).

    Args:
        model (`str` or [`transformers.PreTrainedModel`], *optional*):
            The name of the checkpoint to use for the model, or the instantiated model. If unset, will default to the
            value of the class attribute `default_checkpoint`.
        pre_processor (`str` or `Any`, *optional*):
            The name of the checkpoint to use for the pre-processor, or the instantiated pre-processor (can be a
            tokenizer, an image processor, a feature extractor or a processor). Will default to the value of `model` if
            unset.
        post_processor (`str` or `Any`, *optional*):
            The name of the checkpoint to use for the post-processor, or the instantiated pre-processor (can be a
            tokenizer, an image processor, a feature extractor or a processor). Will default to the `pre_processor` if
            unset.
        device (`int`, `str` or `torch.device`, *optional*):
            The device on which to execute the model. Will default to any accelerator available (GPU, MPS etc...), the
            CPU otherwise.
        device_map (`str` or `dict`, *optional*):
            If passed along, will be used to instantiate the model.
        model_kwargs (`dict`, *optional*):
            Any keyword argument to send to the model instantiation.
        token (`str`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated when
            running `huggingface-cli login` (stored in `~/.huggingface`).
        hub_kwargs (additional keyword arguments, *optional*):
            Any additional keyword argument to send to the methods that will load the data from the Hub.
    """

    pre_processor_class = None
    model_class = None
    post_processor_class = None
    default_checkpoint = None
    description = "This is a pipeline tool"
    name = "pipeline"
    inputs = {"prompt": str}
    output_type = str
    skip_forward_signature_validation = True

    def __init__(
        self,
        model=None,
        pre_processor=None,
        post_processor=None,
        device=None,
        device_map=None,
        model_kwargs=None,
        token=None,
        **hub_kwargs,
    ):
        if not _is_package_available("accelerate") or not _is_package_available("torch"):
            raise ModuleNotFoundError(
                "Please install 'transformers' extra to use a PipelineTool: `pip install 'smolagents[transformers]'`"
            )

        if model is None:
            if self.default_checkpoint is None:
                raise ValueError("This tool does not implement a default checkpoint, you need to pass one.")
            model = self.default_checkpoint
        if pre_processor is None:
            pre_processor = model

        self.model = model
        self.pre_processor = pre_processor
        self.post_processor = post_processor
        self.device = device
        self.device_map = device_map
        self.model_kwargs = {} if model_kwargs is None else model_kwargs
        if device_map is not None:
            self.model_kwargs["device_map"] = device_map
        self.hub_kwargs = hub_kwargs
        self.hub_kwargs["token"] = token

        super().__init__()

    def setup(self):
        """
        Instantiates the `pre_processor`, `model` and `post_processor` if necessary.
        """
        if isinstance(self.pre_processor, str):
            if self.pre_processor_class is None:
                from transformers import AutoProcessor

                self.pre_processor_class = AutoProcessor
            self.pre_processor = self.pre_processor_class.from_pretrained(self.pre_processor, **self.hub_kwargs)

        if isinstance(self.model, str):
            self.model = self.model_class.from_pretrained(self.model, **self.model_kwargs, **self.hub_kwargs)

        if self.post_processor is None:
            self.post_processor = self.pre_processor
        elif isinstance(self.post_processor, str):
            if self.post_processor_class is None:
                from transformers import AutoProcessor

                self.post_processor_class = AutoProcessor
            self.post_processor = self.post_processor_class.from_pretrained(self.post_processor, **self.hub_kwargs)

        if self.device is None:
            if self.device_map is not None:
                self.device = list(self.model.hf_device_map.values())[0]
            else:
                from accelerate import PartialState

                self.device = PartialState().default_device

        if self.device_map is None:
            self.model.to(self.device)

        super().setup()

    def encode(self, raw_inputs):
        """
        Uses the `pre_processor` to prepare the inputs for the `model`.
        """
        return self.pre_processor(raw_inputs)

    def forward(self, inputs):
        """
        Sends the inputs through the `model`.
        """
        import torch

        with torch.no_grad():
            return self.model(**inputs)

    def decode(self, outputs):
        """
        Uses the `post_processor` to decode the model output.
        """
        return self.post_processor(outputs)

    def __call__(self, *args, sanitize_inputs_outputs: bool = False, **kwargs):
        import torch
        from accelerate.utils import send_to_device

        if not self.is_initialized:
            self.setup()

        if sanitize_inputs_outputs:
            args, kwargs = handle_agent_input_types(*args, **kwargs)
        encoded_inputs = self.encode(*args, **kwargs)

        tensor_inputs = {k: v for k, v in encoded_inputs.items() if isinstance(v, torch.Tensor)}
        non_tensor_inputs = {k: v for k, v in encoded_inputs.items() if not isinstance(v, torch.Tensor)}

        encoded_inputs = send_to_device(tensor_inputs, self.device)
        outputs = self.forward({**encoded_inputs, **non_tensor_inputs})
        outputs = send_to_device(outputs, "cpu")
        decoded_outputs = self.decode(outputs)
        if sanitize_inputs_outputs:
            decoded_outputs = handle_agent_output_types(decoded_outputs, self.output_type)
        return decoded_outputs


def get_tools_definition_code(tools: dict[str, Tool]) -> str:
    tool_codes = []
    for tool in tools.values():
        validate_tool_attributes(tool.__class__, check_imports=False)
        tool_code = instance_to_source(tool, base_cls=Tool)
        tool_code = tool_code.replace("from smolagents.tools import Tool", "")
        tool_code += f"\n\n{tool.name} = {tool.__class__.__name__}()\n"
        tool_codes.append(tool_code)

    tool_definition_code = "\n".join([f"import {module}" for module in BASE_BUILTIN_MODULES])
    tool_definition_code += textwrap.dedent(
        """
    from typing import Any

    class Tool:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, *args, **kwargs):
            pass # to be implemented in child class
    """
    )
    tool_definition_code += "\n\n".join(tool_codes)
    return tool_definition_code


def validate_tool_arguments(tool: Tool, arguments: Any) -> None:
    """Validate tool arguments against tool's input schema.

    Checks that all provided arguments match the tool's expected input types and that
    all required arguments are present. Supports both dictionary arguments and single
    value arguments for tools with one input parameter.

    Args:
        tool (`Tool`): Tool whose input schema will be used for validation.
        arguments (`Any`): Arguments to validate. Can be a dictionary mapping
            argument names to values, or a single value for tools with one input.


    Raises:
        ValueError: If an argument is not in the tool's input schema, if a required
            argument is missing, or if the argument value doesn't match the expected type.
        TypeError: If an argument has an incorrect type that cannot be converted
            (e.g., string instead of number, excluding integer to number conversion).

    Note:
        - Supports type coercion from integer to number
        - Handles nullable parameters when explicitly marked in the schema
        - Accepts "any" type as a wildcard that matches all types
    """
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            if key not in tool.inputs:
                raise ValueError(f"Argument {key} is not in the tool's input schema")

            actual_type = _get_json_schema_type(type(value))["type"]
            expected_type = tool.inputs[key]["type"]
            expected_type_is_nullable = tool.inputs[key].get("nullable", False)

            # Type is valid if it matches, is "any", or is null for nullable parameters
            if (
                (actual_type != expected_type if isinstance(expected_type, str) else actual_type not in expected_type)
                and expected_type != "any"
                and not (actual_type == "null" and expected_type_is_nullable)
            ):
                if actual_type == "integer" and expected_type == "number":
                    continue
                raise TypeError(f"Argument {key} has type '{actual_type}' but should be '{tool.inputs[key]['type']}'")

        for key, schema in tool.inputs.items():
            key_is_nullable = schema.get("nullable", False)
            if key not in arguments and not key_is_nullable:
                raise ValueError(f"Argument {key} is required")
        return None
    else:
        expected_type = list(tool.inputs.values())[0]["type"]
        if _get_json_schema_type(type(arguments))["type"] != expected_type and not expected_type == "any":
            raise TypeError(f"Argument has type '{type(arguments).__name__}' but should be '{expected_type}'")


__all__ = [
    "AUTHORIZED_TYPES",
    "Tool",
    "tool",
    "load_tool",
    "launch_gradio_demo",
    "ToolCollection",
]
