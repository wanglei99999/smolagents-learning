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
# agents.py —— smolagents 框架的核心文件（Agent 逻辑）
#
# 本文件定义了三个核心类：
#   1. MultiStepAgent（抽象基类）
#      - 实现 ReAct 框架的通用逻辑：多步循环、记忆管理、规划、保存/加载等
#      - 所有 Agent 的公共父类
#
#   2. ToolCallingAgent（继承 MultiStepAgent）
#      - LLM 以"JSON 格式"调用工具（OpenAI function calling 风格）
#      - LLM 输出: {"tool": "get_weather", "arguments": {"location": "Paris"}}
#      - 适合：简单的单工具调用场景
#
#   3. CodeAgent（继承 MultiStepAgent）
#      - LLM 生成"Python 代码"并在沙箱中执行
#      - LLM 输出: result = get_weather(location="Paris"); final_answer(result)
#      - 适合：多步推理、条件判断、组合多个工具的复杂任务
#
# ReAct 循环（每步执行）：
#   思考（Thought）→ 行动（Action / Code）→ 观察（Observation）→ 循环直到 final_answer
# =============================================================================

import importlib
import json
import os
import tempfile
import textwrap
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Type, TypeAlias, TypedDict, Union

import yaml
from huggingface_hub import create_repo, metadata_update, snapshot_download, upload_folder
from jinja2 import StrictUndefined, Template
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text


if TYPE_CHECKING:
    import PIL.Image

from .agent_types import AgentAudio, AgentImage, handle_agent_output_types
from .default_tools import TOOL_MAPPING, FinalAnswerTool
from .local_python_executor import BASE_BUILTIN_MODULES, LocalPythonExecutor, PythonExecutor, fix_final_answer_code
from .memory import (
    ActionStep,
    AgentMemory,
    CallbackRegistry,
    FinalAnswerStep,
    MemoryStep,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
    Timing,
    ToolCall,
)
from .models import (
    CODEAGENT_RESPONSE_FORMAT,
    MODEL_REGISTRY,
    ChatMessage,
    ChatMessageStreamDelta,
    ChatMessageToolCall,
    MessageRole,
    Model,
    agglomerate_stream_deltas,
    parse_json_if_needed,
)
from .monitoring import (
    YELLOW_HEX,
    AgentLogger,
    LogLevel,
    Monitor,
    TokenUsage,
)
from .remote_executors import BlaxelExecutor, DockerExecutor, E2BExecutor, ModalExecutor, WasmExecutor
from .tools import BaseTool, Tool, validate_tool_arguments
from .utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    AgentToolCallError,
    AgentToolExecutionError,
    create_agent_gradio_app_template,
    extract_code_from_text,
    is_valid_name,
    make_init_file,
    parse_code_blobs,
    truncate_content,
)


logger = getLogger(__name__)


def populate_template(template: str, variables: dict[str, Any]) -> str:
    """使用 Jinja2 渲染提示词模板，将变量填入模板字符串中。
    StrictUndefined：若模板中引用了未定义的变量，立即抛出异常（便于排查提示词错误）。
    """
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")


# -----------------------------------------------------------------------------
# 内部数据结构：用于在流式生成过程中传递中间结果
# -----------------------------------------------------------------------------

@dataclass
class ActionOutput:
    """一步 ReAct 循环的输出结果。
    output: 这一步的输出值（代码执行结果或工具调用结果）
    is_final_answer: 是否是最终答案（True 则终止循环）
    """
    output: Any
    is_final_answer: bool


@dataclass
class ToolOutput:
    """ToolCallingAgent 单次工具调用的完整输出。
    id: 工具调用的唯一 ID（与 LLM 输出中的 tool_call_id 对应）
    output: 工具返回值
    is_final_answer: 是否是 final_answer 工具（True 则终止循环）
    observation: 格式化后的观察结果字符串（写入记忆）
    tool_call: 对应的 ToolCall 请求对象
    """
    id: str
    output: Any
    is_final_answer: bool
    observation: str
    tool_call: ToolCall


class PlanningPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.

    Args:
        plan (`str`): Initial plan prompt.
        update_plan_pre_messages (`str`): Update plan pre-messages prompt.
        update_plan_post_messages (`str`): Update plan post-messages prompt.
    """

    initial_plan: str
    update_plan_pre_messages: str
    update_plan_post_messages: str


class ManagedAgentPromptTemplate(TypedDict):
    """
    Prompt templates for the managed agent.

    Args:
        task (`str`): Task prompt.
        report (`str`): Report prompt.
    """

    task: str
    report: str


class FinalAnswerPromptTemplate(TypedDict):
    """
    Prompt templates for the final answer.

    Args:
        pre_messages (`str`): Pre-messages prompt.
        post_messages (`str`): Post-messages prompt.
    """

    pre_messages: str
    post_messages: str


class PromptTemplates(TypedDict):
    """
    Prompt templates for the agent.

    Args:
        system_prompt (`str`): System prompt.
        planning ([`~agents.PlanningPromptTemplate`]): Planning prompt templates.
        managed_agent ([`~agents.ManagedAgentPromptTemplate`]): Managed agent prompt templates.
        final_answer ([`~agents.FinalAnswerPromptTemplate`]): Final answer prompt templates.
    """

    system_prompt: str
    planning: PlanningPromptTemplate
    managed_agent: ManagedAgentPromptTemplate
    final_answer: FinalAnswerPromptTemplate


# 空提示词模板：当用户未提供自定义提示词时的默认占位值
# 实际运行时会被 code_agent.yaml / toolcalling_agent.yaml 中的内容替换
EMPTY_PROMPT_TEMPLATES = PromptTemplates(
    system_prompt="",
    planning=PlanningPromptTemplate(
        initial_plan="",
        update_plan_pre_messages="",
        update_plan_post_messages="",
    ),
    managed_agent=ManagedAgentPromptTemplate(task="", report=""),
    final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
)


@dataclass
class RunResult:
    # RunResult 可以理解成“整次 Agent 运行的汇总结果对象”。
    # 它不是某一步 step 的日志，也不是底层 executor 的 CodeOutput，
    # 而是 Agent.run(..., return_full_result=True) 时返回给调用方的高层结果快照。
    #
    # 里面会同时放：
    # - output: 最终答案
    # - state: 这次 run 的收尾状态（成功 / 达到最大步数）
    # - steps: 整次运行过程积累下来的 memory steps
    # - token_usage: 整次 run 的 token 统计
    # - timing: 整次 run 的时间统计
    """Holds extended information about an agent run.

    Attributes:
        output (Any | None): The final output of the agent run, if available.
        state (Literal["success", "max_steps_error"]): The final state of the agent after the run.
        steps (list[dict]): The agent's memory, as a list of steps.
        token_usage (TokenUsage | None): Count of tokens used during the run.
        timing (Timing): Timing details of the agent run: start time, end time, duration.
        messages (list[dict]): The agent's memory, as a list of messages.
            <Deprecated version="1.22.0">
            Parameter 'messages' is deprecated and will be removed in version 1.25. Please use 'steps' instead.
            </Deprecated>
    """

    output: Any | None
    state: Literal["success", "max_steps_error"]
    steps: list[dict]
    token_usage: TokenUsage | None
    timing: Timing

    def __init__(self, output=None, state=None, steps=None, token_usage=None, timing=None, messages=None):
        # Handle deprecated 'messages' parameter
        if messages is not None:
            if steps is not None:
                raise ValueError("Cannot specify both 'messages' and 'steps' parameters. Use 'steps' instead.")
            warnings.warn(
                "Parameter 'messages' is deprecated and will be removed in version 1.25. Please use 'steps' instead.",
                FutureWarning,
                stacklevel=2,
            )
            steps = messages

        # Initialize with dataclass fields
        self.output = output
        self.state = state
        self.steps = steps
        self.token_usage = token_usage
        self.timing = timing

    @property
    def messages(self):
        """Backward compatibility property that returns steps."""
        warnings.warn(
            "Parameter 'messages' is deprecated and will be removed in version 1.25. Please use 'steps' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.steps

    def dict(self):
        return {
            "output": self.output,
            "state": self.state,
            "steps": self.steps,
            "token_usage": self.token_usage.dict() if self.token_usage is not None else None,
            "timing": self.timing.dict(),
        }


# 流式事件类型别名：在流式模式下 _run_stream 生成器可能 yield 的所有事件类型
# ChatMessageStreamDelta：LLM 流式输出的增量 token
# ActionOutput：一步完整的行动输出（含 is_final_answer 标志）
# ActionStep / PlanningStep / FinalAnswerStep：写入记忆后的步骤对象
StreamEvent: TypeAlias = Union[
    ChatMessageStreamDelta,
    ChatMessageToolCall,
    ActionOutput,
    ToolCall,
    ToolOutput,
    PlanningStep,
    ActionStep,
    FinalAnswerStep,
]


class MultiStepAgent(ABC):
    # MultiStepAgent 是整个 Agent 体系的“骨架基类”。
    # 可以把它理解成：
    #   - ReAct 多步循环的通用框架
    #   - memory / monitor / logger / tools / managed_agents 的总组织者
    #   - CodeAgent / ToolCallingAgent 这类具体 Agent 的共同父类
    #
    # 它本身不决定“动作到底长什么样”：
    #   - CodeAgent: 动作是 Python 代码
    #   - ToolCallingAgent: 动作是结构化工具调用
    #
    # 但它负责定义这类 Agent 的共同运行模式：
    #   任务输入 -> 多轮 step 循环 -> 每轮思考/行动/观察 -> memory 累积 -> 最终答案
    """
    Agent class that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of action (given by the LLM) and observation (obtained from the environment).

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        instructions (`str`, *optional*): Custom instructions for the agent, will be inserted in the system prompt.
        max_steps (`int`, default `20`): Maximum number of steps the agent can take to solve the task.
        add_base_tools (`bool`, default `False`): Whether to add the base tools to the agent's tools.
        verbosity_level (`LogLevel`, default `LogLevel.INFO`): Level of verbosity of the agent's logs.
        managed_agents (`list`, *optional*): Managed agents that the agent can call.
        step_callbacks (`list[Callable]` | `dict[Type[MemoryStep], Callable | list[Callable]]`, *optional*): Callbacks that will be called at each step.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        name (`str`, *optional*): Necessary for a managed agent only - the name by which this agent can be called.
        description (`str`, *optional*): Necessary for a managed agent only - the description of this agent.
        provide_run_summary (`bool`, *optional*): Whether to provide a run summary when called as a managed agent.
        final_answer_checks (`list[Callable]`, *optional*): List of validation functions to run before accepting a final answer.
            Each function should:
            - Take the final answer, the agent's memory, and the agent itself as arguments.
            - Return a boolean indicating whether the final answer is valid.
        return_full_result (`bool`, default `False`): Whether to return the full [`RunResult`] object or just the final answer output from the agent run.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        prompt_templates: PromptTemplates | None = None,
        instructions: str | None = None,
        max_steps: int = 20,
        add_base_tools: bool = False,
        verbosity_level: LogLevel = LogLevel.INFO,
        managed_agents: list | None = None,
        step_callbacks: list[Callable] | dict[Type[MemoryStep], Callable | list[Callable]] | None = None,
        planning_interval: int | None = None,
        name: str | None = None,
        description: str | None = None,
        provide_run_summary: bool = False,
        final_answer_checks: list[Callable] | None = None,
        return_full_result: bool = False,
        logger: AgentLogger | None = None,
    ):
        """初始化 MultiStepAgent。
        
        这个构造函数负责：
        1. 验证和设置提示词模板
        2. 初始化工具和子 Agent
        3. 设置记忆、日志、监控系统
        4. 注册回调函数
        
        Args:
            tools: Agent 可用的工具列表
            model: 用于生成 Agent 行动的 LLM 模型
            prompt_templates: 提示词模板（系统提示词、规划、最终答案等）
            instructions: 自定义指令，会插入到系统提示词中
            max_steps: ReAct 循环的最大步数（防止无限循环）
            add_base_tools: 是否添加基础工具（如 web_search, python_interpreter 等）
            verbosity_level: 日志详细程度
            managed_agents: 子 Agent 列表（可以像工具一样被调用）
            step_callbacks: 每步执行后的回调函数
            planning_interval: 规划步骤的间隔（None 表示不规划）
            name: Agent 名称（作为子 Agent 时必填）
            description: Agent 描述（作为子 Agent 时必填）
            provide_run_summary: 作为子 Agent 时是否提供执行摘要
            final_answer_checks: 最终答案的验证函数列表
            return_full_result: 是否返回完整的 RunResult 对象
            logger: 自定义日志记录器
        """
        self.agent_name = self.__class__.__name__
        self.model = model
        
        # 校验自定义提示词模板，确保所有必需的键都存在
        self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
        if prompt_templates is not None:
            # 检查顶层键（system_prompt, planning, managed_agent, final_answer）
            missing_keys = set(EMPTY_PROMPT_TEMPLATES.keys()) - set(prompt_templates.keys())
            assert not missing_keys, (
                f"Some prompt templates are missing from your custom `prompt_templates`: {missing_keys}"
            )
            # 检查嵌套键（如 planning.initial_plan, planning.update_plan_pre_messages 等）
            for key, value in EMPTY_PROMPT_TEMPLATES.items():
                if isinstance(value, dict):
                    for subkey in value.keys():
                        assert key in prompt_templates.keys() and (subkey in prompt_templates[key].keys()), (
                            f"Some prompt templates are missing from your custom `prompt_templates`: {subkey} under {key}"
                        )

        self.max_steps = max_steps          # ReAct 循环的最大步数，防止无限循环
        self.step_number = 0                 # 当前步骤编号，每次 run() 会重置
        self.planning_interval = planning_interval  # 每隔多少步触发一次规划，None 表示不规划
        self.state: dict[str, Any] = {}     # Agent 的全局状态字典，用于在步骤间传递变量（如图片、数据框）
        self.name = self._validate_name(name)       # 作为 managed_agent 时必填，用于主 Agent 识别
        self.description = description               # 作为 managed_agent 时必填，告知主 Agent 何时调用
        self.provide_run_summary = provide_run_summary  # 作为 managed_agent 时，是否在返回结果中附带执行摘要
        self.final_answer_checks = final_answer_checks if final_answer_checks is not None else []
        self.return_full_result = return_full_result  # True: 返回 RunResult 对象; False: 只返回最终答案
        self.instructions = instructions              # 插入系统提示词的自定义指令
        
        # 初始化子 Agent 和工具
        self._setup_managed_agents(managed_agents)   # 初始化子 Agent 字典
        self._setup_tools(tools, add_base_tools)     # 初始化工具字典，并自动添加 final_answer 工具
        self._validate_tools_and_managed_agents(tools, managed_agents)  # 检查名称唯一性

        self.task: str | None = None
        self.memory = AgentMemory(self.system_prompt)  # 初始化记忆，以当前系统提示词为起点

        # 初始化日志和监控系统
        if logger is None:
            self.logger = AgentLogger(level=verbosity_level)
        else:
            self.logger = logger

        self.monitor = Monitor(self.model, self.logger)  # 监控 Token 用量
        self._setup_step_callbacks(step_callbacks)       # 注册步骤回调函数
        self.stream_outputs = False  # 子类会根据自身情况覆盖此值

    #property类似于getter方法
    @property
    def system_prompt(self) -> str:
        return self.initialize_system_prompt()

    @system_prompt.setter
    def system_prompt(self, value: str):
        raise AttributeError(
            """The 'system_prompt' property is read-only. Use 'self.prompt_templates["system_prompt"]' instead."""
        )

    def _validate_name(self, name: str | None) -> str | None:
        if name is not None and not is_valid_name(name):
            raise ValueError(f"Agent name '{name}' must be a valid Python identifier and not a reserved keyword.")
        return name

    def _setup_managed_agents(self, managed_agents: list | None = None) -> None:
        """初始化子 Agent 字典。
        将子 Agent 包装成"工具"的接口形式（统一 inputs/output_type），
        使主 Agent 可以像调用工具一样调用子 Agent。
        """
        self.managed_agents = {}
        if managed_agents:
            assert all(agent.name and agent.description for agent in managed_agents), (
                "All managed agents need both a name and a description!"
            )
            #构建名称映射，创建{agent名称：agent对象}
            self.managed_agents = {agent.name: agent for agent in managed_agents}
            # 给每个子 Agent 统一设置"工具接口"，让 LLM 可以像调用工具一样描述对子 Agent 的调用
            for agent in self.managed_agents.values():
                #输入格式
                agent.inputs = {
                    #要执行的任务描述（字符串）
                    "task": {
                        "type": "string", 
                        "description": "Long detailed description of the task."
                        },
                    #额外的上下文数据（可选的字典对象）
                    "additional_args": {
                        "type": "object",
                        "description": "Dictionary of extra inputs to pass to the managed agent, e.g. images, dataframes, or any other contextual data it may need.",
                        "nullable": True,
                    },
                }
                #输出格式：固定字符串
                agent.output_type = "string"

    def _setup_tools(self, tools, add_base_tools):
        """初始化工具字典。
        注意：final_answer 工具是强制内置的（Agent 通过它返回最终答案），
        python_interpreter 工具只为 ToolCallingAgent 添加（CodeAgent 直接执行代码，不需要它）。
        """
        assert all(isinstance(tool, BaseTool) for tool in tools), (
            "All elements must be instance of BaseTool (or a subclass)"
        )
        self.tools = {tool.name: tool for tool in tools}
        if add_base_tools:
            self.tools.update(
                {
                    name: cls()
                    for name, cls in TOOL_MAPPING.items()
                    if name != "python_interpreter" or self.__class__.__name__ == "ToolCallingAgent"
                }
            )
        # final_answer 是所有 Agent 的内置工具，LLM 调用它时循环终止并返回答案
        self.tools.setdefault("final_answer", FinalAnswerTool())

    def _validate_tools_and_managed_agents(self, tools, managed_agents):
        #名字不能重复，不然模型调用会出问题
        tool_and_managed_agent_names = [tool.name for tool in tools]
        if managed_agents is not None:
            tool_and_managed_agent_names += [agent.name for agent in managed_agents]
        if self.name:
            tool_and_managed_agent_names.append(self.name)
        #有点意思，通过初始长度和set去重长度比较，判断是否有重复
        if len(tool_and_managed_agent_names) != len(set(tool_and_managed_agent_names)):
            raise ValueError(
                "Each tool or managed_agent should have a unique name! You passed these duplicate names: "
                f"{[name for name in tool_and_managed_agent_names if tool_and_managed_agent_names.count(name) > 1]}"
            )

    def _setup_step_callbacks(self, step_callbacks):
        """
        初始化步骤回调函数系统
        
        回调函数允许外部代码监听和响应 Agent 执行过程中的各种步骤，
        实现日志记录、性能监控、调试等功能。
        
        Args:
            step_callbacks: 回调函数配置，支持两种格式：
                - list: 回调函数列表，默认注册到 ActionStep（向后兼容）
                - dict: {步骤类型: 回调函数} 的映射，支持精确控制
        """
        # 初始化步骤回调注册表：管理所有回调函数的中央注册表
        self.step_callbacks = CallbackRegistry()
        
        if step_callbacks:
            # 处理列表形式的回调配置（向后兼容旧版本 API）
            if isinstance(step_callbacks, list):
                # 将所有回调函数注册到 ActionStep 类型
                # ActionStep 是最常见的步骤类型，包含工具调用等核心操作
                for callback in step_callbacks:
                    self.step_callbacks.register(ActionStep, callback)
            
            # 处理字典形式的回调配置（新版本功能，支持精确控制）
            elif isinstance(step_callbacks, dict):
                # 遍历每个步骤类型及其对应的回调函数
                for step_cls, callbacks in step_callbacks.items():
                    # 确保回调函数是列表格式（统一处理单个函数和函数列表）
                    if not isinstance(callbacks, list):
                        callbacks = [callbacks]
                    
                    # 将每个回调函数注册到指定的步骤类型
                    for callback in callbacks:
                        self.step_callbacks.register(step_cls, callback)
            else:
                # 参数类型错误，抛出异常提示正确用法
                raise ValueError("step_callbacks must be a list or a dict")
        
        # 自动注册内置的性能监控回调（向后兼容，只监控 ActionStep）
        # self.monitor.update_metrics 会在每个 ActionStep 后更新性能指标
        self.step_callbacks.register(ActionStep, self.monitor.update_metrics)

    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: list["PIL.Image.Image"] | None = None,
        additional_args: dict | None = None,
        max_steps: int | None = None,
        return_full_result: bool | None = None,
    ) -> Any | RunResult:
        """
        运行 Agent 执行任务 - 这是 Agent 的主入口方法
        
        这个方法是用户与 Agent 交互的主要接口，负责：
        1. 初始化执行环境和状态
        2. 处理输入参数（任务、图片、额外数据等）
        3. 选择执行模式（流式 vs 非流式）
        4. 调用核心执行逻辑 _run_stream()
        5. 处理和返回执行结果
        
        Args:
            task (`str`): 要执行的任务描述
                例如："帮我计算 25 * 4 + 10" 或 "分析这张图片的内容"
            stream (`bool`): 是否使用流式输出模式
                - True: 返回生成器，可以实时获取每个执行步骤（适合 UI 显示进度）
                - False: 内部执行完所有步骤后只返回最终结果（适合批处理）
            reset (`bool`): 是否重置 Agent 状态
                - True: 清空对话历史，开始全新会话
                - False: 保持之前的对话上下文，继续对话
            images (`list[PIL.Image.Image]`, *optional*): 可选的图片输入
                支持多模态任务，如图片分析、OCR 等
            additional_args (`dict`, *optional*): 额外的上下文数据
                可以传入数据框、变量等，Agent 可以在执行中直接访问
                例如：{"df": pandas_dataframe, "config": settings}
            max_steps (`int`, *optional*): 最大执行步数限制
                防止无限循环，如果不提供则使用 Agent 的默认值
            return_full_result (`bool`, *optional*): 是否返回完整结果对象
                - True: 返回 RunResult 对象（包含步骤详情、token 使用量、时间等）
                - False: 只返回最终答案
                - None: 使用 Agent 的默认设置

        Returns:
            Any | RunResult: 根据 return_full_result 参数决定返回格式
            - 简单模式：直接返回最终答案（字符串、数字等）
            - 完整模式：返回 RunResult 对象，包含详细的执行信息

        Example:
        ```py
        from smolagents import CodeAgent
        
        # 基本使用
        agent = CodeAgent(tools=[])
        result = agent.run("What is the result of 2 power 3.7384?")
        
        # 流式使用（实时获取进度）
        for step in agent.run("复杂任务", stream=True):
            print(f"执行步骤: {step}")
        
        # 多模态使用
        from PIL import Image
        image = Image.open("chart.png")
        result = agent.run("分析这个图表", images=[image])
        
        # 传入额外数据
        import pandas as pd
        df = pd.read_csv("data.csv")
        result = agent.run("分析数据并生成报告", additional_args={"data": df})
        ```
        """
        # === 第1步：初始化执行参数 ===
        # 使用传入的 max_steps 或 Agent 的默认最大步数
        max_steps = max_steps or self.max_steps
        self.task = task  # 保存当前任务到实例变量
        
        # 初始化中断开关：允许外部调用 agent.interrupt() 来优雅停止执行
        # 这对长时间运行的任务很有用，可以避免强制终止
        self.interrupt_switch = False
        
        # === 第2步：处理额外参数和上下文数据 ===
        # 将用户提供的额外数据（如数据框、图片、配置等）注入到 Agent 的全局状态中
        # 这些变量可以在后续的工具调用或代码执行中直接访问
        if additional_args:
            self.state.update(additional_args)
            # 在任务描述中明确告知 LLM 这些变量的存在和名称
            # 这样 LLM 就知道可以直接使用这些变量，无需重新定义
            self.task += f"""
You have been provided with these additional arguments, that you can access directly using the keys as variables:
{str(additional_args)}."""

        # === 第3步：更新系统提示词和重置状态 ===
        # 重新生成系统提示词（可能因为工具列表变化、配置更新等原因需要更新）
        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        
        if reset:
            # 重置记忆：清空之前的对话历史，开始全新的会话
            # 重置监控器：清空性能统计数据，重新开始计时
            self.memory.reset()
            self.monitor.reset()

        # === 第4步：记录任务开始和初始化记忆 ===
        # 记录任务开始的日志，包含模型信息和任务内容
        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )
        
        # 将用户任务作为第一个步骤写入记忆系统
        # TaskStep 包含任务描述和可选的图片，是整个执行流程的起点
        self.memory.steps.append(TaskStep(task=self.task, task_images=images))

        # === 第5步：CodeAgent 特殊处理 ===
        # 如果当前是 CodeAgent（支持 Python 代码执行），需要将状态和工具发送到执行器
        # 这样在代码执行时就能访问这些变量和调用工具函数
        if getattr(self, "python_executor", None):
            # 发送状态变量：让 Python 执行环境能访问 self.state 中的所有变量
            self.python_executor.send_variables(variables=self.state)
            # 发送工具和子 Agent：让 Python 代码能直接调用工具函数
            self.python_executor.send_tools({**self.tools, **self.managed_agents})

        # === 第6步：选择执行模式 ===
        if stream:
            # 流式模式：直接返回生成器，调用方需要通过迭代来获取每个步骤
            # 适用场景：
            # - UI 界面需要实时显示执行进度
            # - 需要在执行过程中进行干预或监控
            # - 处理长时间运行的任务时提供用户反馈
            return self._run_stream(task=self.task, max_steps=max_steps, images=images)

        # === 第7步：非流式模式执行 ===
        run_start_time = time.time()  # 记录开始时间，用于性能统计
        
        # 非流式模式：内部完全消费生成器，只返回最终结果
        # 将生成器转换为列表，这会触发完整的执行流程
        # 适用场景：
        # - 批处理任务，只关心最终结果
        # - 简单的脚本调用，不需要中间过程
        # - 自动化流程中的一个环节
        steps = list(self._run_stream(task=self.task, max_steps=max_steps, images=images))

        # === 第8步：提取最终结果 ===
        # 生成器的最后一个元素必须是 FinalAnswerStep（包含最终答案）
        # 这是 Agent 执行流程的设计保证
        assert isinstance(steps[-1], FinalAnswerStep)
        output = steps[-1].output  # 提取最终答案内容

        # === 第9步：决定返回格式 ===
        # 根据参数决定返回简单答案还是完整结果对象
        return_full_result = return_full_result if return_full_result is not None else self.return_full_result
        
        if return_full_result:
            # === 完整结果模式：构建详细的 RunResult 对象 ===
            
            # 统计所有步骤的 token 使用量（用于成本分析和性能监控）
            total_input_tokens = 0
            total_output_tokens = 0
            correct_token_usage = True  # 标记是否所有步骤都有完整的 token 统计
            
            # 遍历记忆中的所有步骤，累计 token 使用量
            for step in self.memory.steps:
                # 只统计会调用 LLM 的步骤（ActionStep 和 PlanningStep）
                if isinstance(step, (ActionStep, PlanningStep)):
                    if step.token_usage is None:
                        # 如果任何步骤缺少 token 统计，则放弃整体统计
                        # 这可能发生在某些模型不支持 token 计数的情况下
                        correct_token_usage = False
                        break
                    else:
                        total_input_tokens += step.token_usage.input_tokens
                        total_output_tokens += step.token_usage.output_tokens
            
            # 根据统计结果创建 TokenUsage 对象
            if correct_token_usage:
                token_usage = TokenUsage(input_tokens=total_input_tokens, output_tokens=total_output_tokens)
            else:
                token_usage = None

            # === 判断执行状态 ===
            # 检查是否因为达到最大步数限制而结束
            if self.memory.steps and isinstance(getattr(self.memory.steps[-1], "error", None), AgentMaxStepsError):
                state = "max_steps_error"  # 达到最大步数限制
            else:
                state = "success"  # 正常完成

            # === 准备步骤数据 ===
            # 将内存中的步骤对象转换为字典格式，便于序列化和存储
            step_dicts = self.memory.get_full_steps()

            # === 返回完整的运行结果对象 ===
            return RunResult(
                output=output,                    # 最终答案
                token_usage=token_usage,          # token 使用统计
                steps=step_dicts,                 # 所有执行步骤的详细信息
                timing=Timing(                    # 执行时间统计
                    start_time=run_start_time, 
                    end_time=time.time()
                ),
                state=state,                      # 执行状态（成功/超时等）
            )

        # === 简单模式：只返回最终答案 ===
        # 大多数情况下用户只关心结果，不需要执行细节
        # 这是默认行为，保持 API 的简洁性
        return output

    def _run_stream(
        self, task: str, max_steps: int, images: list["PIL.Image.Image"] | None = None
    ) -> Generator[ActionStep | PlanningStep | FinalAnswerStep | ChatMessageStreamDelta]:
        """ReAct 循环的核心生成器。
        
        这是 Agent 执行的心脏：不断循环执行"思考→行动→观察"，直到任务完成。
        
        工作流程：
        1. 检查是否需要规划（第1步 或 每隔 planning_interval 步）
        2. 执行一个行动步骤（调用 LLM → 解析输出 → 执行工具/代码）
        3. 将结果写入记忆
        4. 检查是否得到最终答案，否则继续下一步
        
        终止条件：
        - LLM 调用了 final_answer 工具/函数
        - 达到 max_steps 上限（会强制生成最终答案）
        - 外部调用 agent.interrupt() 中断
        
        Args:
            task: 要执行的任务描述
            max_steps: 最大步数限制（防止无限循环）
            images: 多模态输入的图片列表
            
        Yields:
            各种中间事件：PlanningStep, ActionStep, FinalAnswerStep, ChatMessageStreamDelta
        """
        self.step_number = 1  # 步骤计数器，从 1 开始
        returned_final_answer = False  # 标记是否已得到最终答案
        
        # ReAct 主循环：持续执行直到得到答案或达到步数上限
        while not returned_final_answer and self.step_number <= max_steps:
            if self.interrupt_switch:  # 支持从外部调用 agent.interrupt() 中断执行，给了一个优雅终止长任务的入口
                raise AgentError("Agent interrupted.", self.logger)

            # ========== 阶段 1：规划步骤（可选）PlanningStep  ==========
            # 检查是否需要触发规划步骤（第 1 步 或 每隔 planning_interval 步）
            # 规划步骤会让 LLM 生成/更新执行计划，帮助 Agent 保持方向感
            if self.planning_interval is not None and (
                self.step_number == 1 or (self.step_number - 1) % self.planning_interval == 0
            ):
                planning_start_time = time.time()
                planning_step = None
                # 生成规划步骤（可能是初始规划或更新规划）
                for element in self._generate_planning_step(
                    task, is_first_step=len(self.memory.steps) == 1, step=self.step_number
                ):  # 注意：这里用 len(self.memory.steps) 而非 step_number，因为可能有之前运行的步骤
                    yield element  # 流式 yield 规划过程中的所有事件
                    planning_step = element  # 保存最后一个元素（应该是 PlanningStep）
                assert isinstance(planning_step, PlanningStep)  # 确保最后 yield 的是 PlanningStep
                planning_end_time = time.time()
                # 记录规划步骤的耗时
                planning_step.timing = Timing(
                    start_time=planning_start_time,
                    end_time=planning_end_time,
                )
                # 完成步骤：触发回调、写入记忆
                self._finalize_step(planning_step)
                self.memory.steps.append(planning_step)

            # ========== 阶段 2：行动步骤（核心）ActionStep ==========
            # 开始执行一个行动步骤（Action Step）
            action_step_start_time = time.time()
            action_step = ActionStep(
                step_number=self.step_number,
                timing=Timing(start_time=action_step_start_time),
                observations_images=images,  # 多模态输入：将图片注入每一步的观察中
            )
            # 注意：这里还没有真正执行“动作”。
            # 这一步只是先创建一个 ActionStep 容器，用来记录当前这一轮 ReAct 循环里发生的所有事情。
            # 后面这一轮执行过程中产生的内容，都会逐步写回这个对象，例如：
            # - 发给模型的输入消息
            # - 模型输出
            # - 解析出的工具调用或代码
            # - 工具/代码执行后的 observation
            # - 错误信息、token 使用量、是否已经得到最终答案
            #
            # 真正的执行发生在下面的 self._step_stream(action_step)：
            # - ToolCallingAgent：生成并执行 tool calls
            # - CodeAgent：生成并执行 Python 代码
            #
            # 之所以先创建 ActionStep，再执行动作，是因为无论成功还是失败，
            # 框架都需要一个统一的记录对象在最后写入 memory，供下一轮推理和调试回放使用。
            self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
            try:
                # 执行一步 ReAct 循环：思考 → 行动 → 观察
                # _step_stream 由子类实现（ToolCallingAgent 或 CodeAgent）
                for output in self._step_stream(action_step):
                    yield output  # 流式 yield 所有中间事件（LLM token / 工具调用 / 工具结果）

                    # 检查是否得到了最终答案
                    if isinstance(output, ActionOutput) and output.is_final_answer:
                        final_answer = output.output
                        self.logger.log(
                            Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                            level=LogLevel.INFO,
                        )

                        # 运行用户自定义的答案验证函数（如果有）
                        if self.final_answer_checks:
                            self._validate_final_answer(final_answer)  # 验证失败会抛出异常
                        returned_final_answer = True  # 标记已得到最终答案，准备退出循环
                        action_step.is_final_answer = True

            except AgentGenerationError as e:
                # AgentGenerationError 是代码实现错误（非模型错误），应立即终止
                # 例如：模型 API 调用失败、代码解析失败等
                raise e
            except AgentError as e:
                # 其他 AgentError（如工具执行失败）是模型导致的，记录后继续下一步
                # 例如：工具调用参数错误、代码执行出错等
                # 这些错误会写入记忆，让 LLM 在下一步中看到并尝试修正
                action_step.error = e
            finally:
                # 无论成功还是失败，都要完成步骤的收尾工作
                self._finalize_step(action_step)  # 触发回调、记录耗时
                self.memory.steps.append(action_step)  # 写入记忆
                yield action_step  # yield 完整的步骤对象
                self.step_number += 1  # 步骤计数器 +1

        # ========== 阶段 3：处理达到最大步数的情况 ==========
        if not returned_final_answer and self.step_number == max_steps + 1:
            # 达到最大步数仍未得到答案，强制让 LLM 基于当前记忆生成最终答案
            final_answer = self._handle_max_steps_reached(task)
            yield action_step  # yield 最后一个步骤（包含错误信息）
            
        # ========== 阶段 4：返回最终答案 ==========
        # 创建最终答案步骤并 yield
        final_answer_step = FinalAnswerStep(handle_agent_output_types(final_answer))
        self._finalize_step(final_answer_step)
        yield final_answer_step

    def _validate_final_answer(self, final_answer: Any):
        for check_function in self.final_answer_checks:
            try:
                assert check_function(final_answer, self.memory, agent=self)
            except Exception as e:
                raise AgentError(f"Check {check_function.__name__} failed with error: {e}", self.logger)

    def _finalize_step(self, memory_step: ActionStep | PlanningStep | FinalAnswerStep):
        # 统一收尾一个 step。
        # 这个方法不负责执行推理，也不负责把 step 写入 memory，
        # 它只负责“某一步已经结束后”的公共收尾逻辑。
        #
        # 作用 1：补齐结束时间
        # 对 ActionStep 和 PlanningStep，需要在这里写入 timing.end_time，
        # 这样 monitoring.py 里的 Timing.duration 才能正确计算本步耗时。
        # FinalAnswerStep 更像一个结果封装事件，而不是一次完整的执行过程，
        # 因此这里不额外补 end_time。
        #
        # 作用 2：触发回调
        # 调用 self.step_callbacks.callback(...)，把当前 step 分发给已注册的回调函数。
        # 这些回调通常用于：
        # - 更新监控指标（如 token、耗时）
        # - 输出日志
        # - 自定义调试或埋点
        #
        # 可以把它理解成“step 生命周期结束时的统一钩子”。
        if not isinstance(memory_step, FinalAnswerStep):
            memory_step.timing.end_time = time.time()
        self.step_callbacks.callback(memory_step, agent=self)

    def _handle_max_steps_reached(self, task: str) -> Any:
        action_step_start_time = time.time()
        final_answer = self.provide_final_answer(task)
        final_memory_step = ActionStep(
            step_number=self.step_number,
            error=AgentMaxStepsError("Reached max steps.", self.logger),
            timing=Timing(start_time=action_step_start_time, end_time=time.time()),
            token_usage=final_answer.token_usage,
        )
        final_memory_step.action_output = final_answer.content
        self._finalize_step(final_memory_step)
        self.memory.steps.append(final_memory_step)
        return final_answer.content

    def _generate_planning_step(
        self, task, is_first_step: bool, step: int
    ) -> Generator[ChatMessageStreamDelta | PlanningStep]:
        start_time = time.time()
        if is_first_step:
            input_messages = [
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "text",
                            "text": populate_template(
                                self.prompt_templates["planning"]["initial_plan"],
                                variables={"task": task, "tools": self.tools, "managed_agents": self.managed_agents},
                            ),
                        }
                    ],
                )
            ]
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                plan_message_content = ""
                output_stream = self.model.generate_stream(input_messages, stop_sequences=["<end_plan>"])  # type: ignore
                input_tokens, output_tokens = 0, 0
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in output_stream:
                        if event.content is not None:
                            plan_message_content += event.content
                            live.update(Markdown(plan_message_content))
                            if event.token_usage:
                                input_tokens = event.token_usage.input_tokens
                                output_tokens += event.token_usage.output_tokens
                        yield event
            else:
                plan_message = self.model.generate(input_messages, stop_sequences=["<end_plan>"])
                plan_message_content = plan_message.content
                input_tokens, output_tokens = 0, 0
                if plan_message.token_usage:
                    input_tokens = plan_message.token_usage.input_tokens
                    output_tokens = plan_message.token_usage.output_tokens
            plan = textwrap.dedent(
                f"""Here are the facts I know and the plan of action that I will follow to solve the task:\n```\n{plan_message_content}\n```"""
            )
        else:
            # Summary mode removes the system prompt and previous planning messages output by the model.
            # Removing previous planning messages avoids influencing too much the new plan.
            memory_messages = self.write_memory_to_messages(summary_mode=True)
            plan_update_pre = ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["update_plan_pre_messages"], variables={"task": task}
                        ),
                    }
                ],
            )
            plan_update_post = ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["update_plan_post_messages"],
                            variables={
                                "task": task,
                                "tools": self.tools,
                                "managed_agents": self.managed_agents,
                                "remaining_steps": (self.max_steps - step),
                            },
                        ),
                    }
                ],
            )
            input_messages = [plan_update_pre] + memory_messages + [plan_update_post]
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                plan_message_content = ""
                input_tokens, output_tokens = 0, 0
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in self.model.generate_stream(
                        input_messages,
                        stop_sequences=["<end_plan>"],
                    ):  # type: ignore
                        if event.content is not None:
                            plan_message_content += event.content
                            live.update(Markdown(plan_message_content))
                            if event.token_usage:
                                input_tokens = event.token_usage.input_tokens
                                output_tokens += event.token_usage.output_tokens
                        yield event
            else:
                plan_message = self.model.generate(input_messages, stop_sequences=["<end_plan>"])
                plan_message_content = plan_message.content
                input_tokens, output_tokens = 0, 0
                if plan_message.token_usage:
                    input_tokens = plan_message.token_usage.input_tokens
                    output_tokens = plan_message.token_usage.output_tokens
            plan = textwrap.dedent(
                f"""I still need to solve the task I was given:\n```\n{self.task}\n```\n\nHere are the facts I know and my new/updated plan of action to solve the task:\n```\n{plan_message_content}\n```"""
            )
        log_headline = "Initial plan" if is_first_step else "Updated plan"
        self.logger.log(Rule(f"[bold]{log_headline}", style="orange"), Text(plan), level=LogLevel.INFO)
        yield PlanningStep(
            model_input_messages=input_messages,
            plan=plan,
            model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content=plan_message_content),
            token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            timing=Timing(start_time=start_time, end_time=time.time()),
        )

    @abstractmethod
    def initialize_system_prompt(self) -> str:
        """To be implemented in child classes"""
        ...

    def interrupt(self):
        """Interrupts the agent execution."""
        self.interrupt_switch = True

    def write_memory_to_messages(
        self,
        summary_mode: bool = False,
    ) -> list[ChatMessage]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages

    def _step_stream(
        self, memory_step: ActionStep
    ) -> Generator[ChatMessageStreamDelta | ToolCall | ToolOutput | ActionOutput]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields ChatMessageStreamDelta during the run if streaming is enabled.
        At the end, yields either None if the step is not final, or the final answer.
        """
        raise NotImplementedError("This method should be implemented in child classes")

    def step(self, memory_step: ActionStep) -> Any:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns either None if the step is not final, or the final answer.
        """
        return list(self._step_stream(memory_step))[-1]

    def extract_action(self, model_output: str, split_token: str) -> tuple[str, str]:
        """
        Parse action from the LLM output

        Args:
            model_output (`str`): Output of the LLM
            split_token (`str`): Separator for the action. Should match the example in the system prompt.
        """
        try:
            split = model_output.split(split_token)
            rationale, action = (
                split[-2],
                split[-1],
            )  # NOTE: using indexes starting from the end solves for when you have more than one split_token in the output
        except Exception:
            raise AgentParsingError(
                f"No '{split_token}' token provided in your output.\nYour output:\n{model_output}\n. Be sure to include an action, prefaced with '{split_token}'!",
                self.logger,
            )
        return rationale.strip(), action.strip()

    def provide_final_answer(self, task: str) -> ChatMessage:
        """
        Provide the final answer to the task, based on the logs of the agent's interactions.

        Args:
            task (`str`): Task to perform.
            images (`list[PIL.Image.Image]`, *optional*): Image(s) objects.

        Returns:
            `str`: Final answer to the task.
        """
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    {
                        "type": "text",
                        "text": self.prompt_templates["final_answer"]["pre_messages"],
                    }
                ],
            )
        ]
        messages += self.write_memory_to_messages()[1:]
        messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["final_answer"]["post_messages"], variables={"task": task}
                        ),
                    }
                ],
            )
        )
        try:
            chat_message: ChatMessage = self.model.generate(messages)
            return chat_message
        except Exception as e:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=[{"type": "text", "text": f"Error in generating final LLM output: {e}"}],
            )

    def visualize(self):
        """Creates a rich tree visualization of the agent's structure."""
        self.logger.visualize_agent_tree(self)

    def replay(self, detailed: bool = False):
        """Prints a pretty replay of the agent's steps.

        Args:
            detailed (bool, optional): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        self.memory.replay(self.logger, detailed=detailed)

    def __call__(self, task: str, **kwargs):
        """当该 Agent 作为子 Agent 被主 Agent 调用时触发（而非直接 .run()）。
        会在任务前后插入特定的提示词模板（managed_agent.task / managed_agent.report），
        让子 Agent 知道自己处于被管理的角色中，并将结果格式化后返回给主 Agent。
        """
        full_task = populate_template(
            self.prompt_templates["managed_agent"]["task"],
            variables=dict(name=self.name, task=task),
        )
        result = self.run(full_task, **kwargs)
        if isinstance(result, RunResult):
            report = result.output
        else:
            report = result
        answer = populate_template(
            self.prompt_templates["managed_agent"]["report"], variables=dict(name=self.name, final_answer=report)
        )
        if self.provide_run_summary:
            answer += "\n\nFor more detail, find below a summary of this agent's work:\n<summary_of_work>\n"
            for message in self.write_memory_to_messages(summary_mode=True):
                content = message.content
                answer += "\n" + truncate_content(str(content)) + "\n---"
            answer += "\n</summary_of_work>"
        return answer

    def save(self, output_dir: str | Path, relative_path: str | None = None):
        """
        Saves the relevant code files for your agent. This will copy the code of your agent in `output_dir` as well as autogenerate:

        - a `tools` folder containing the logic for each of the tools under `tools/{tool_name}.py`.
        - a `managed_agents` folder containing the logic for each of the managed agents.
        - an `agent.json` file containing a dictionary representing your agent.
        - a `prompt.yaml` file containing the prompt templates used by your agent.
        - an `app.py` file providing a UI for your agent when it is exported to a Space with `agent.push_to_hub()`
        - a `requirements.txt` containing the names of the modules used by your tool (as detected when inspecting its
          code)

        Args:
            output_dir (`str` or `Path`): The folder in which you want to save your agent.
        """
        make_init_file(output_dir)

        # Recursively save managed agents
        if self.managed_agents:
            make_init_file(os.path.join(output_dir, "managed_agents"))
            for agent_name, agent in self.managed_agents.items():
                agent_suffix = f"managed_agents.{agent_name}"
                if relative_path:
                    agent_suffix = relative_path + "." + agent_suffix
                agent.save(os.path.join(output_dir, "managed_agents", agent_name), relative_path=agent_suffix)

        class_name = self.__class__.__name__

        # Save tools to different .py files
        for tool in self.tools.values():
            make_init_file(os.path.join(output_dir, "tools"))
            tool.save(os.path.join(output_dir, "tools"), tool_file_name=tool.name, make_gradio_app=False)

        # Save prompts to yaml
        yaml_prompts = yaml.safe_dump(
            self.prompt_templates,
            default_style="|",  # This forces block literals for all strings
            default_flow_style=False,
            width=float("inf"),
            sort_keys=False,
            allow_unicode=True,
            indent=2,
        )

        with open(os.path.join(output_dir, "prompts.yaml"), "w", encoding="utf-8") as f:
            f.write(yaml_prompts)

        # Save agent dictionary to json
        agent_dict = self.to_dict()
        agent_dict["tools"] = [tool.name for tool in self.tools.values()]
        agent_dict["managed_agents"] = {agent.name: agent.__class__.__name__ for agent in self.managed_agents.values()}
        with open(os.path.join(output_dir, "agent.json"), "w", encoding="utf-8") as f:
            json.dump(agent_dict, f, indent=4)

        # Save requirements
        with open(os.path.join(output_dir, "requirements.txt"), "w", encoding="utf-8") as f:
            f.writelines(f"{r}\n" for r in agent_dict["requirements"])

        # Make agent.py file with Gradio UI
        agent_name = f"agent_{self.name}" if getattr(self, "name", None) else "agent"
        managed_agent_relative_path = relative_path + "." if relative_path is not None else ""
        app_template = create_agent_gradio_app_template()

        # Render the app.py file from Jinja2 template
        app_text = app_template.render(
            {
                "agent_name": agent_name,
                "class_name": class_name,
                "agent_dict": agent_dict,
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "managed_agent_relative_path": managed_agent_relative_path,
            }
        )

        with open(os.path.join(output_dir, "app.py"), "w", encoding="utf-8") as f:
            f.write(app_text + "\n")  # Append newline at the end

    def to_dict(self) -> dict[str, Any]:
        """Convert the agent to a dictionary representation.

        Returns:
            `dict`: Dictionary representation of the agent.
        """
        # TODO: handle serializing step_callbacks and final_answer_checks
        for attr in ["final_answer_checks", "step_callbacks"]:
            if getattr(self, attr, None):
                self.logger.log(f"This agent has {attr}: they will be ignored by this method.", LogLevel.INFO)

        tool_dicts = [tool.to_dict() for tool in self.tools.values()]
        tool_requirements = {req for tool in self.tools.values() for req in tool.to_dict()["requirements"]}
        managed_agents_requirements = {
            req for managed_agent in self.managed_agents.values() for req in managed_agent.to_dict()["requirements"]
        }
        requirements = tool_requirements | managed_agents_requirements
        if hasattr(self, "authorized_imports"):
            requirements.update(
                {package.split(".")[0] for package in self.authorized_imports if package not in BASE_BUILTIN_MODULES}
            )

        agent_dict = {
            "class": self.__class__.__name__,
            "tools": tool_dicts,
            "model": {
                "class": self.model.__class__.__name__,
                "data": self.model.to_dict(),
            },
            "managed_agents": [managed_agent.to_dict() for managed_agent in self.managed_agents.values()],
            "prompt_templates": self.prompt_templates,
            "max_steps": self.max_steps,
            "verbosity_level": int(self.logger.level),
            "planning_interval": self.planning_interval,
            "name": self.name,
            "description": self.description,
            "requirements": sorted(requirements),
        }
        return agent_dict

    @classmethod
    def from_dict(cls, agent_dict: dict[str, Any], **kwargs) -> "MultiStepAgent":
        """Create agent from a dictionary representation.

        Args:
            agent_dict (`dict[str, Any]`): Dictionary representation of the agent.
            **kwargs: Additional keyword arguments that will override agent_dict values.

        Returns:
            `MultiStepAgent`: Instance of the agent class.
        """
        # Load model
        model_info = agent_dict["model"]
        model_class = MODEL_REGISTRY.get(model_info["class"])
        if model_class is None:
            raise ValueError(
                f"Unknown model class '{model_info['class']}'. "
                f"Supported models: {', '.join(sorted(MODEL_REGISTRY.keys()))}"
            )
        model = model_class.from_dict(model_info["data"])
        # Load tools
        tools = []
        for tool_info in agent_dict["tools"]:
            tools.append(Tool.from_code(tool_info["code"]))
        # Load managed agents
        managed_agents = []
        for managed_agent_dict in agent_dict["managed_agents"]:
            agent_class = AGENT_REGISTRY.get(managed_agent_dict["class"])
            if agent_class is None:
                raise ValueError(
                    f"Unknown agent class '{managed_agent_dict['class']}'. "
                    f"Supported agents: {', '.join(sorted(AGENT_REGISTRY.keys()))}"
                )
            managed_agent = agent_class.from_dict(managed_agent_dict, **kwargs)
            managed_agents.append(managed_agent)
        # Extract base agent parameters
        agent_args = {
            "model": model,
            "tools": tools,
            "managed_agents": managed_agents,
            "prompt_templates": agent_dict.get("prompt_templates"),
            "max_steps": agent_dict.get("max_steps"),
            "verbosity_level": agent_dict.get("verbosity_level"),
            "planning_interval": agent_dict.get("planning_interval"),
            "name": agent_dict.get("name"),
            "description": agent_dict.get("description"),
        }
        # Filter out None values to use defaults from __init__
        agent_args = {k: v for k, v in agent_args.items() if v is not None}
        # Update with any additional kwargs
        agent_args.update(kwargs)
        # Create agent instance
        return cls(**agent_args)

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        token: str | None = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Loads an agent defined on the Hub.

        <Tip warning={true}>

        Loading a tool from the Hub means that you'll download the tool and execute it locally.
        ALWAYS inspect the tool you're downloading before loading it within your runtime, as you would do when
        installing a package using pip/npm/apt.

        </Tip>

        Args:
            repo_id (`str`):
                The name of the repo on the Hub where your tool is defined.
            token (`str`, *optional*):
                The token to identify you on hf.co. If unset, will use the token generated when running
                `huggingface-cli login` (stored in `~/.huggingface`).
            trust_remote_code(`bool`, *optional*, defaults to False):
                This flags marks that you understand the risk of running remote code and that you trust this tool.
                If not setting this to True, loading the tool from Hub will fail.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
                `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your agent, and the
                others will be passed along to its init.
        """
        if not trust_remote_code:
            raise ValueError(
                "Loading an agent from Hub requires to acknowledge you trust its code: to do so, pass `trust_remote_code=True`."
            )

        # Get the agent's Hub folder.
        download_kwargs = {"token": token, "repo_type": "space"} | {
            key: kwargs.pop(key)
            for key in [
                "cache_dir",
                "force_download",
                "proxies",
                "revision",
                "local_files_only",
            ]
            if key in kwargs
        }

        download_folder = Path(snapshot_download(repo_id=repo_id, **download_kwargs))
        return cls.from_folder(download_folder, **kwargs)

    @classmethod
    def from_folder(cls, folder: str | Path, **kwargs):
        """Loads an agent from a local folder.

        Args:
            folder (`str` or `Path`): The folder where the agent is saved.
            **kwargs: Additional keyword arguments that will be passed to the agent's init.
        """
        # Load agent.json
        folder = Path(folder)
        agent_dict = json.loads((folder / "agent.json").read_text())
        # Handle HfApiModel -> InferenceClientModel rename for old agents
        if agent_dict.get("model", {}).get("class") == "HfApiModel":
            agent_dict["model"]["class"] = "InferenceClientModel"
            logger.warning(
                "The agent you're loading uses the deprecated 'HfApiModel' class: it was automatically updated to 'InferenceClientModel'."
            )
        # Load managed agents from their respective folders, recursively
        managed_agents = []
        for managed_agent_name, managed_agent_class_name in agent_dict["managed_agents"].items():
            agent_cls = AGENT_REGISTRY.get(managed_agent_class_name)
            if agent_cls is None:
                raise ValueError(
                    f"Unknown agent class '{managed_agent_class_name}'. "
                    f"Supported agents: {', '.join(sorted(AGENT_REGISTRY.keys()))}"
                )
            managed_agents.append(agent_cls.from_folder(folder / "managed_agents" / managed_agent_name))
        agent_dict["managed_agents"] = {}

        # Load tools
        tools = []
        for tool_name in agent_dict["tools"]:
            tool_code = (folder / "tools" / f"{tool_name}.py").read_text()
            tools.append({"name": tool_name, "code": tool_code})
        agent_dict["tools"] = tools

        # Add managed agents to kwargs to override the empty list in from_dict
        if managed_agents:
            kwargs["managed_agents"] = managed_agents

        return cls.from_dict(agent_dict, **kwargs)

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload agent",
        private: bool | None = None,
        token: bool | str | None = None,
        create_pr: bool = False,
    ) -> str:
        """
        Upload the agent to the Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push to. It should contain your organization name when
                pushing to a given organization.
            commit_message (`str`, *optional*, defaults to `"Upload agent"`):
                Message to commit while pushing.
            private (`bool`, *optional*, defaults to `None`):
                Whether to make the repo private. If `None`, the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether to create a PR with the uploaded files or directly commit.
        """
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="space",
            space_sdk="gradio",
        )
        repo_id = repo_url.repo_id
        metadata_update(
            repo_id,
            {"tags": ["smolagents", "agent"]},
            repo_type="space",
            token=token,
            overwrite=True,
        )

        with tempfile.TemporaryDirectory() as work_dir:
            self.save(work_dir)
            logger.info(f"Uploading the following files to {repo_id}: {','.join(os.listdir(work_dir))}")
            return upload_folder(
                repo_id=repo_id,
                commit_message=commit_message,
                folder_path=work_dir,
                token=token,
                create_pr=create_pr,
                repo_type="space",
            )


# =============================================================================
# ToolCallingAgent —— 工具调用型 Agent
#
# 工作方式：
#   LLM 输出结构化 JSON 格式的工具调用，框架解析 JSON 后执行对应工具。
#   示例 LLM 输出：
#     {"tool": "get_weather", "arguments": {"location": "Paris", "celsius": true}}
#
# 优点：与 OpenAI function calling 风格兼容，简单直观
# 缺点：每步只能做一件事，复杂多步逻辑需要更多步数
# 适合：简单的单工具调用、与 OpenAI 生态集成
# =============================================================================
class ToolCallingAgent(MultiStepAgent):
    """
    基于 JSON 工具调用的 Agent 实现
    
    ToolCallingAgent 是 MultiStepAgent 的一个特化版本，专门设计用于利用现代 LLM 
    （如 OpenAI GPT、Claude 等）的原生工具调用能力。与 CodeAgent 不同，它不生成
    Python 代码，而是使用标准化的 JSON 格式来调用工具。
    
    核心特性：
    1. **原生工具调用**：利用 LLM 的内置工具调用 API（如 OpenAI 的 function calling）
    2. **JSON 格式**：使用结构化的 JSON 来描述工具调用，而非代码
    3. **并行调用**：支持在一次响应中调用多个工具
    4. **流式输出**：支持实时显示 LLM 的思考过程
    5. **标准兼容**：与主流 LLM 服务的工具调用标准兼容
    
    适用场景：
    - 需要与 OpenAI、Anthropic 等服务的工具调用功能集成
    - 希望使用标准化的工具调用格式
    - 需要并行执行多个工具调用
    - 对代码执行安全性要求不高的场景
    
    与 CodeAgent 的区别：
    - CodeAgent：生成 Python 代码执行工具调用，更灵活但需要代码执行环境
    - ToolCallingAgent：使用 JSON 格式调用工具，更标准但灵活性相对较低
    
    Args:
        tools (`list[Tool]`): Agent 可以使用的工具列表
            每个工具都会被转换为 LLM 可理解的工具描述格式
        model (`Model`): 用于生成 Agent 行为的语言模型
            必须支持工具调用功能（如 OpenAI GPT-3.5/4、Claude 等）
        prompt_templates ([`~agents.PromptTemplates`], *optional*): 自定义提示词模板
            如果不提供，将使用默认的 toolcalling_agent.yaml 模板
        planning_interval (`int`, *optional*): 规划步骤的执行间隔
            每隔多少步执行一次任务规划，None 表示不进行规划
        stream_outputs (`bool`, *optional*, default `False`): 是否启用流式输出
            True: 实时显示 LLM 的思考过程（需要模型支持 generate_stream）
            False: 等待完整响应后一次性显示
        max_tool_threads (`int`, *optional*): 并行工具调用的最大线程数
            当 LLM 在一次响应中请求多个工具调用时，控制并发执行的线程数
            更高的值增加并发性但也增加资源使用，默认使用 ThreadPoolExecutor 的默认值
        **kwargs: 传递给父类 MultiStepAgent 的额外参数
    
    Example:
        ```python
        from smolagents import ToolCallingAgent, OpenAIModel
        from smolagents.tools import WebSearchTool, CalculatorTool
        
        # 创建支持工具调用的模型
        model = OpenAIModel("gpt-4")
        
        # 创建工具列表
        tools = [
            WebSearchTool(),
            CalculatorTool()
        ]
        
        # 创建 ToolCallingAgent
        agent = ToolCallingAgent(
            tools=tools,
            model=model,
            stream_outputs=True,  # 启用流式输出
            max_tool_threads=3    # 最多并行执行3个工具
        )
        
        # 执行任务
        result = agent.run("搜索今天的天气，然后计算华氏度转摄氏度")
        ```
    
    工具调用流程：
        1. LLM 分析任务并决定需要调用哪些工具
        2. 生成标准化的 JSON 工具调用请求
        3. Agent 解析 JSON 并并行执行工具调用
        4. 将工具执行结果返回给 LLM
        5. LLM 基于结果继续推理或给出最终答案
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        prompt_templates: PromptTemplates | None = None,
        planning_interval: int | None = None,
        stream_outputs: bool = False,
        max_tool_threads: int | None = None,
        **kwargs,
    ):
        """
        初始化 ToolCallingAgent
        
        这个初始化过程专门为基于 JSON 的工具调用进行了优化，
        包括加载专用的提示词模板和配置并行执行参数。
        """
        # === 第1步：加载专用提示词模板 ===
        # 如果用户没有提供自定义模板，则加载 ToolCallingAgent 专用的 YAML 模板
        # 这个模板专门为 JSON 工具调用格式进行了优化
        prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files("smolagents.prompts").joinpath("toolcalling_agent.yaml").read_text()
        )
        
        # === 第2步：调用父类初始化 ===
        # 继承 MultiStepAgent 的所有基础功能（工具管理、记忆系统、回调等）
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            **kwargs,
        )
        
        # === 第3步：配置流式输出功能 ===
        # 流式输出允许实时显示 LLM 的思考过程，提升用户体验
        self.stream_outputs = stream_outputs
        
        # 验证模型是否支持流式输出
        if self.stream_outputs and not hasattr(self.model, "generate_stream"):
            raise ValueError(
                "`stream_outputs` is set to True, but the model class implements no `generate_stream` method."
            )
        
        # === 第4步：配置并行工具调用 ===
        # 现代 LLM 可以在一次响应中请求多个工具调用
        # 这个参数控制同时执行的工具调用数量，平衡性能和资源使用
        self.max_tool_threads = max_tool_threads

    @property
    def tools_and_managed_agents(self):
        """
        获取所有可用的工具和子 Agent 的组合列表
        
        这个属性将工具字典和子 Agent 字典的值合并为一个列表，
        方便传递给 LLM 的工具调用接口。
        
        Returns:
            list: 包含所有tool和子 Agent 对象的列表
        """
        return list(self.tools.values()) + list(self.managed_agents.values())

    def initialize_system_prompt(self) -> str:
        """
        初始化系统提示词
        
        为 ToolCallingAgent 生成专门的系统提示词，告诉 LLM 如何使用
        JSON 格式进行工具调用，以及可用的工具和子 Agent 信息。
        
        Returns:
            str: 完整的系统提示词字符串
        """
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,                    # 可用工具列表
                "managed_agents": self.managed_agents,  # 可用子 Agent 列表
                "custom_instructions": self.instructions, # 用户自定义指令
            },
        )
        return system_prompt

    def _step_stream(
        self, memory_step: ActionStep
    ) -> Generator[ChatMessageStreamDelta | ToolCall | ToolOutput | ActionOutput]:
        """
        执行 ReAct 框架中的单个步骤：思考、行动、观察
        
        这是 ToolCallingAgent 的核心执行方法，负责：
        1. 准备对话历史和上下文
        2. 调用 LLM 生成响应（支持流式输出）
        3. 解析 LLM 的工具调用请求
        4. 并行执行工具调用
        5. 处理工具执行结果
        
        与 CodeAgent 的区别：
        - CodeAgent: 解析和执行 Python 代码
        - ToolCallingAgent: 解析和执行 JSON 工具调用
        
        Args:
            memory_step (ActionStep): 当前执行步骤的记忆对象
        
        Yields:
            ChatMessageStreamDelta: 流式输出时的文本增量（如果启用流式输出）
            ToolCall: 工具调用请求对象
            ToolOutput: 工具执行结果对象
            ActionOutput: 最终的行动输出（如果这是最后一步）
        """
        # === 第1步：准备输入消息 ===
        # 将记忆系统中的历史对话转换为 LLM 可理解的消息格式
        memory_messages = self.write_memory_to_messages()
        input_messages = memory_messages.copy()

        # 记录模型输入到记忆步骤中（用于调试和回放）
        memory_step.model_input_messages = input_messages

        try:
            # === 第2步：选择生成模式（流式 vs 非流式） ===
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                # === 流式模式：实时显示 LLM 思考过程 ===
                output_stream = self.model.generate_stream(
                    input_messages,
                    stop_sequences=["Observation:", "Calling tools:"],  # 停止序列，控制输出边界
                    tools_to_call_from=self.tools_and_managed_agents,   # 可调用的工具列表
                )
                
                # 收集流式输出的文本片段
                chat_message_stream_deltas: list[ChatMessageStreamDelta] = []
                
                # 使用 Rich 库实现实时显示（类似打字机效果）
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in output_stream:
                        chat_message_stream_deltas.append(event)
                        # 实时更新显示内容
                        live.update(
                            Markdown(agglomerate_stream_deltas(chat_message_stream_deltas).render_as_markdown())
                        )
                        yield event  # 向调用方返回流式事件
                
                # 将所有流式片段合并为完整消息
                chat_message = agglomerate_stream_deltas(chat_message_stream_deltas)
            else:
                # === 非流式模式：等待完整响应 ===
                chat_message: ChatMessage = self.model.generate(
                    input_messages,
                    stop_sequences=["Observation:", "Calling tools:"],
                    tools_to_call_from=self.tools_and_managed_agents,
                )
                # 记录 LLM 输出到日志
                self.logger.log_markdown(
                    content=str(chat_message.content or chat_message.raw or ""),
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )

            # === 第3步：记录模型输出 ===
            # 将 LLM 的完整响应保存到记忆步骤中
            memory_step.model_output_message = chat_message
            memory_step.model_output = chat_message.content
            memory_step.token_usage = chat_message.token_usage
        except Exception as e:
            raise AgentGenerationError(f"Error while generating output:\n{e}", self.logger) from e

        # 如果模型没有原生支持 tool_calls（如老版模型），尝试从文本中解析工具调用
        if chat_message.tool_calls is None or len(chat_message.tool_calls) == 0:
            try:
                chat_message = self.model.parse_tool_calls(chat_message)
            except Exception as e:
                raise AgentParsingError(f"Error while parsing tool call from model output: {e}", self.logger)
        else:
            # 将 arguments 从字符串解析为 dict（部分模型返回 JSON 字符串而非 dict）
            for tool_call in chat_message.tool_calls:
                tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
        final_answer, got_final_answer = None, False
        for output in self.process_tool_calls(chat_message, memory_step):
            yield output
            if isinstance(output, ToolOutput):
                if output.is_final_answer:
                    if len(chat_message.tool_calls) > 1:
                        raise AgentExecutionError(
                            "If you want to return an answer, please do not perform any other tool calls than the final answer tool call!",
                            self.logger,
                        )
                    if got_final_answer:
                        raise AgentToolExecutionError(
                            "You returned multiple final answers. Please return only one single final answer!",
                            self.logger,
                        )
                    final_answer = output.output
                    got_final_answer = True

                    # Manage state variables
                    if isinstance(final_answer, str) and final_answer in self.state.keys():
                        final_answer = self.state[final_answer]
        yield ActionOutput(
            output=final_answer,
            is_final_answer=got_final_answer,
        )

    def process_tool_calls(
        self, chat_message: ChatMessage, memory_step: ActionStep
    ) -> Generator[ToolCall | ToolOutput]:
        """处理 LLM 输出的工具调用请求，并更新 Agent 记忆。 
        
        这个方法负责：
        1. 解析 LLM 输出中的所有工具调用请求
        2. 执行这些工具（单个工具直接调用，多个工具并行执行）
        3. 收集工具执行结果并格式化为观察（Observation）
        4. 将工具调用和观察写入记忆

        Args:
            chat_message: LLM 的输出消息，包含 tool_calls 列表
            memory_step: 当前步骤的记忆对象，用于记录工具调用和观察

        Yields:
            ToolCall: 工具调用请求（在执行前 yield）
            ToolOutput: 工具执行结果（在执行后 yield）
        """
        # 第一步：收集所有工具调用请求
        parallel_calls: dict[str, ToolCall] = {}
        assert chat_message.tool_calls is not None
        for chat_tool_call in chat_message.tool_calls:
            # 将 ChatMessageToolCall 转换为内部的 ToolCall 格式
            tool_call = ToolCall(
                name=chat_tool_call.function.name, arguments=chat_tool_call.function.arguments, id=chat_tool_call.id
            )
            yield tool_call  # 先 yield 工具调用请求（让外部知道即将执行什么）
            parallel_calls[tool_call.id] = tool_call

        # 第二步：定义单个工具调用的处理函数
        def process_single_tool_call(tool_call: ToolCall) -> ToolOutput:
            """执行单个工具调用并返回结果。
            
            这个内部函数会被主线程或线程池调用。
            """
            tool_name = tool_call.name
            tool_arguments = tool_call.arguments or {}
            self.logger.log(
                Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
                level=LogLevel.INFO,
            )
            
            # 执行工具调用（会进行参数验证和错误处理）
            tool_call_result = self.execute_tool_call(tool_name, tool_arguments)
            tool_call_result_type = type(tool_call_result)
            
            # 特殊处理：如果工具返回的是图片或音频，存储到状态中
            # 这样后续步骤可以通过变量名引用这些资源
            if tool_call_result_type in [AgentImage, AgentAudio]:
                if tool_call_result_type == AgentImage:
                    observation_name = "image.png"
                elif tool_call_result_type == AgentAudio:
                    observation_name = "audio.mp3"
                # TODO: 未来可以支持更灵活的命名（如 image_1.png, image_2.png）
                self.state[observation_name] = tool_call_result
                observation = f"Stored '{observation_name}' in memory."
            else:
                # 普通结果：直接转为字符串作为观察
                observation = str(tool_call_result).strip()
                
            self.logger.log(
                f"Observations: {observation.replace('[', '|')}",  # 转义可能的 rich 标签字符
                level=LogLevel.INFO,
            )
            
            # 检查是否是最终答案工具
            is_final_answer = tool_name == "final_answer"

            return ToolOutput(
                id=tool_call.id,
                output=tool_call_result,
                is_final_answer=is_final_answer,
                observation=observation,
                tool_call=tool_call,
            )

        # 第三步：执行工具调用（单个或并行）
        outputs = {}
        if len(parallel_calls) == 1:
            # 只有一个工具调用：直接在主线程执行
            tool_call = list(parallel_calls.values())[0]
            tool_output = process_single_tool_call(tool_call)
            outputs[tool_output.id] = tool_output
            yield tool_output
        else:
            # 多个工具调用：使用线程池并行执行（提升效率）
            # copy_context() 确保每个线程独立继承当前 contextvars（如 trace ID）
            with ThreadPoolExecutor(self.max_tool_threads) as executor:
                futures = []
                for tool_call in parallel_calls.values():
                    ctx = copy_context()  # 复制当前上下文（包含 contextvars）
                    futures.append(executor.submit(ctx.run, process_single_tool_call, tool_call))
                # 按完成顺序收集结果（而非提交顺序）
                for future in as_completed(futures):
                    tool_output = future.result()
                    outputs[tool_output.id] = tool_output
                    yield tool_output

        # 第四步：将工具调用和观察写入记忆
        # 按 ID 排序确保顺序一致性
        memory_step.tool_calls = [parallel_calls[k] for k in sorted(parallel_calls.keys())]
        memory_step.observations = memory_step.observations or ""
        # 合并所有工具的观察结果
        for tool_output in [outputs[k] for k in sorted(outputs.keys())]:
            memory_step.observations += tool_output.observation + "\n"
        # 去除末尾的换行符
        memory_step.observations = (
            memory_step.observations.rstrip("\n") if memory_step.observations else memory_step.observations
        )

    def _substitute_state_variables(self, arguments: dict[str, str] | str) -> dict[str, Any] | str:
        """
        状态变量替换器：将参数中的变量名替换为实际存储的对象
        
        在多步推理中，前面步骤的结果（如图片、数据框）会被存储到 self.state 中。
        后续步骤的 LLM 只能通过字符串变量名来引用这些对象，这个方法负责将变量名解析为实际对象。
        
        例如：
        - self.state = {"image.png": <PIL.Image>, "data": <DataFrame>}
        - arguments = {"img": "image.png", "threshold": 0.5}
        - 返回: {"img": <PIL.Image>, "threshold": 0.5}
        
        Replace string values in arguments with their corresponding state values if they exist.
        """
        # 如果参数是字典，遍历每个键值对进行替换
        if isinstance(arguments, dict):
            return {
                key: self.state.get(value, value)  # 如果 value 是状态中的变量名，替换为实际值；否则保持原值
                     if isinstance(value, str)      # 只替换字符串类型的值（可能是变量名）
                     else value                     # 非字符串值（数字、列表等）直接保留
                for key, value in arguments.items()
            }
        # 如果参数不是字典（可能是单个字符串），直接返回
        return arguments

    def execute_tool_call(self, tool_name: str, arguments: dict[str, str] | str) -> Any:
        """
        工具执行器：执行指定的工具或子 Agent
        
        这是工具调用的核心方法，负责：
        1. 验证工具是否存在
        2. 替换参数中的状态变量
        3. 验证参数格式
        4. 实际执行工具
        5. 处理执行错误
        
        Execute a tool or managed agent with the provided arguments.

        The arguments are replaced with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the tool or managed agent to execute.
            arguments (dict[str, str] | str): Arguments passed to the tool call.
        """
        # === 第1步：检查工具是否存在 ===
        # Check if the tool exists
        # 合并普通工具和子 Agent（它们都可以被调用）
        # 【**】是字典解包的意思 
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            # 工具不存在，抛出友好的错误提示
            raise AgentToolExecutionError(
                f"Unknown tool {tool_name}, should be one of: {', '.join(available_tools)}.", self.logger
            )

        # === 第2步：准备工具和参数 ===
        # Get the tool and substitute state variables in arguments
        tool = available_tools[tool_name]  # 获取工具对象
        # 替换参数中的状态变量（如 "image.png" → <PIL.Image>）
        arguments = self._substitute_state_variables(arguments)
        # 判断是工具还是子 Agent（影响错误提示和调用方式）
        is_managed_agent = tool_name in self.managed_agents

        # === 第3步：验证参数 ===
        try:
            # 检查参数是否符合工具的输入规范（类型、必填项等）
            validate_tool_arguments(tool, arguments)
        except (ValueError, TypeError) as e:
            # 参数格式错误（如缺少必填参数、类型不匹配）
            raise AgentToolCallError(str(e), self.logger) from e
        except Exception as e:
            # 其他验证错误
            error_msg = f"Error executing tool '{tool_name}' with arguments {str(arguments)}: {type(e).__name__}: {e}"
            raise AgentToolExecutionError(error_msg, self.logger) from e

        # === 第4步：执行工具 ===
        try:
            # Call tool with appropriate arguments
            # 根据参数类型选择调用方式
            if isinstance(arguments, dict):
                # 字典参数：使用关键字参数调用，例如：tool(location="Paris", units="metric")
                # 子 Agent 不需要 sanitize（它们自己会处理输入输出）
                # 普通工具需要 sanitize（清理和验证输入输出）
                return tool(**arguments) if is_managed_agent else tool(**arguments, sanitize_inputs_outputs=True)
            else:
                # 单个参数：直接传递，例如：tool("search query")
                return tool(arguments) if is_managed_agent else tool(arguments, sanitize_inputs_outputs=True)

        except Exception as e:
            # === 第5步：处理执行错误 ===
            # Handle execution errors
            # 根据是工具还是子 Agent 生成不同的错误提示
            if is_managed_agent:
                # 子 Agent 执行失败：建议尝试其他团队成员
                error_msg = (
                    f"Error executing request to team member '{tool_name}' with arguments {str(arguments)}: {e}\n"
                    "Please try again or request to another team member"
                )
            else:
                error_msg = (
                    f"Error executing tool '{tool_name}' with arguments {str(arguments)}: {type(e).__name__}: {e}\n"
                    "Please try again or use another tool"
                )
            raise AgentToolExecutionError(error_msg, self.logger) from e


# =============================================================================
# CodeAgent —— 代码型 Agent（smolagents 的核心特色）
#
# 工作方式：
#   LLM 生成完整的 Python 代码片段，框架在沙箱中执行该代码。
#   示例 LLM 输出：
#     temp = get_weather(location="Paris", celsius=True)
#     result = f"巴黎当前温度: {temp}"
#     final_answer(result)
#
# 优点：
#   - 一段代码可同时调用多个工具、进行计算、条件判断、循环
#   - 比 ToolCallingAgent 更灵活，通常需要更少步数
#   - 代码本身就是思维链（Chain of Thought），可读性好
#
# 适合：复杂多步任务、需要计算或数据处理的场景（官方推荐方案）
#
# 执行流程（每步）：
#   1. 将历史记忆转为消息列表 → 调用 LLM 生成代码
#   2. 从 LLM 输出中解析代码块（<code>...</code> 或 ```python...```）
#   3. 在 Python 沙箱（LocalPythonExecutor 或远程执行器）中运行代码
#   4. 将执行日志/输出写入记忆，进入下一步
#   5. 若代码调用了 final_answer()，循环终止
# =============================================================================
class CodeAgent(MultiStepAgent):
    """
    CodeAgent：基于代码生成的智能体实现
    
    这是 smolagents 的核心 Agent 类型，LLM 通过生成 Python 代码来执行任务。
    相比 ToolCallingAgent 的 JSON 调用方式，CodeAgent 更灵活、更强大。
    
    核心特点：
    1. LLM 生成 Python 代码而非 JSON
    2. 支持复杂逻辑（循环、条件判断、变量）
    3. 可以组合多个工具调用
    4. 性能更好（比传统方法少 30% 的步骤）
    
    工作原理：
    ```
    LLM 输出：
    <code>
    result = web_search("Python教程")
    for item in result[:3]:
        print(item)
    final_answer(result)
    </code>
    
    → 解析代码块
    → 在沙箱中执行
    → 收集输出作为观察
    → 继续下一轮推理
    ```
    
    安全性：
    - 代码在沙箱环境中执行（local/e2b/docker/modal/wasm）
    - 限制 import 白名单
    - 截断过长的输出
    
    In this agent, the tool calls will be formulated by the LLM in code format, then parsed and executed.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Model`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        additional_authorized_imports (`list[str]`, *optional*): Additional authorized imports for the agent.
            额外授权的 Python 模块，例如 ["pandas", "numpy"]
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        executor ([`PythonExecutor`], *optional*): Custom Python code executor. If not provided, a default executor will be created based on `executor_type`.
            自定义代码执行器，通常不需要手动提供
        executor_type (`Literal["local", "blaxel", "e2b", "modal", "docker", "wasm"]`, default `"local"`): Type of code executor.
            代码执行环境类型：
            - "local": 本地沙箱（最快，安全性较低）
            - "e2b": E2B 云沙箱（安全，需要 API key）
            - "docker": Docker 容器（安全，需要 Docker）
            - "modal": Modal 云函数（安全，需要账号）
            - "wasm": WebAssembly 沙箱（安全，浏览器兼容）
            - "blaxel": Blaxel 云沙箱（安全，需要 API key）
        executor_kwargs (`dict`, *optional*): Additional arguments to pass to initialize the executor.
        max_print_outputs_length (`int`, *optional*): Maximum length of the print outputs.
            截断过长的 print 输出，防止超出 LLM 上下文限制
        stream_outputs (`bool`, *optional*, default `False`): Whether to stream outputs during execution.
            是否流式输出 LLM 生成的代码（实时显示）
        use_structured_outputs_internally (`bool`, default `False`): Whether to use structured generation at each action step: improves performance for many models.
            是否使用结构化输出（强制 LLM 输出 JSON 格式的 {"thought": "...", "code": "..."}）
            优点：减少解析错误；缺点：不是所有模型都支持

            <Added version="1.17.0"/>
        code_block_tags (`tuple[str, str]` | `Literal["markdown"]`, *optional*): Opening and closing tags for code blocks (regex strings). Pass a custom tuple, or pass 'markdown' to use ("```(?:python|py)", "\\n```"), leave empty to use ("<code>", "</code>").
            代码块标签：
            - None: 使用默认 ("<code>", "</code>")
            - "markdown": 使用 Markdown 格式 ("```python", "```")
            - 自定义元组: 例如 ("<<<", ">>>")
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        prompt_templates: PromptTemplates | None = None,
        additional_authorized_imports: list[str] | None = None,
        planning_interval: int | None = None,
        executor: PythonExecutor = None,
        executor_type: Literal["local", "blaxel", "e2b", "modal", "docker", "wasm"] = "local",
        executor_kwargs: dict[str, Any] | None = None,
        max_print_outputs_length: int | None = None,
        stream_outputs: bool = False,
        use_structured_outputs_internally: bool = False,
        code_block_tags: str | tuple[str, str] | None = None,
        **kwargs,
    ):
        """
        初始化 CodeAgent
        
        这个构造函数负责：
        1. 设置 import 白名单（安全控制）
        2. 加载提示词模板（普通模式 vs 结构化模式）
        3. 配置代码块标签（如何识别代码）
        4. 创建 Python 执行器（本地 vs 远程沙箱）
        """
        # === 第1步：配置 import 白名单 ===
        # import 白名单 = 内置安全模块 + 用户额外授权的模块
        # 沙箱执行器会拦截所有不在白名单中的 import 语句
        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = sorted(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        
        # === 第2步：配置输出限制 ===
        self.max_print_outputs_length = max_print_outputs_length  # 截断过长的 print 输出，防止超出上下文
        
        # === 第3步：选择提示词模板 ===
        self._use_structured_outputs_internally = use_structured_outputs_internally
        # 结构化输出模式：使用 JSON Schema 强制 LLM 输出 {"thought": "...", "code": "..."} 格式
        # 优点：减少代码块解析错误；缺点：并非所有模型都支持
        if self._use_structured_outputs_internally:
            prompt_templates = prompt_templates or yaml.safe_load(
                importlib.resources.files("smolagents.prompts").joinpath("structured_code_agent.yaml").read_text()
            )
        else:
            # 默认模式：LLM 自由输出，通过正则表达式解析代码块
            prompt_templates = prompt_templates or yaml.safe_load(
                importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text()
            )

        # === 第4步：配置代码块标签 ===
        # 代码块标签用于从 LLM 输出中提取代码
        if isinstance(code_block_tags, str) and not code_block_tags == "markdown":
            raise ValueError("Only 'markdown' is supported for a string argument to `code_block_tags`.")
        self.code_block_tags = (
            code_block_tags
            if isinstance(code_block_tags, tuple)  # 自定义标签，如 ("<<<", ">>>")
            else ("```python", "```")              # Markdown 格式
            if code_block_tags == "markdown"
            else ("<code>", "</code>")             # 默认 XML 格式
        )

        # === 第5步：调用父类初始化 ===
        # 初始化工具、模型、记忆等通用组件
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            **kwargs,
        )
        
        # === 第6步：配置流式输出 ===
        self.stream_outputs = stream_outputs
        if self.stream_outputs and not hasattr(self.model, "generate_stream"):
            # 检查模型是否支持流式生成
            raise ValueError(
                "`stream_outputs` is set to True, but the model class implements no `generate_stream` method."
            )
        
        # === 第7步：安全警告 ===
        if "*" in self.additional_authorized_imports:
            # 如果允许所有 import，发出警告（可能导入不存在的包）
            self.logger.log(
                "Caution: you set an authorization for all imports, meaning your agent can decide to import any package it deems necessary. This might raise issues if the package is not installed in your environment.",
                level=LogLevel.INFO,
            )
        
        # === 第8步：创建 Python 执行器 ===
        self.executor_type = executor_type
        self.executor_kwargs: dict[str, Any] = executor_kwargs or {}
        # 如果用户没有提供自定义执行器，根据 executor_type 创建默认执行器
        self.python_executor = executor or self.create_python_executor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def cleanup(self):
        """Clean up resources used by the agent, such as the remote Python executor."""
        if hasattr(self.python_executor, "cleanup"):
            self.python_executor.cleanup()

    def create_python_executor(self) -> PythonExecutor:
        """创建 Python 代码执行器。
        
        根据 executor_type 选择不同的执行环境：
        - local: 本地沙箱执行（默认，最快但安全性较低）
        - e2b: E2B 云沙箱（安全，需要 API key）
        - docker: Docker 容器（安全，需要 Docker 环境）
        - modal: Modal 云函数（安全，需要 Modal 账号）
        - wasm: WebAssembly 沙箱（安全，浏览器兼容）
        - blaxel: Blaxel 云沙箱（安全，需要 API key）
        
        Returns:
            PythonExecutor: 配置好的代码执行器实例
            
        Raises:
            ValueError: 如果 executor_type 不支持
            Exception: 如果远程执行器不支持 managed_agents
        """
        if self.executor_type not in {"local", "blaxel", "e2b", "modal", "docker", "wasm"}:
            raise ValueError(f"Unsupported executor type: {self.executor_type}")

        if self.executor_type == "local":
            # 本地执行器：在当前进程中运行代码（通过 RestrictedPython 限制危险操作）
            return LocalPythonExecutor(
                self.additional_authorized_imports,
                **{"max_print_outputs_length": self.max_print_outputs_length} | self.executor_kwargs,
            )
        else:
            # 远程执行器：在隔离环境中运行代码（更安全但有网络延迟）
            if self.managed_agents:
                # 限制：远程执行器暂不支持子 Agent（因为需要序列化整个 Agent 对象）
                raise Exception("Managed agents are not yet supported with remote code execution.")
            remote_executors = {
                "blaxel": BlaxelExecutor,
                "e2b": E2BExecutor,
                "docker": DockerExecutor,
                "wasm": WasmExecutor,
                "modal": ModalExecutor,
            }
            return remote_executors[self.executor_type](
                self.additional_authorized_imports, self.logger, **self.executor_kwargs
            )

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "authorized_imports": (
                    "You can import from any package you want."
                    if "*" in self.authorized_imports
                    else str(self.authorized_imports)
                ),
                "custom_instructions": self.instructions,
                "code_block_opening_tag": self.code_block_tags[0],
                "code_block_closing_tag": self.code_block_tags[1],
            },
        )
        return system_prompt

    def _step_stream(
        self, memory_step: ActionStep
    ) -> Generator[ChatMessageStreamDelta | ToolCall | ToolOutput | ActionOutput]:
        """执行一步 ReAct 循环：思考 → 行动 → 观察。
        
        这是 CodeAgent 的核心方法，负责：
        1. 调用 LLM 生成 Python 代码
        2. 从 LLM 输出中解析代码块
        3. 在沙箱中执行代码
        4. 收集执行结果并写入记忆
        
        执行流程：
        阶段一：LLM 生成代码
            - 将历史记忆转为消息列表
            - 调用 LLM（支持流式输出）
            - 记录 token 使用量
        阶段二：解析代码块
            - 从 LLM 输出中提取代码（<code>...</code> 或 ```python...```）
            - 修正 final_answer 的写法（如 return xxx → final_answer(xxx)）
        阶段三：执行代码
            - 在 Python 沙箱中运行代码
            - 捕获执行日志和输出
            - 检查是否调用了 final_answer()
        
        Args:
            memory_step: 当前步骤的记忆对象，用于记录输入输出
            
        Yields:
            ChatMessageStreamDelta: LLM 流式输出的 token（如果启用流式）
            ToolCall: 代码执行请求（工具名为 "python_interpreter"）
            ActionOutput: 代码执行结果（包含 is_final_answer 标志）
        """
        # 将历史记忆转换为 LLM 可理解的消息列表
        memory_messages = self.write_memory_to_messages()

        input_messages = memory_messages.copy()
        
        # ========== 阶段一：调用 LLM 生成代码 ==========
        memory_step.model_input_messages = input_messages
        
        # 停止序列：当 LLM 输出这些字符串时立即停止，防止生成过多内容
        # 1. "Observation:" / "Calling tools:" - 防止 LLM 自己模拟观察结果
        # 2. 代码块关闭标签（如 "```"）- 避免 LLM 继续生成解释文字
        stop_sequences = ["Observation:", "Calling tools:"]
        if self.code_block_tags[1] not in self.code_block_tags[0]:
            # 仅当关闭标签不是开放标签的子串时才添加（避免误截断）
            # 例如：("<code>", "</code>") 可以添加，但 ("```python", "```") 不能添加 "```"
            stop_sequences.append(self.code_block_tags[1])
            
        try:
            additional_args: dict[str, Any] = {}
            # 如果启用结构化输出，强制 LLM 输出 {"thought": "...", "code": "..."} 格式
            if self._use_structured_outputs_internally:
                additional_args["response_format"] = CODEAGENT_RESPONSE_FORMAT
                
            if self.stream_outputs:
                # 流式模式：逐 token 生成并实时显示
                output_stream = self.model.generate_stream(
                    input_messages,
                    stop_sequences=stop_sequences,
                    **additional_args,
                )
                chat_message_stream_deltas: list[ChatMessageStreamDelta] = []
                # 使用 Rich Live 实时渲染 Markdown
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in output_stream:
                        chat_message_stream_deltas.append(event)
                        # 将所有增量合并后渲染为 Markdown
                        live.update(
                            Markdown(agglomerate_stream_deltas(chat_message_stream_deltas).render_as_markdown())
                        )
                        yield event  # yield 每个 token 增量
                # 将所有增量合并为完整消息
                chat_message = agglomerate_stream_deltas(chat_message_stream_deltas)
                memory_step.model_output_message = chat_message
                output_text = chat_message.content
            else:
                # 非流式模式：一次性生成完整输出
                chat_message: ChatMessage = self.model.generate(
                    input_messages,
                    stop_sequences=stop_sequences,
                    **additional_args,
                )
                memory_step.model_output_message = chat_message
                output_text = chat_message.content
                self.logger.log_markdown(
                    content=output_text or "",
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )

            if not self._use_structured_outputs_internally:
                # 技巧：在输出末尾添加代码块关闭标签（如果缺失）
                # 这会让后续 LLM 调用学会以此标签结束，从而高效停止生成
                if output_text and not output_text.strip().endswith(self.code_block_tags[1]):
                    output_text += self.code_block_tags[1]
                    memory_step.model_output_message.content = output_text

            memory_step.token_usage = chat_message.token_usage
            memory_step.model_output = output_text
        except Exception as e:
            raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

        # ========== 阶段二：从 LLM 输出中解析代码块 ==========
        try:
            if self._use_structured_outputs_internally:
                # 结构化输出模式：直接从 JSON 中取 "code" 字段
                code_action = json.loads(output_text)["code"]
                # 尝试进一步提取代码块（如果 code 字段中还包含标签）
                code_action = extract_code_from_text(code_action, self.code_block_tags) or code_action
            else:
                # 普通模式：从文本中提取代码块标签之间的内容
                # 支持 <code>...</code> 或 ```python...``` 格式
                code_action = parse_code_blobs(output_text, self.code_block_tags)
            # 修正 final_answer 的写法（如 return xxx → final_answer(xxx)）
            code_action = fix_final_answer_code(code_action)
            memory_step.code_action = code_action
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            raise AgentParsingError(error_msg, self.logger)

        # 创建工具调用对象（代码执行被视为调用 "python_interpreter" 工具）
        tool_call = ToolCall(
            name="python_interpreter",
            arguments=code_action,
            id=f"call_{len(self.memory.steps)}",  # 使用步骤数作为唯一 ID
        )
        yield tool_call
        memory_step.tool_calls = [tool_call]

        # ========== 阶段三：在沙箱中执行代码 ==========
        self.logger.log_code(title="Executing parsed code:", content=code_action, level=LogLevel.INFO)
        try:
            # python_executor 是 LocalPythonExecutor 或远程执行器（E2B/Docker/Modal/Wasm）
            # 返回 CodeOutput(output, logs, is_final_answer)
            code_output = self.python_executor(code_action)
            execution_outputs_console = []
            # 收集执行日志（print 输出、警告等）
            if len(code_output.logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(code_output.logs),
                ]
            observation = "Execution logs:\n" + code_output.logs
        except Exception as e:
            # 代码执行失败：尝试获取部分日志（如果有）
            if hasattr(self.python_executor, "state") and "_print_outputs" in self.python_executor.state:
                execution_logs = str(self.python_executor.state["_print_outputs"])
                if len(execution_logs) > 0:
                    execution_outputs_console = [
                        Text("Execution logs:", style="bold"),
                        Text(execution_logs),
                    ]
                    memory_step.observations = "Execution logs:\n" + execution_logs
                    self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
            error_msg = str(e)
            # 特殊提示：如果是 import 权限错误，提醒用户添加授权
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "[bold red]Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent.",
                    level=LogLevel.INFO,
                )
            raise AgentExecutionError(error_msg, self.logger)

        # 截断过长的输出（防止超出上下文窗口）
        truncated_output = truncate_content(str(code_output.output))
        observation += "Last output from code snippet:\n" + truncated_output
        memory_step.observations = observation

        # 准备控制台输出
        if not code_output.is_final_answer:
            execution_outputs_console += [
                Text(
                    f"Out: {truncated_output}",
                ),
            ]
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        memory_step.action_output = code_output.output
        
        # yield 最终的行动输出（包含 is_final_answer 标志）
        yield ActionOutput(output=code_output.output, is_final_answer=code_output.is_final_answer)

    def to_dict(self) -> dict[str, Any]:
        """Convert the agent to a dictionary representation.

        Returns:
            `dict`: Dictionary representation of the agent.
        """
        agent_dict = super().to_dict()
        agent_dict["authorized_imports"] = self.authorized_imports
        agent_dict["executor_type"] = self.executor_type
        agent_dict["executor_kwargs"] = self.executor_kwargs
        agent_dict["max_print_outputs_length"] = self.max_print_outputs_length
        return agent_dict

    @classmethod
    def from_dict(cls, agent_dict: dict[str, Any], **kwargs) -> "CodeAgent":
        """Create CodeAgent from a dictionary representation.

        Args:
            agent_dict (`dict[str, Any]`): Dictionary representation of the agent.
            **kwargs: Additional keyword arguments that will override agent_dict values.

        Returns:
            `CodeAgent`: Instance of the CodeAgent class.
        """
        # Add CodeAgent-specific parameters to kwargs
        code_agent_kwargs = {
            "additional_authorized_imports": agent_dict.get("authorized_imports"),
            "executor_type": agent_dict.get("executor_type"),
            "executor_kwargs": agent_dict.get("executor_kwargs"),
            "max_print_outputs_length": agent_dict.get("max_print_outputs_length"),
            "code_block_tags": agent_dict.get("code_block_tags"),
        }
        # Filter out None values
        code_agent_kwargs = {k: v for k, v in code_agent_kwargs.items() if v is not None}
        # Update with any additional kwargs
        code_agent_kwargs.update(kwargs)
        # Call the parent class's from_dict method
        return super().from_dict(agent_dict, **code_agent_kwargs)


# Agent Registry for secure deserialization
# This registry maps agent class names to their actual classes.
# Only classes listed here can be instantiated during deserialization (from_dict/from_folder).
# This prevents arbitrary code execution via importlib-based dynamic loading.
AGENT_REGISTRY = {
    "ToolCallingAgent": ToolCallingAgent,
    "CodeAgent": CodeAgent,
}
