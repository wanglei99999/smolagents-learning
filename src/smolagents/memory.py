# =============================================================================
# memory.py —— Agent 记忆系统
#
# 本文件定义了 smolagents 的记忆模型，即 Agent 在 ReAct 循环中如何记录每一步的状态。
#
# 核心概念：
#   AgentMemory 是 Agent 的"对话历史 + 执行日志"合体，
#   每一步执行完后，Agent 将对应的 MemoryStep 追加到 memory.steps 中。
#   下一步调用 LLM 时，memory.write_memory_to_messages() 将所有步骤序列化为消息列表，
#   作为 LLM 的上下文输入（这就是 Agent 能"记住"之前做了什么的原因）。
#
# 记忆步骤类型（MemoryStep 的子类）：
#
#   SystemPromptStep  ← 系统提示词（每次 run 开始时设置，不重复追加）
#   TaskStep          ← 用户任务描述（一次 run 只有一个）
#   PlanningStep      ← 规划步骤（planning_interval 触发时生成，包含计划文本）
#   ActionStep        ← 行动步骤（ReAct 循环中最核心的步骤，每步一个）
#       - model_input_messages: 本步 LLM 收到的消息列表（完整上下文）
#       - model_output: LLM 的原始输出文本（思考 + 行动代码 / 工具调用）
#       - code_action: 解析出的代码（CodeAgent 专用）
#       - tool_calls: 工具调用列表（ToolCallingAgent 专用）
#       - observations: 执行结果 / 工具输出（写回历史供 LLM 下步参考）
#       - error: 执行错误（若有，LLM 下步会尝试修正）
#       - token_usage: 本步消耗的 Token 数
#   FinalAnswerStep   ← 最终答案（循环结束时生成，不写入对话历史）
#
# 序列化（summary_mode）：
#   summary_mode=True 时，ActionStep.to_messages() 会省略 model_output（减少 token 消耗）
#   summary_mode=True 时，PlanningStep.to_messages() 返回空列表（避免旧计划干扰新计划）
#   summary_mode=True 时，SystemPromptStep.to_messages() 返回空列表
#
# CallbackRegistry：
#   管理步骤回调函数（step_callbacks），每步完成后按类型触发对应的回调。
# =============================================================================

import inspect
from dataclasses import asdict, dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable, Type

from smolagents.models import ChatMessage, MessageRole, get_dict_from_nested_dataclasses
from smolagents.monitoring import AgentLogger, LogLevel, Timing, TokenUsage
from smolagents.utils import AgentError, make_json_serializable


if TYPE_CHECKING:
    import PIL.Image

    from smolagents.models import ChatMessage
    from smolagents.monitoring import AgentLogger


__all__ = ["AgentMemory"]


logger = getLogger(__name__)


@dataclass
class ToolCall:
    """记录 Agent 执行的一次工具调用请求（记忆中的持久化版本）。
    与 models.py 中的 ChatMessageToolCall 的区别：
      - ChatMessageToolCall 是 LLM API 返回的原始格式（用于 API 交互）
      - memory.ToolCall 是写入记忆的序列化版本（用于记录和回放）
    """
    name: str      # 工具名称
    arguments: Any # 工具参数（dict）
    id: str        # 唯一 ID，用于关联 ToolCallingAgent 的并行调用

    def dict(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": make_json_serializable(self.arguments),
            },
        }


@dataclass
class MemoryStep:
    def dict(self):
        return asdict(self)

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        raise NotImplementedError


@dataclass
class ActionStep(MemoryStep):
    """一步完整的 ReAct 行动记录（记忆中最核心的步骤类型）。
    
    ActionStep 是 ReAct 循环中最重要的数据结构，它记录了一轮完整的 Think-Act-Observe 过程。
    每个 ActionStep 包含：
    1. Think（思考）：model_output - LLM 的推理过程
    2. Act（行动）：tool_calls/code_action - 要执行的工具调用或代码
    3. Observe（观察）：observations - 执行结果
    
    这个类承担双重角色：
    - 运行记录：保存执行历史，用于调试和回放
    - 下一轮输入：通过 to_messages() 转换为 LLM 的上下文
    
    字段说明：
      step_number: 步骤编号（从 1 开始）
      timing: 执行时间统计（开始时间、结束时间、耗时）
      model_input_messages: 本步发给 LLM 的完整消息列表（包含历史）
      model_output_message: LLM 返回的完整消息对象（包含 tool_calls 等元数据）
      model_output: LLM 的原始输出文本（含思考和行动描述）
      code_action: 从 model_output 解析出的 Python 代码（CodeAgent 专用）
      tool_calls: 解析出的工具调用列表（ToolCallingAgent 专用）
      observations: 代码/工具执行后的输出结果字符串（写入下一步的上下文）⭐ ReAct 的关键
      observations_images: 多模态输入的图像（每步都会传入）
      action_output: 代码执行的 Python 返回值（未序列化的原始对象）
      error: 执行错误（LLM 下一步会看到错误并尝试修正）
      token_usage: 本步的 token 使用统计（输入 token 数、输出 token 数）
      is_final_answer: 是否是最后一步（调用了 final_answer 工具/函数）
    """
    # === 基础信息 ===
    step_number: int  # 步骤编号，用于标识这是第几轮 ReAct 循环
    timing: Timing  # 时间统计：start_time, end_time, duration
    
    # === Think（思考）阶段的数据 ===
    model_input_messages: list[ChatMessage] | None = None  # 发给 LLM 的输入（包含历史上下文）
    model_output_message: ChatMessage | None = None  # LLM 返回的完整消息对象
    model_output: str | list[dict[str, Any]] | None = None  # LLM 的文本输出（推理过程）
    
    # === Act（行动）阶段的数据 ===
    tool_calls: list[ToolCall] | None = None  # ToolCallingAgent：解析出的工具调用列表
    code_action: str | None = None  # CodeAgent：解析出的 Python 代码
    
    # === Observe（观察）阶段的数据 ===
    observations: str | None = None  # 执行结果（字符串格式），会被写入下一轮的 LLM 输入
    observations_images: list["PIL.Image.Image"] | None = None  # 多模态：图像输入
    action_output: Any = None  # 原始的 Python 对象（未序列化）
    error: AgentError | None = None  # 执行错误（也是观察的一部分）
    
    # === 元数据 ===
    token_usage: TokenUsage | None = None  # Token 使用统计（用于成本计算）
    is_final_answer: bool = False  # 是否调用了 final_answer（标记循环结束）

    def dict(self):
        """将 ActionStep 序列化为字典（用于保存和传输）。
        
        这个方法手动处理复杂字段的序列化：
        - tool_calls: 转换为字典列表
        - action_output: 使用 make_json_serializable 处理任意 Python 对象
        - model_input_messages: 递归序列化嵌套的 dataclass
        - observations_images: 转换为字节数据
        
        Returns:
            dict: 可 JSON 序列化的字典
        """
        return {
            "step_number": self.step_number,
            "timing": self.timing.dict(),
            "model_input_messages": [
                make_json_serializable(get_dict_from_nested_dataclasses(msg)) for msg in self.model_input_messages
            ]
            if self.model_input_messages
            else None,
            "tool_calls": [tc.dict() for tc in self.tool_calls] if self.tool_calls else [],
            "error": self.error.dict() if self.error else None,
            "model_output_message": make_json_serializable(get_dict_from_nested_dataclasses(self.model_output_message))
            if self.model_output_message
            else None,
            "model_output": self.model_output,
            "code_action": self.code_action,
            "observations": self.observations,
            "observations_images": [image.tobytes() for image in self.observations_images]
            if self.observations_images
            else None,
            "action_output": make_json_serializable(self.action_output),
            "token_usage": asdict(self.token_usage) if self.token_usage else None,
            "is_final_answer": self.is_final_answer,
        }

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        """⭐ ReAct 框架的核心方法：将此步骤转换为 LLM 的输入消息。
        
        这个方法实现了 ReAct 的"反馈闭环"：
        1. 将本步的 observations（观察结果）转换为消息
        2. 下一轮 LLM 调用时会看到这些消息
        3. LLM 基于观察结果调整策略
        
        消息顺序（对应 ReAct 的三个阶段）：
        1. Think: model_output（LLM 的推理过程）
        2. Act: tool_calls（工具调用或代码）
        3. Observe: observations（执行结果）或 error（错误信息）
        
        Args:
            summary_mode: 是否使用摘要模式
                - False（默认）：完整输出，包含 model_output
                - True：省略 model_output，只保留工具调用和观察（用于规划时减少 token）
        
        Returns:
            list[ChatMessage]: 消息列表，会被添加到下一轮 LLM 的输入中
        
        Example:
            >>> action_step = ActionStep(
            ...     model_output="我需要搜索天气",
            ...     tool_calls=[ToolCall(name="web_search", ...)],
            ...     observations="Paris: 20°C"
            ... )
            >>> messages = action_step.to_messages()
            >>> # messages 包含：
            >>> # 1. {"role": "assistant", "content": "我需要搜索天气"}
            >>> # 2. {"role": "tool_call", "content": "Calling tools: ..."}
            >>> # 3. {"role": "tool_response", "content": "Observation: Paris: 20°C"}
        """
        messages = []
        
        # === 1. Think（思考）：LLM 的推理过程 ===
        # 在摘要模式下省略，因为推理过程通常很长，且对下一步不是必需的
        if self.model_output is not None and not summary_mode:
            messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.model_output.strip()}])
            )

        # === 2. Act（行动）：工具调用或代码 ===
        # 告诉 LLM "我调用了哪些工具"
        if self.tool_calls is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL_CALL,
                    content=[
                        {
                            "type": "text",
                            "text": "Calling tools:\n" + str([tc.dict() for tc in self.tool_calls]),
                        }
                    ],
                )
            )

        # === 多模态支持：图像输入 ===
        if self.observations_images:
            messages.append(
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "image",
                            "image": image,
                        }
                        for image in self.observations_images
                    ],
                )
            )

        # === 3. Observe（观察）：执行结果 ===
        # ⭐ 这是 ReAct 闭环的关键：将执行结果传递给下一轮 LLM
        if self.observations is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[
                        {
                            "type": "text",
                            "text": f"Observation:\n{self.observations}",
                        }
                    ],
                )
            )
        
        # === 错误也是观察的一部分 ===
        # 让 LLM 看到错误信息，从而能够自我纠错
        if self.error is not None:
            error_message = (
                "Error:\n"
                + str(self.error)
                + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
            )
            message_content = f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else ""
            message_content += error_message
            messages.append(
                ChatMessage(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": message_content}])
            )

        return messages


@dataclass
class PlanningStep(MemoryStep):
    """规划步骤记录（planning_interval 触发时生成）。
    plan: LLM 生成的计划文本（格式："Here are the facts I know and the plan..."）
    注意：to_messages(summary_mode=True) 返回空列表，
    这样在生成新计划时不会受旧计划影响（避免路径依赖）。
    """
    model_input_messages: list[ChatMessage]
    model_output_message: ChatMessage
    plan: str
    timing: Timing
    token_usage: TokenUsage | None = None

    def dict(self):
        return {
            "model_input_messages": [
                make_json_serializable(get_dict_from_nested_dataclasses(msg)) for msg in self.model_input_messages
            ],
            "model_output_message": make_json_serializable(
                get_dict_from_nested_dataclasses(self.model_output_message)
            ),
            "plan": self.plan,
            "timing": self.timing.dict(),
            "token_usage": asdict(self.token_usage) if self.token_usage else None,
        }

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        if summary_mode:
            return []
        return [
            ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.plan.strip()}]),
            ChatMessage(
                role=MessageRole.USER, content=[{"type": "text", "text": "Now proceed and carry out this plan."}]
            ),
            # This second message creates a role change to prevent models models from simply continuing the plan message
        ]


@dataclass
class TaskStep(MemoryStep):
    task: str
    task_images: list["PIL.Image.Image"] | None = None

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        content = [{"type": "text", "text": f"New task:\n{self.task}"}]
        if self.task_images:
            content.extend([{"type": "image", "image": image} for image in self.task_images])

        return [ChatMessage(role=MessageRole.USER, content=content)]


@dataclass
class SystemPromptStep(MemoryStep):
    system_prompt: str

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        if summary_mode:
            return []
        return [ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}])]


@dataclass
class FinalAnswerStep(MemoryStep):
    output: Any


class AgentMemory:
    """Memory for the agent, containing the system prompt and all steps taken by the agent.

    This class is used to store the agent's steps, including tasks, actions, and planning steps.
    It allows for resetting the memory, retrieving succinct or full step information, and replaying the agent's steps.

    Args:
        system_prompt (`str`): System prompt for the agent, which sets the context and instructions for the agent's behavior.

    **Attributes**:
        - **system_prompt** (`SystemPromptStep`) -- System prompt step for the agent.
        - **steps** (`list[TaskStep | ActionStep | PlanningStep]`) -- List of steps taken by the agent, which can include tasks, actions, and planning steps.
    """

    def __init__(self, system_prompt: str):
        self.system_prompt: SystemPromptStep = SystemPromptStep(system_prompt=system_prompt)
        self.steps: list[TaskStep | ActionStep | PlanningStep] = []

    def reset(self):
        """Reset the agent's memory, clearing all steps and keeping the system prompt."""
        self.steps = []

    def get_succinct_steps(self) -> list[dict]:
        """Return a succinct representation of the agent's steps, excluding model input messages."""
        return [
            {key: value for key, value in step.dict().items() if key != "model_input_messages"} for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        """Return a full representation of the agent's steps, including model input messages."""
        if len(self.steps) == 0:
            return []
        return [step.dict() for step in self.steps]

    def replay(self, logger: AgentLogger, detailed: bool = False):
        """Prints a pretty replay of the agent's steps.

        Args:
            logger (`AgentLogger`): The logger to print replay logs to.
            detailed (`bool`, default `False`): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        logger.console.log("Replaying the agent's steps:")
        logger.log_markdown(title="System prompt", content=self.system_prompt.system_prompt, level=LogLevel.ERROR)
        for step in self.steps:
            if isinstance(step, TaskStep):
                logger.log_task(step.task, "", level=LogLevel.ERROR)
            elif isinstance(step, ActionStep):
                logger.log_rule(f"Step {step.step_number}", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                if step.model_output is not None:
                    logger.log_markdown(title="Agent output:", content=step.model_output, level=LogLevel.ERROR)
            elif isinstance(step, PlanningStep):
                logger.log_rule("Planning step", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                logger.log_markdown(title="Agent output:", content=step.plan, level=LogLevel.ERROR)

    def return_full_code(self) -> str:
        """Returns all code actions from the agent's steps, concatenated as a single script."""
        return "\n\n".join(
            [step.code_action for step in self.steps if isinstance(step, ActionStep) and step.code_action is not None]
        )


class CallbackRegistry:
    """Registry for callbacks that are called at each step of the agent's execution.

    Callbacks are registered by passing a step class and a callback function.
    """

    def __init__(self):
        self._callbacks: dict[Type[MemoryStep], list[Callable]] = {}

    def register(self, step_cls: Type[MemoryStep], callback: Callable):
        """Register a callback for a step class.

        Args:
            step_cls (Type[MemoryStep]): Step class to register the callback for.
            callback (Callable): Callback function to register.
        """
        if step_cls not in self._callbacks:
            self._callbacks[step_cls] = []
        self._callbacks[step_cls].append(callback)

    def callback(self, memory_step, **kwargs):
        """Call callbacks registered for a step type.

        Args:
            memory_step (MemoryStep): Step to call the callbacks for.
            **kwargs: Additional arguments to pass to callbacks that accept them.
                Typically, includes the agent instance.

        Notes:
            For backwards compatibility, callbacks with a single parameter signature
            receive only the memory_step, while callbacks with multiple parameters
            receive both the memory_step and any additional kwargs.
        """
        # For compatibility with old callbacks that only take the step as an argument
        for cls in memory_step.__class__.__mro__:
            for cb in self._callbacks.get(cls, []):
                cb(memory_step) if len(inspect.signature(cb).parameters) == 1 else cb(memory_step, **kwargs)
