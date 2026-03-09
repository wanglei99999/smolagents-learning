# 演示如何使用不同的 LLM 后端（推理提供商）来驱动同一个 Agent
# smolagents 是模型无关的，只需切换 Model 类即可

from smolagents import (
    CodeAgent,             # 代码型 Agent：LLM 生成 Python 代码并执行
    InferenceClientModel,  # 使用 HuggingFace 推理 API（云端，最简单）
    LiteLLMModel,          # 使用 LiteLLM 统一接口，支持 OpenAI/Anthropic/Ollama 等
    OpenAIModel,           # 使用 OpenAI 官方 API
    ToolCallingAgent,      # 工具调用型 Agent：LLM 以 JSON 格式调用工具（OpenAI 风格）
    TransformersModel,     # 使用本地 HuggingFace Transformers 模型（离线）
    tool,                  # 将普通函数注册为 Agent 可用工具的装饰器
)


# ============================================================
# 第一步：选择推理后端（修改 chosen_inference 来切换）
# ============================================================

# 所有可用的推理后端列表（仅供参考，实际由 chosen_inference 决定）
available_inferences = ["inference_client", "transformers", "ollama", "litellm", "openai"]

# 当前选择的推理后端，修改此变量即可切换模型
chosen_inference = "inference_client"

print(f"Chose model: '{chosen_inference}'")

# --- 方案1：HuggingFace 推理 API（需要 HF_TOKEN 环境变量）---
# 优点：无需本地 GPU，直接调用云端模型
if chosen_inference == "inference_client":
    model = InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct", provider="nebius")

# --- 方案2：本地 Transformers 模型（需要本地 GPU/CPU）---
# 优点：完全离线，数据不出本地
# device_map="auto" 会自动分配到可用的 GPU/CPU
# max_new_tokens 控制每次生成的最大 token 数
elif chosen_inference == "transformers":
    model = TransformersModel(model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct", device_map="auto", max_new_tokens=1000)

# --- 方案3：本地 Ollama 服务（需要先启动 ollama serve）---
# 优点：本地运行，通过 OpenAI 兼容接口访问
# num_ctx 是上下文窗口大小，ollama 默认 2048 太小，推荐 8192 以上
elif chosen_inference == "ollama":
    model = LiteLLMModel(
        model_id="ollama_chat/llama3.2",
        api_base="http://localhost:11434",  # replace with remote open-ai compatible server if necessary
        api_key="your-api-key",  # replace with API key if necessary
        num_ctx=8192,  # ollama default is 2048 which will often fail horribly. 8192 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
    )

# --- 方案4：通过 LiteLLM 调用任意云端模型（需要对应 API Key 环境变量）---
# 支持 OpenAI、Anthropic、Gemini、Cohere 等几乎所有主流模型
# 切换 Anthropic：将 model_id 改为 'anthropic/claude-3-5-sonnet-latest'
elif chosen_inference == "litellm":
    # For anthropic: change model_id below to 'anthropic/claude-3-5-sonnet-latest'
    model = LiteLLMModel(model_id="gpt-4o")

# --- 方案5：直接使用 OpenAI 官方 API（需要 OPENAI_API_KEY 环境变量）---
# 与 LiteLLMModel 类似，但专门针对 OpenAI 做了优化
elif chosen_inference == "openai":
    # For anthropic: change model_id below to 'anthropic/claude-3-5-sonnet-latest'
    model = OpenAIModel(model_id="gpt-4o")


# ============================================================
# 第二步：定义工具（Tool）
# ============================================================

# @tool 装饰器将普通函数注册为 Agent 可调用的工具
# 框架会自动从以下内容提取工具描述传给 LLM：
#   - 函数名       → 工具名称
#   - docstring    → 工具功能描述（LLM 据此决定何时调用）
#   - Args 部分    → 每个参数的说明（必须写！否则注册失败）
#   - 类型注解     → 参数类型（用于生成 JSON Schema，必须写！）
@tool
def get_weather(location: str, celsius: bool | None = False) -> str:
    """
    Get weather in the next days at given location.
    Secretly this tool does not care about the location, it hates the weather everywhere.

    Args:
        location: the location
        celsius: the temperature
    """
    # 注意：这是一个演示工具，实际上忽略了所有参数，总是返回固定结果
    # 真实场景中应在这里调用天气 API
    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"


# ============================================================
# 第三步：创建 Agent 并运行
# ============================================================

# --- 演示1：ToolCallingAgent ---
# 工作方式：LLM 输出结构化 JSON 来调用工具，例如：
#   {"tool": "get_weather", "arguments": {"location": "Paris"}}
# 适合：简单的单工具调用场景，与 OpenAI function calling 风格一致
# verbosity_level=2 会打印每一步的详细日志（0=静默, 1=基本, 2=详细）
agent = ToolCallingAgent(tools=[get_weather], model=model, verbosity_level=2)

print("ToolCallingAgent:", agent.run("What's the weather like in Paris?"))

# --- 演示2：CodeAgent ---
# 工作方式：LLM 生成完整的 Python 代码并在沙箱中执行，例如：
#   result = get_weather(location="Paris", celsius=True)
#   final_answer(result)
# 适合：需要多步推理、条件判断、循环等复杂逻辑的场景（通常优于 ToolCallingAgent）
# stream_outputs=True 会实时流式输出 LLM 生成的内容（需要模型支持 generate_stream）
agent = CodeAgent(tools=[get_weather], model=model, verbosity_level=2, stream_outputs=True)

print("CodeAgent:", agent.run("What's the weather like in Paris?"))
