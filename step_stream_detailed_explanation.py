#!/usr/bin/env python3
"""
_step_stream 方法详解：ToolCallingAgent 的核心执行逻辑

这个文件详细解释了 _step_stream 方法的完整执行流程，
包括每个步骤的作用、设计思想和实际应用场景。
"""

def explain_method_overview():
    """
    解释方法的整体概览
    """
    print("=" * 80)
    print("🔄 _step_stream 方法详解")
    print("=" * 80)
    
    print("\n📋 方法职责：")
    print("-" * 50)
    print("这是 ToolCallingAgent 执行 ReAct 循环的核心方法，负责：")
    print("1. **Reasoning（推理）**：LLM 分析任务并决定下一步行动")
    print("2. **Acting（行动）**：解析并执行 LLM 请求的工具调用")
    print("3. **Observing（观察）**：收集工具执行结果并准备下一轮推理")
    
    print("\n🎯 执行流程：")
    print("-" * 50)
    print("输入 → 准备消息 → LLM生成 → 解析工具调用 → 执行工具 → 返回结果")
    
    print("\n🔍 返回类型（Generator）：")
    print("-" * 50)
    print("• ChatMessageStreamDelta: 流式输出的文本片段")
    print("• ToolCall: 工具调用请求")
    print("• ToolOutput: 工具执行结果")
    print("• ActionOutput: 最终的行动输出")

def explain_step1_message_preparation():
    """
    解释第1步：消息准备
    """
    print("\n" + "=" * 80)
    print("📝 第1步：准备输入消息")
    print("=" * 80)
    
    print("\n🔍 代码分析：")
    print("-" * 30)
    print("""
    memory_messages = self.write_memory_to_messages()
    input_messages = memory_messages.copy()
    memory_step.model_input_messages = input_messages
    """)
    
    print("\n💡 作用解释：")
    print("-" * 30)
    print("• **write_memory_to_messages()**: 将记忆系统中的历史对话转换为 LLM API 格式")
    print("• **copy()**: 创建副本，避免修改原始记忆数据")
    print("• **记录输入**: 保存到 memory_step 中，用于调试和回放")
    
    print("\n🎯 实际场景：")
    print("-" * 30)
    print("""
    # 记忆系统中的对话历史
    [
        TaskStep(task="帮我搜索天气并计算温度转换"),
        ActionStep(tool_name="web_search", result="今天北京25°C"),
        ActionStep(tool_name="calculator", result="77°F")
    ]
    
    # 转换为 LLM API 格式
    [
        {"role": "system", "content": "你是一个智能助手..."},
        {"role": "user", "content": "帮我搜索天气并计算温度转换"},
        {"role": "assistant", "content": "我来帮你搜索天气..."},
        {"role": "tool", "content": "今天北京25°C"},
        ...
    ]
    """)

def explain_step2_generation_modes():
    """
    解释第2步：生成模式选择
    """
    print("\n" + "=" * 80)
    print("⚡ 第2步：选择生成模式")
    print("=" * 80)
    
    print("\n🔍 流式 vs 非流式对比：")
    print("-" * 30)
    print("| 特性         | 流式模式                | 非流式模式              |")
    print("|-------------|------------------------|------------------------|")
    print("| 用户体验     | 实时显示，类似打字机效果  | 等待完整响应后显示      |")
    print("| 适用场景     | UI界面、交互式应用      | 批处理、API调用        |")
    print("| 资源使用     | 持续占用连接            | 一次性请求             |")
    print("| 调试便利性   | 可以看到思考过程        | 只能看到最终结果       |")
    
    print("\n🎯 流式模式详解：")
    print("-" * 30)
    print("""
    # 流式输出的工作原理
    output_stream = self.model.generate_stream(
        input_messages,
        stop_sequences=["Observation:", "Calling tools:"],  # 停止词
        tools_to_call_from=self.tools_and_managed_agents,   # 可用工具
    )
    
    # 实时显示效果
    chat_message_stream_deltas = []
    with Live("", console=self.logger.console) as live:
        for event in output_stream:
            chat_message_stream_deltas.append(event)
            # 实时更新显示（类似 ChatGPT 的打字效果）
            live.update(Markdown(...))
            yield event  # 返回给调用方
    """)
    
    print("\n🛑 停止序列的作用：")
    print("-" * 30)
    print("• **'Observation:'**: 防止 LLM 自己编造观察结果")
    print("• **'Calling tools:'**: 控制工具调用的格式边界")
    print("• 确保 LLM 在合适的时机停止生成，等待真实的工具执行结果")

def explain_step3_error_handling():
    """
    解释第3步：错误处理
    """
    print("\n" + "=" * 80)
    print("🚨 第3步：错误处理机制")
    print("=" * 80)
    
    print("\n🔍 代码分析：")
    print("-" * 30)
    print("""
    try:
        # LLM 生成逻辑
        ...
    except Exception as e:
        raise AgentGenerationError(f"Error while generating output:\\n{e}", self.logger) from e
    """)
    
    print("\n💡 错误类型和处理：")
    print("-" * 30)
    print("• **网络错误**: API 调用失败、超时等")
    print("• **模型错误**: 模型服务不可用、配额超限等")
    print("• **格式错误**: 返回格式不符合预期")
    print("• **AgentGenerationError**: 统一的错误类型，便于上层处理")
    
    print("\n🎯 错误恢复策略：")
    print("-" * 30)
    print("• 记录详细错误信息到日志")
    print("• 保留原始异常链（from e）")
    print("• 提供给上层重试或降级处理的机会")

def explain_step4_tool_call_parsing():
    """
    解释第4步：工具调用解析
    """
    print("\n" + "=" * 80)
    print("🔧 第4步：工具调用解析")
    print("=" * 80)
    
    print("\n🔍 两种解析模式：")
    print("-" * 30)
    print("""
    # 模式1：原生支持（现代模型如 GPT-4）
    if chat_message.tool_calls is None or len(chat_message.tool_calls) == 0:
        # 模式2：文本解析（老版本模型）
        chat_message = self.model.parse_tool_calls(chat_message)
    else:
        # 处理参数格式（JSON字符串 → dict）
        for tool_call in chat_message.tool_calls:
            tool_call.function.arguments = parse_json_if_needed(...)
    """)
    
    print("\n💡 原生支持 vs 文本解析：")
    print("-" * 30)
    print("**原生支持（推荐）：**")
    print("• 模型直接返回结构化的工具调用对象")
    print("• 格式标准、解析可靠")
    print("• 支持并行工具调用")
    
    print("\n**文本解析（兼容模式）：**")
    print("• 从模型的文本输出中提取工具调用")
    print("• 需要复杂的正则表达式或解析逻辑")
    print("• 容易出现格式错误")
    
    print("\n🎯 实际示例：")
    print("-" * 30)
    print("""
    # 原生格式
    tool_calls = [
        {
            "id": "call_123",
            "function": {
                "name": "web_search",
                "arguments": {"query": "北京天气"}
            }
        }
    ]
    
    # 文本格式（需要解析）
    "我需要搜索天气信息。
    
    ```json
    {
        "tool": "web_search",
        "arguments": {"query": "北京天气"}
    }
    ```"
    """)

def explain_step5_tool_execution():
    """
    解释第5步：工具执行和结果处理
    """
    print("\n" + "=" * 80)
    print("⚙️ 第5步：工具执行和结果处理")
    print("=" * 80)
    
    print("\n🔍 核心执行循环：")
    print("-" * 30)
    print("""
    final_answer, got_final_answer = None, False
    
    for output in self.process_tool_calls(chat_message, memory_step):
        yield output  # 实时返回工具调用事件
        
        if isinstance(output, ToolOutput):
            if output.is_final_answer:
                # 处理最终答案
                final_answer = output.output
                got_final_answer = True
    """)
    
    print("\n💡 process_tool_calls 的作用：")
    print("-" * 30)
    print("• **并行执行**: 同时执行多个工具调用（如果 LLM 请求了多个）")
    print("• **错误处理**: 捕获和处理工具执行错误")
    print("• **结果收集**: 收集所有工具的执行结果")
    print("• **状态管理**: 更新 Agent 的内部状态")
    
    print("\n🎯 输出类型处理：")
    print("-" * 30)
    print("**ToolOutput 类型：**")
    print("• 普通工具输出：搜索结果、计算结果等")
    print("• 最终答案：is_final_answer=True 的特殊输出")
    
    print("\n**最终答案验证：**")
    print("• 确保只有一个最终答案")
    print("• 不允许最终答案与其他工具调用混合")
    print("• 支持状态变量引用")

def explain_step6_final_processing():
    """
    解释第6步：最终处理和返回
    """
    print("\n" + "=" * 80)
    print("🎯 第6步：最终处理和返回")
    print("=" * 80)
    
    print("\n🔍 最终答案处理：")
    print("-" * 30)
    print("""
    # 状态变量解析
    if isinstance(final_answer, str) and final_answer in self.state.keys():
        final_answer = self.state[final_answer]
    
    # 返回 ActionOutput
    yield ActionOutput(
        output=final_answer,
        is_final_answer=got_final_answer,
    )
    """)
    
    print("\n💡 状态变量机制：")
    print("-" * 30)
    print("• Agent 可以将复杂对象存储在 self.state 中")
    print("• 工具可以返回变量名而不是完整数据")
    print("• 最终输出时自动解析变量引用")
    
    print("\n🎯 实际场景：")
    print("-" * 30)
    print("""
    # 工具执行过程
    1. 数据分析工具执行，生成图表并存储
       self.state["chart_data"] = <matplotlib.figure.Figure>
       
    2. 工具返回变量名
       return "chart_data"
       
    3. 最终输出时解析
       final_answer = self.state["chart_data"]  # 获取实际图表对象
       
    4. 返回给用户
       ActionOutput(output=<图表对象>, is_final_answer=True)
    """)

def explain_error_scenarios():
    """
    解释各种错误场景
    """
    print("\n" + "=" * 80)
    print("🚨 错误场景和处理")
    print("=" * 80)
    
    print("\n❌ 常见错误类型：")
    print("-" * 30)
    print("1. **AgentGenerationError**: LLM 生成失败")
    print("   • 网络问题、API 限制、模型错误")
    
    print("\n2. **AgentParsingError**: 工具调用解析失败")
    print("   • 格式不正确、JSON 解析错误")
    
    print("\n3. **AgentExecutionError**: 执行逻辑错误")
    print("   • 最终答案与其他工具调用混合")
    
    print("\n4. **AgentToolExecutionError**: 工具执行错误")
    print("   • 多个最终答案、工具调用失败")
    
    print("\n🛡️ 错误处理策略：")
    print("-" * 30)
    print("• **早期检测**: 在问题扩大前及时发现")
    print("• **详细日志**: 记录完整的错误上下文")
    print("• **优雅降级**: 提供有意义的错误信息")
    print("• **状态保护**: 确保 Agent 状态不被破坏")

def demonstrate_complete_flow():
    """
    演示完整的执行流程
    """
    print("\n" + "=" * 80)
    print("🔄 完整执行流程演示")
    print("=" * 80)
    
    print("\n📝 场景：用户询问天气并要求温度转换")
    print("-" * 50)
    
    print("\n**步骤1：准备消息**")
    print("输入：'帮我查询北京天气，并将摄氏度转换为华氏度'")
    print("转换为：[{role: 'user', content: '帮我查询...'}]")
    
    print("\n**步骤2：LLM 生成响应**")
    print("LLM 思考：'我需要先搜索天气，然后进行温度转换'")
    print("生成工具调用：[{function: {name: 'web_search', arguments: {query: '北京天气'}}}]")
    
    print("\n**步骤3：执行工具调用**")
    print("调用 web_search('北京天气') → '今天北京气温25°C'")
    print("yield ToolOutput(output='今天北京气温25°C')")
    
    print("\n**步骤4：继续推理**")
    print("LLM 看到结果：'现在我需要将25°C转换为华氏度'")
    print("生成工具调用：[{function: {name: 'calculator', arguments: {expression: '25 * 9/5 + 32'}}}]")
    
    print("\n**步骤5：执行计算**")
    print("调用 calculator('25 * 9/5 + 32') → '77'")
    print("yield ToolOutput(output='77')")
    
    print("\n**步骤6：生成最终答案**")
    print("LLM 整合结果：'北京今天25°C，相当于77°F'")
    print("yield ActionOutput(output='北京今天25°C，相当于77°F', is_final_answer=True)")

def main():
    """
    主函数：运行所有解释和演示
    """
    explain_method_overview()
    explain_step1_message_preparation()
    explain_step2_generation_modes()
    explain_step3_error_handling()
    explain_step4_tool_call_parsing()
    explain_step5_tool_execution()
    explain_step6_final_processing()
    explain_error_scenarios()
    demonstrate_complete_flow()
    
    print("\n" + "=" * 80)
    print("📚 总结")
    print("=" * 80)
    print("_step_stream 方法实现了完整的 ReAct 循环：")
    print("1. **消息准备**：转换历史对话为 LLM 格式")
    print("2. **智能生成**：支持流式和非流式两种模式")
    print("3. **工具解析**：兼容原生和文本两种格式")
    print("4. **并行执行**：高效处理多个工具调用")
    print("5. **结果整合**：智能处理最终答案和状态变量")
    print("6. **错误处理**：全面的异常捕获和恢复机制")
    print("\n这是一个设计精良的执行引擎，")
    print("平衡了性能、可靠性和用户体验！")

if __name__ == "__main__":
    main()