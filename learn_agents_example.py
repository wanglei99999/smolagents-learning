"""
smolagents 学习示例：从零开始理解 Agent 执行流程

这个示例会：
1. 创建一个简单的 CodeAgent
2. 逐步展示 ReAct 循环的执行过程
3. 解释每个步骤发生了什么
"""

from smolagents import CodeAgent, Tool, InferenceClientModel
from smolagents.monitoring import LogLevel


# ========== 第1步：创建自定义工具 ==========
class CalculatorTool(Tool):
    name = "calculator"
    description = "执行数学计算，输入一个数学表达式字符串"
    inputs = {
        "expression": {
            "type": "string",
            "description": "数学表达式，如 '2 + 3 * 4'"
        }
    }
    output_type = "number"
    
    def forward(self, expression: str) -> float:
        """安全地执行数学表达式"""
        try:
            # 只允许数学运算，不允许其他代码
            result = eval(expression, {"__builtins__": {}}, {})
            print(f"[Calculator] 计算 {expression} = {result}")
            return result
        except Exception as e:
            return f"计算错误: {e}"


# ========== 第2步：创建 Agent ==========
print("=" * 60)
print("创建 CodeAgent")
print("=" * 60)

model = InferenceClientModel()  # 使用 HuggingFace 推理 API
agent = CodeAgent(
    tools=[CalculatorTool()],
    model=model,
    max_steps=5,  # 最多5步
    verbosity_level=LogLevel.INFO,  # 显示详细日志
    stream_outputs=False  # 非流式模式（便于学习）
)

print(f"✓ Agent 创建成功")
print(f"  - 可用工具: {list(agent.tools.keys())}")
print(f"  - 最大步数: {agent.max_steps}")
print(f"  - 模型: {type(agent.model).__name__}")
print()


# ========== 第3步：运行任务（非流式） ==========
print("=" * 60)
print("运行任务（非流式模式）")
print("=" * 60)

task = "计算 (15 + 27) * 3 的结果，然后除以 2"
print(f"任务: {task}\n")

# 运行并获取完整结果
result = agent.run(
    task=task,
    return_full_result=True  # 返回完整的 RunResult 对象
)

print("\n" + "=" * 60)
print("执行结果")
print("=" * 60)
print(f"最终答案: {result.output}")
print(f"执行状态: {result.state}")
print(f"总步数: {len([s for s in result.steps if s.get('type') == 'ActionStep'])}")
print(f"Token 使用: {result.token_usage}")
print(f"耗时: {result.timing.duration:.2f}秒")
print()


# ========== 第4步：查看执行步骤 ==========
print("=" * 60)
print("详细执行步骤")
print("=" * 60)

for i, step in enumerate(result.steps, 1):
    step_type = step.get('type', 'Unknown')
    print(f"\n步骤 {i}: {step_type}")
    
    if step_type == 'TaskStep':
        print(f"  任务: {step.get('task', '')[:50]}...")
    
    elif step_type == 'ActionStep':
        print(f"  步骤编号: {step.get('step_number')}")
        
        # 显示 LLM 生成的代码
        if 'code_action' in step:
            print(f"  生成的代码:")
            for line in step['code_action'].split('\n'):
                print(f"    {line}")
        
        # 显示执行结果
        if 'observations' in step:
            print(f"  观察结果:")
            print(f"    {step['observations'][:100]}...")
        
        # 显示是否是最终答案
        if step.get('is_final_answer'):
            print(f"  ✓ 这是最终答案！")
    
    elif step_type == 'FinalAnswerStep':
        print(f"  最终输出: {step.get('output')}")

print()


# ========== 第5步：运行任务（流式） ==========
print("=" * 60)
print("运行任务（流式模式）")
print("=" * 60)

task2 = "计算 100 除以 4 的结果"
print(f"任务: {task2}\n")

# 流式运行：实时看到每个事件
for event in agent.run(task=task2, stream=True, reset=True):
    event_type = type(event).__name__
    print(f"[事件] {event_type}")
    
    # 根据事件类型显示不同信息
    if event_type == 'ActionStep':
        print(f"  → 步骤 {event.step_number} 完成")
        if hasattr(event, 'code_action') and event.code_action:
            print(f"  → 执行的代码: {event.code_action[:50]}...")
    
    elif event_type == 'FinalAnswerStep':
        print(f"  → 最终答案: {event.output}")
        print(f"  ✓ 任务完成！")

print()


# ========== 第6步：可视化 Agent 结构 ==========
print("=" * 60)
print("Agent 结构可视化")
print("=" * 60)
agent.visualize()
print()


# ========== 第7步：回放执行过程 ==========
print("=" * 60)
print("回放执行过程")
print("=" * 60)
agent.replay(detailed=False)  # detailed=True 会显示每步的完整记忆


# ========== 学习总结 ==========
print("\n" + "=" * 60)
print("学习总结")
print("=" * 60)
print("""
你已经学会了：

1. ✓ 如何创建自定义工具（Tool）
2. ✓ 如何初始化 CodeAgent
3. ✓ 理解 ReAct 循环的执行流程
4. ✓ 区分流式 vs 非流式模式
5. ✓ 查看执行步骤和调试信息
6. ✓ 使用 visualize() 和 replay() 工具

下一步建议：
- 阅读 src/smolagents/memory.py 了解记忆系统
- 阅读 src/smolagents/tools.py 了解工具系统
- 尝试创建更复杂的工具和多 Agent 系统
""")
