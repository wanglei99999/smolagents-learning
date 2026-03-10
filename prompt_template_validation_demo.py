"""
提示词模板验证完整演示

这个示例展示：
1. 默认模板的使用
2. 完整自定义模板的验证
3. 不完整模板的错误处理
4. 实际的 Agent 创建过程
"""

from typing import TypedDict

# ========== 定义提示词模板结构 ==========

class PlanningPromptTemplate(TypedDict):
    initial_plan: str
    update_plan_pre_messages: str
    update_plan_post_messages: str

class ManagedAgentPromptTemplate(TypedDict):
    task: str
    report: str

class FinalAnswerPromptTemplate(TypedDict):
    pre_messages: str
    post_messages: str

class PromptTemplates(TypedDict):
    system_prompt: str
    planning: PlanningPromptTemplate
    managed_agent: ManagedAgentPromptTemplate
    final_answer: FinalAnswerPromptTemplate

# 空模板（定义结构）
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


# ========== 模拟 Agent 初始化中的验证逻辑 ==========

def validate_prompt_templates(prompt_templates):
    """模拟 MultiStepAgent.__init__ 中的验证逻辑"""
    
    print("🔍 开始验证提示词模板...")
    
    # 第1步：设置默认值
    final_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
    print(f"第1步：设置模板 - {'使用自定义模板' if prompt_templates else '使用默认空模板'}")
    
    # 第2步：检查是否需要验证
    if prompt_templates is not None:
        print("第2步：检测到自定义模板，开始验证...")
        
        # 第3步：验证顶层键
        print("\n第3步：验证顶层键")
        missing_keys = set(EMPTY_PROMPT_TEMPLATES.keys()) - set(prompt_templates.keys())
        print(f"  必需的键: {list(EMPTY_PROMPT_TEMPLATES.keys())}")
        print(f"  用户提供: {list(prompt_templates.keys())}")
        print(f"  缺少的键: {list(missing_keys) if missing_keys else '无'}")
        
        assert not missing_keys, (
            f"Some prompt templates are missing from your custom `prompt_templates`: {missing_keys}"
        )
        print("  ✅ 顶层键验证通过")
        
        # 第4步：验证嵌套键
        print("\n第4步：验证嵌套键")
        for key, value in EMPTY_PROMPT_TEMPLATES.items():
            if isinstance(value, dict):
                print(f"  检查 '{key}' 的子键:")
                for subkey in value.keys():
                    print(f"    检查 '{subkey}'...", end=" ")
                    
                    assert key in prompt_templates.keys() and (subkey in prompt_templates[key].keys()), (
                        f"Some prompt templates are missing from your custom `prompt_templates`: {subkey} under {key}"
                    )
                    print("✅")
            else:
                print(f"  跳过 '{key}' (不是字典)")
        
        print("\n✅ 所有验证通过！")
    else:
        print("第2步：未提供自定义模板，跳过验证")
    
    return final_templates


# ========== 场景1：不提供自定义模板（使用默认） ==========

print("=" * 70)
print("场景1：不提供自定义模板（使用默认）")
print("=" * 70)

try:
    result = validate_prompt_templates(None)
    print(f"\n✅ 成功！使用默认模板")
except AssertionError as e:
    print(f"\n❌ 失败：{e}")


# ========== 场景2：提供完整的自定义模板 ==========

print("\n" + "=" * 70)
print("场景2：提供完整的自定义模板")
print("=" * 70)

complete_custom_templates = {
    "system_prompt": "你是一个专业的AI助手，擅长解决复杂问题。",
    "planning": {
        "initial_plan": "让我制定一个详细的执行计划来解决这个任务...",
        "update_plan_pre_messages": "基于之前的执行结果，我需要更新计划...",
        "update_plan_post_messages": "更新后的计划如下..."
    },
    "managed_agent": {
        "task": "作为团队成员 {name}，请执行以下任务：{task}",
        "report": "团队成员 {name} 完成任务，结果：{final_answer}"
    },
    "final_answer": {
        "pre_messages": "基于以上分析和执行结果，我的最终答案是：",
        "post_messages": "以上就是我对这个问题的完整回答。"
    }
}

try:
    result = validate_prompt_templates(complete_custom_templates)
    print(f"\n✅ 成功！使用完整自定义模板")
    print(f"系统提示词预览: {result['system_prompt'][:50]}...")
except AssertionError as e:
    print(f"\n❌ 失败：{e}")


# ========== 场景3：缺少顶层键 ==========

print("\n" + "=" * 70)
print("场景3：缺少顶层键（final_answer）")
print("=" * 70)

incomplete_top_level = {
    "system_prompt": "你是一个AI助手。",
    "planning": {
        "initial_plan": "制定计划...",
        "update_plan_pre_messages": "更新前...",
        "update_plan_post_messages": "更新后..."
    },
    "managed_agent": {
        "task": "执行任务...",
        "report": "报告结果..."
    }
    # ❌ 缺少 "final_answer"
}

try:
    result = validate_prompt_templates(incomplete_top_level)
    print(f"\n✅ 成功！")
except AssertionError as e:
    print(f"\n❌ 失败：{e}")


# ========== 场景4：缺少嵌套键 ==========

print("\n" + "=" * 70)
print("场景4：缺少嵌套键（planning.update_plan_post_messages）")
print("=" * 70)

incomplete_nested = {
    "system_prompt": "你是一个AI助手。",
    "planning": {
        "initial_plan": "制定计划...",
        "update_plan_pre_messages": "更新前..."
        # ❌ 缺少 "update_plan_post_messages"
    },
    "managed_agent": {
        "task": "执行任务...",
        "report": "报告结果..."
    },
    "final_answer": {
        "pre_messages": "最终答案前...",
        "post_messages": "最终答案后..."
    }
}

try:
    result = validate_prompt_templates(incomplete_nested)
    print(f"\n✅ 成功！")
except AssertionError as e:
    print(f"\n❌ 失败：{e}")


# ========== 场景5：模拟真实的 Agent 创建 ==========

print("\n" + "=" * 70)
print("场景5：模拟真实的 Agent 创建")
print("=" * 70)

class MockAgent:
    """模拟 MultiStepAgent 的初始化过程"""
    
    def __init__(self, prompt_templates=None):
        print("🚀 开始创建 Agent...")
        
        # 这就是原代码中的验证逻辑
        self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
        
        if prompt_templates is not None:
            # 检查顶层键
            missing_keys = set(EMPTY_PROMPT_TEMPLATES.keys()) - set(prompt_templates.keys())
            assert not missing_keys, (
                f"Some prompt templates are missing from your custom `prompt_templates`: {missing_keys}"
            )
            
            # 检查嵌套键
            for key, value in EMPTY_PROMPT_TEMPLATES.items():
                if isinstance(value, dict):
                    for subkey in value.keys():
                        assert key in prompt_templates.keys() and (subkey in prompt_templates[key].keys()), (
                            f"Some prompt templates are missing from your custom `prompt_templates`: {subkey} under {key}"
                        )
        
        print("✅ Agent 创建成功！")
        print(f"使用的模板类型: {'自定义' if prompt_templates else '默认'}")

# 测试1：使用默认模板
print("\n测试1：使用默认模板")
try:
    agent1 = MockAgent()
except Exception as e:
    print(f"❌ 创建失败：{e}")

# 测试2：使用完整自定义模板
print("\n测试2：使用完整自定义模板")
try:
    agent2 = MockAgent(complete_custom_templates)
except Exception as e:
    print(f"❌ 创建失败：{e}")

# 测试3：使用不完整模板
print("\n测试3：使用不完整模板")
try:
    agent3 = MockAgent(incomplete_top_level)
except Exception as e:
    print(f"❌ 创建失败：{e}")


# ========== 总结 ==========

print("\n" + "=" * 70)
print("总结")
print("=" * 70)
print("""
这段验证代码的核心价值：

1. 🛡️ 防御性编程
   - 在初始化时就发现问题
   - 避免运行时的神秘崩溃
   - 提供清晰的错误信息

2. 🔧 用户友好
   - 支持完全自定义提示词
   - 也支持使用默认模板
   - 错误信息指出具体缺少什么

3. 📋 结构完整性
   - 确保所有必需的提示词都存在
   - 支持嵌套结构的验证
   - 保证 Agent 功能完整

4. 🎯 实际应用
   - 每个 Agent 都需要这些提示词
   - 缺少任何一个都会影响功能
   - 验证机制保证了系统的稳定性

记忆要点：
- prompt_templates or EMPTY_TEMPLATES → 设置默认值
- if prompt_templates is not None → 只验证自定义模板
- 集合差集 → 找出缺少的键
- isinstance(value, dict) → 只验证嵌套字典
- assert → 条件不满足就抛出错误
""")