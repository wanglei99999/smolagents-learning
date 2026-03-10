"""
理解提示词模板验证机制

这个示例展示：
1. 为什么需要验证提示词模板
2. 集合差集运算如何工作
3. 完整的验证流程
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


# ========== 空模板（定义所有必需的键） ==========

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

print("=" * 60)
print("空模板包含的所有必需键：")
print("=" * 60)
print(f"顶层键: {list(EMPTY_PROMPT_TEMPLATES.keys())}")
print(f"planning 子键: {list(EMPTY_PROMPT_TEMPLATES['planning'].keys())}")
print(f"managed_agent 子键: {list(EMPTY_PROMPT_TEMPLATES['managed_agent'].keys())}")
print(f"final_answer 子键: {list(EMPTY_PROMPT_TEMPLATES['final_answer'].keys())}")
print()


# ========== 示例1：完整的自定义模板（✅ 通过验证） ==========

print("=" * 60)
print("示例1：完整的自定义模板")
print("=" * 60)

custom_templates_complete = PromptTemplates(
    system_prompt="你是一个有用的助手",
    planning=PlanningPromptTemplate(
        initial_plan="制定初始计划...",
        update_plan_pre_messages="更新计划前...",
        update_plan_post_messages="更新计划后...",
    ),
    managed_agent=ManagedAgentPromptTemplate(
        task="执行任务...",
        report="报告结果..."
    ),
    final_answer=FinalAnswerPromptTemplate(
        pre_messages="最终答案前...",
        post_messages="最终答案后..."
    ),
)

# 验证顶层键
missing_keys = set(EMPTY_PROMPT_TEMPLATES.keys()) - set(custom_templates_complete.keys())
print(f"缺少的顶层键: {missing_keys if missing_keys else '无'}")
print(f"✅ 验证通过！所有必需的键都存在。")
print()


# ========== 示例2：不完整的模板（❌ 验证失败） ==========

print("=" * 60)
print("示例2：不完整的模板（缺少 final_answer）")
print("=" * 60)

custom_templates_incomplete = {
    "system_prompt": "你是一个有用的助手",
    "planning": {
        "initial_plan": "制定初始计划...",
        "update_plan_pre_messages": "更新计划前...",
        "update_plan_post_messages": "更新计划后...",
    },
    "managed_agent": {
        "task": "执行任务...",
        "report": "报告结果..."
    },
    # ❌ 缺少 "final_answer"
}

# 验证顶层键
missing_keys = set(EMPTY_PROMPT_TEMPLATES.keys()) - set(custom_templates_incomplete.keys())
print(f"缺少的顶层键: {missing_keys}")
print(f"❌ 验证失败！缺少必需的键。")
print()


# ========== 示例3：顶层键完整但嵌套键不完整（❌ 验证失败） ==========

print("=" * 60)
print("示例3：planning 缺少子键")
print("=" * 60)

custom_templates_missing_subkey = {
    "system_prompt": "你是一个有用的助手",
    "planning": {
        "initial_plan": "制定初始计划...",
        # ❌ 缺少 "update_plan_pre_messages" 和 "update_plan_post_messages"
    },
    "managed_agent": {
        "task": "执行任务...",
        "report": "报告结果..."
    },
    "final_answer": {
        "pre_messages": "最终答案前...",
        "post_messages": "最终答案后..."
    },
}

# 验证顶层键
missing_keys = set(EMPTY_PROMPT_TEMPLATES.keys()) - set(custom_templates_missing_subkey.keys())
print(f"缺少的顶层键: {missing_keys if missing_keys else '无'}")

# 验证嵌套键
print("\n检查嵌套键：")
for key, value in EMPTY_PROMPT_TEMPLATES.items():
    if isinstance(value, dict):
        required_subkeys = set(value.keys())
        actual_subkeys = set(custom_templates_missing_subkey.get(key, {}).keys())
        missing_subkeys = required_subkeys - actual_subkeys
        
        if missing_subkeys:
            print(f"  ❌ {key} 缺少子键: {missing_subkeys}")
        else:
            print(f"  ✅ {key} 的子键完整")
print()


# ========== 集合运算详解 ==========

print("=" * 60)
print("集合差集运算详解")
print("=" * 60)

set_a = {"system_prompt", "planning", "managed_agent", "final_answer"}
set_b = {"system_prompt", "planning"}

print(f"集合 A (必需的键): {set_a}")
print(f"集合 B (用户提供的键): {set_b}")
print(f"A - B (缺少的键): {set_a - set_b}")
print()

# 更多例子
print("更多例子：")
print(f"{{1, 2, 3, 4}} - {{1, 2}} = {{{1, 2, 3, 4} - {1, 2}}}")
print(f"{{1, 2, 3}} - {{1, 2, 3}} = {{{1, 2, 3} - {1, 2, 3}}}")
print(f"{{1, 2}} - {{1, 2, 3, 4}} = {{{1, 2} - {1, 2, 3, 4}}}")
print()


# ========== 为什么需要这个验证？ ==========

print("=" * 60)
print("为什么需要这个验证？")
print("=" * 60)
print("""
1. 防止运行时错误
   - 如果缺少某个模板，Agent 在运行时会崩溃
   - 提前验证可以在初始化时就发现问题

2. 提供清晰的错误信息
   - 告诉用户具体缺少哪些键
   - 避免模糊的 KeyError

3. 保证 Agent 的完整性
   - 每个 Agent 都需要完整的提示词模板
   - 缺少任何一个都会导致功能不完整

4. 支持自定义提示词
   - 用户可以自定义提示词内容
   - 但必须保持结构完整

示例场景：
- 用户想自定义系统提示词 ✅
- 但忘记提供 final_answer 模板 ❌
- 验证机制会立即提醒用户 ✅
""")


# ========== 实际使用示例 ==========

print("=" * 60)
print("实际使用示例")
print("=" * 60)

def validate_prompt_templates(prompt_templates):
    """验证提示词模板是否完整"""
    # 第1步：检查顶层键
    missing_keys = set(EMPTY_PROMPT_TEMPLATES.keys()) - set(prompt_templates.keys())
    if missing_keys:
        raise ValueError(
            f"Some prompt templates are missing from your custom `prompt_templates`: {missing_keys}"
        )
    
    # 第2步：检查嵌套键
    for key, value in EMPTY_PROMPT_TEMPLATES.items():
        if isinstance(value, dict):
            for subkey in value.keys():
                if key not in prompt_templates or subkey not in prompt_templates[key]:
                    raise ValueError(
                        f"Some prompt templates are missing from your custom `prompt_templates`: "
                        f"{subkey} under {key}"
                    )
    
    print("✅ 提示词模板验证通过！")
    return True

# 测试完整模板
try:
    validate_prompt_templates(custom_templates_complete)
except ValueError as e:
    print(f"❌ 验证失败: {e}")

# 测试不完整模板
try:
    validate_prompt_templates(custom_templates_incomplete)
except ValueError as e:
    print(f"❌ 验证失败: {e}")

print()


# ========== 总结 ==========

print("=" * 60)
print("总结")
print("=" * 60)
print("""
关键点：
1. set(A) - set(B) 返回在 A 中但不在 B 中的元素
2. 用于检查用户提供的模板是否包含所有必需的键
3. 分两步验证：顶层键 + 嵌套键
4. 提前发现问题，避免运行时错误

记忆技巧：
- 想象成"检查清单"
- EMPTY_PROMPT_TEMPLATES 是完整的清单
- 用户的 prompt_templates 是已完成的项
- missing_keys 是还没完成的项
""")
