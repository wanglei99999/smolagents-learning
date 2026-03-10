#!/usr/bin/env python3
"""
完整的提示词模板校验机制详解

这个文件详细解释了 MultiStepAgent.__init__() 中的提示词模板校验逻辑，
包括为什么需要这种校验、如何工作、以及实际应用场景。
"""

# 首先，让我们看看 EMPTY_PROMPT_TEMPLATES 的结构
EMPTY_PROMPT_TEMPLATES = {
    "system_prompt": "",
    "planning": {
        "initial_plan": "",
        "update_plan_pre_messages": "",
        "update_plan_post_messages": ""
    },
    "managed_agent": {
        "pre_messages": "",
        "post_messages": ""
    },
    "final_answer": {
        "pre_messages": "",
        "post_messages": ""
    }
}

def explain_prompt_validation():
    """
    详细解释提示词模板校验机制
    """
    print("=" * 80)
    print("🔍 提示词模板校验机制完整解析")
    print("=" * 80)
    
    print("\n📋 1. 校验的目的和重要性")
    print("-" * 50)
    print("• 确保用户自定义的提示词模板包含所有必需的键")
    print("• 防止运行时因缺少模板而导致的错误")
    print("• 保证 Agent 的各个功能模块都有对应的提示词")
    print("• 提供清晰的错误信息，帮助用户快速定位问题")
    
    print("\n🏗️ 2. 校验机制的工作流程")
    print("-" * 50)
    print("步骤1: 设置默认模板")
    print("   self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES")
    print("   → 如果用户没有提供自定义模板，使用默认的空模板")
    
    print("\n步骤2: 检查是否需要校验")
    print("   if prompt_templates is not None:")
    print("   → 只有当用户提供了自定义模板时才进行校验")
    
    print("\n步骤3: 顶层键校验")
    print("   missing_keys = set(EMPTY_PROMPT_TEMPLATES.keys()) - set(prompt_templates.keys())")
    print("   → 检查 system_prompt, planning, managed_agent, final_answer 是否都存在")
    
    print("\n步骤4: 嵌套键校验")
    print("   → 检查每个字典类型的顶层键下的所有子键是否都存在")
    print("   → 例如 planning.initial_plan, planning.update_plan_pre_messages 等")

def demonstrate_validation_scenarios():
    """
    演示不同的校验场景
    """
    print("\n" + "=" * 80)
    print("🧪 校验场景演示")
    print("=" * 80)
    
    # 场景1: 完整的自定义模板（通过校验）
    print("\n✅ 场景1: 完整的自定义模板")
    print("-" * 30)
    complete_custom_templates = {
        "system_prompt": "你是一个智能助手",
        "planning": {
            "initial_plan": "制定初始计划",
            "update_plan_pre_messages": "更新计划前",
            "update_plan_post_messages": "更新计划后"
        },
        "managed_agent": {
            "pre_messages": "代理前置消息",
            "post_messages": "代理后置消息"
        },
        "final_answer": {
            "pre_messages": "最终答案前",
            "post_messages": "最终答案后"
        }
    }
    
    print("这个模板包含所有必需的键，校验会通过 ✓")
    
    # 场景2: 缺少顶层键（校验失败）
    print("\n❌ 场景2: 缺少顶层键")
    print("-" * 30)
    incomplete_top_level = {
        "system_prompt": "你是一个智能助手",
        "planning": {
            "initial_plan": "制定初始计划",
            "update_plan_pre_messages": "更新计划前",
            "update_plan_post_messages": "更新计划后"
        }
        # 缺少 managed_agent 和 final_answer
    }
    
    print("缺少的顶层键: {'managed_agent', 'final_answer'}")
    print("错误信息: Some prompt templates are missing from your custom `prompt_templates`: {'managed_agent', 'final_answer'}")
    
    # 场景3: 缺少嵌套键（校验失败）
    print("\n❌ 场景3: 缺少嵌套键")
    print("-" * 30)
    incomplete_nested = {
        "system_prompt": "你是一个智能助手",
        "planning": {
            "initial_plan": "制定初始计划"
            # 缺少 update_plan_pre_messages 和 update_plan_post_messages
        },
        "managed_agent": {
            "pre_messages": "代理前置消息",
            "post_messages": "代理后置消息"
        },
        "final_answer": {
            "pre_messages": "最终答案前",
            "post_messages": "最终答案后"
        }
    }
    
    print("缺少的嵌套键: planning 下的 update_plan_pre_messages 和 update_plan_post_messages")
    print("错误信息: Some prompt templates are missing from your custom `prompt_templates`: update_plan_pre_messages under planning")

def explain_validation_code_details():
    """
    详细解释校验代码的每一行
    """
    print("\n" + "=" * 80)
    print("🔬 校验代码逐行解析")
    print("=" * 80)
    
    print("\n📝 代码行1: 设置模板")
    print("self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES")
    print("解释: 使用 Python 的 or 操作符，如果 prompt_templates 为 None，则使用默认模板")
    
    print("\n📝 代码行2: 检查是否需要校验")
    print("if prompt_templates is not None:")
    print("解释: 只有用户提供了自定义模板时才需要校验，使用默认模板不需要校验")
    
    print("\n📝 代码行3-4: 顶层键校验")
    print("missing_keys = set(EMPTY_PROMPT_TEMPLATES.keys()) - set(prompt_templates.keys())")
    print("assert not missing_keys, (...)")
    print("解释:")
    print("  • set(EMPTY_PROMPT_TEMPLATES.keys()) = {'system_prompt', 'planning', 'managed_agent', 'final_answer'}")
    print("  • set(prompt_templates.keys()) = 用户提供的模板的顶层键集合")
    print("  • 两个集合相减得到缺少的键")
    print("  • assert not missing_keys 确保没有缺少的键")
    
    print("\n📝 代码行5-10: 嵌套键校验")
    print("for key, value in EMPTY_PROMPT_TEMPLATES.items():")
    print("    if isinstance(value, dict):")
    print("        for subkey in value.keys():")
    print("            assert key in prompt_templates.keys() and (subkey in prompt_templates[key].keys()), (...)")
    print("解释:")
    print("  • 遍历默认模板的每个顶层键")
    print("  • 如果值是字典类型（如 planning, managed_agent, final_answer）")
    print("  • 检查用户模板中是否包含所有子键")
    print("  • 双重检查：顶层键存在 AND 子键也存在")

def practical_usage_examples():
    """
    实际使用场景示例
    """
    print("\n" + "=" * 80)
    print("💡 实际使用场景")
    print("=" * 80)
    
    print("\n🎯 场景1: 自定义中文提示词")
    print("-" * 30)
    print("""
# 用户想要使用中文提示词
chinese_templates = {
    "system_prompt": "你是一个专业的AI助手，能够帮助用户解决各种问题。",
    "planning": {
        "initial_plan": "请制定一个详细的执行计划：",
        "update_plan_pre_messages": "根据当前情况，需要更新计划：",
        "update_plan_post_messages": "计划已更新，继续执行。"
    },
    "managed_agent": {
        "pre_messages": "现在调用专门的代理来处理：",
        "post_messages": "代理执行完成，结果如下："
    },
    "final_answer": {
        "pre_messages": "经过分析和处理，最终答案是：",
        "post_messages": "希望这个答案对您有帮助。"
    }
}

# 创建 Agent 时会自动校验这些模板
agent = MultiStepAgent(model=model, prompt_templates=chinese_templates)
    """)
    
    print("\n🎯 场景2: 专业领域定制")
    print("-" * 30)
    print("""
# 为医疗领域定制的提示词模板
medical_templates = {
    "system_prompt": "你是一个医疗AI助手，请基于循证医学提供建议。",
    "planning": {
        "initial_plan": "制定诊断和治疗计划：",
        "update_plan_pre_messages": "根据新的症状信息，调整诊疗方案：",
        "update_plan_post_messages": "方案已调整，继续评估。"
    },
    # ... 其他模板
}
    """)
    
    print("\n🎯 场景3: 调试和开发")
    print("-" * 30)
    print("""
# 开发者可能只想修改部分模板，但必须提供完整结构
debug_templates = {
    "system_prompt": "DEBUG MODE: 详细记录每个步骤",
    "planning": {
        "initial_plan": "[DEBUG] 初始计划：",
        "update_plan_pre_messages": "[DEBUG] 计划更新前：",
        "update_plan_post_messages": "[DEBUG] 计划更新后："
    },
    "managed_agent": {
        "pre_messages": "[DEBUG] 调用子代理：",
        "post_messages": "[DEBUG] 子代理完成："
    },
    "final_answer": {
        "pre_messages": "[DEBUG] 最终结果：",
        "post_messages": "[DEBUG] 执行完成"
    }
}
    """)

def main():
    """
    主函数：运行所有解释和演示
    """
    explain_prompt_validation()
    demonstrate_validation_scenarios()
    explain_validation_code_details()
    practical_usage_examples()
    
    print("\n" + "=" * 80)
    print("📚 总结")
    print("=" * 80)
    print("这个校验机制确保了：")
    print("1. 用户自定义模板的完整性")
    print("2. 运行时的稳定性")
    print("3. 清晰的错误提示")
    print("4. 灵活的定制能力")
    print("\n这是一个典型的防御性编程实践，通过在初始化时进行严格校验，")
    print("避免了运行时可能出现的各种模板缺失错误。")

if __name__ == "__main__":
    main()