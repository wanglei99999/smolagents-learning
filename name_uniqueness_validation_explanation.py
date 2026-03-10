#!/usr/bin/env python3
"""
名称唯一性校验详解：_validate_tools_and_managed_agents 方法

这个文件详细解释了为什么需要确保工具和子 Agent 名称的唯一性，
以及这种校验机制如何防止运行时的各种问题。
"""

def explain_name_uniqueness_importance():
    """
    解释名称唯一性的重要性
    """
    print("=" * 80)
    print("🔍 名称唯一性校验的重要性")
    print("=" * 80)
    
    print("\n📋 为什么需要名称唯一性？")
    print("-" * 50)
    print("1. **LLM 调用混淆**：相同名称会让 LLM 不知道调用哪个工具/Agent")
    print("2. **运行时冲突**：名称冲突可能导致调用错误的工具或 Agent")
    print("3. **调试困难**：重复名称让错误追踪变得复杂")
    print("4. **行为不确定**：可能出现随机选择工具的情况")
    print("5. **用户体验差**：不可预测的行为影响系统可靠性")

def analyze_validation_code():
    """
    逐行分析校验代码的实现
    """
    print("\n" + "=" * 80)
    print("🔬 校验代码逐行分析")
    print("=" * 80)
    
    print("\n📝 第1步：收集所有工具名称")
    print("-" * 40)
    print("tool_and_managed_agent_names = [tool.name for tool in tools]")
    print("• 提取所有工具的名称到列表中")
    print("• 例如：['calculator', 'web_search', 'file_reader']")
    
    print("\n📝 第2步：添加子 Agent 名称")
    print("-" * 40)
    print("if managed_agents is not None:")
    print("    tool_and_managed_agent_names += [agent.name for agent in managed_agents]")
    print("• 将子 Agent 的名称也加入到同一个列表中")
    print("• 例如：['calculator', 'web_search', 'data_analyst', 'code_reviewer']")
    
    print("\n📝 第3步：添加当前 Agent 自身名称")
    print("-" * 40)
    print("if self.name:")
    print("    tool_and_managed_agent_names.append(self.name)")
    print("• 如果当前 Agent 有名称（作为子 Agent 时），也加入列表")
    print("• 防止子 Agent 与其管理的工具/Agent 名称冲突")
    
    print("\n📝 第4步：巧妙的重复检测")
    print("-" * 40)
    print("if len(tool_and_managed_agent_names) != len(set(tool_and_managed_agent_names)):")
    print("• 原理：set() 会自动去重，如果长度不同说明有重复")
    print("• 例如：[1,2,2,3] 长度=4，set([1,2,2,3])={1,2,3} 长度=3")
    print("• 这是一个非常 Pythonic 的重复检测方法！")

def demonstrate_duplicate_detection():
    """
    演示重复检测的工作原理
    """
    print("\n" + "=" * 80)
    print("🧪 重复检测原理演示")
    print("=" * 80)
    
    print("\n✅ 无重复的情况：")
    print("-" * 30)
    names_no_dup = ['calculator', 'web_search', 'data_analyst', 'code_reviewer']
    print(f"原始列表: {names_no_dup}")
    print(f"列表长度: {len(names_no_dup)}")
    print(f"去重后集合: {set(names_no_dup)}")
    print(f"集合长度: {len(set(names_no_dup))}")
    print(f"长度相等: {len(names_no_dup) == len(set(names_no_dup))} → 无重复 ✓")
    
    print("\n❌ 有重复的情况：")
    print("-" * 30)
    names_with_dup = ['calculator', 'web_search', 'calculator', 'data_analyst']
    print(f"原始列表: {names_with_dup}")
    print(f"列表长度: {len(names_with_dup)}")
    print(f"去重后集合: {set(names_with_dup)}")
    print(f"集合长度: {len(set(names_with_dup))}")
    print(f"长度相等: {len(names_with_dup) == len(set(names_with_dup))} → 有重复 ✗")

def show_error_message_generation():
    """
    展示错误信息的生成逻辑
    """
    print("\n" + "=" * 80)
    print("📢 错误信息生成机制")
    print("=" * 80)
    
    print("\n🎯 错误信息构造：")
    print("-" * 30)
    print("""
    f"{[name for name in tool_and_managed_agent_names 
        if tool_and_managed_agent_names.count(name) > 1]}"
    """)
    
    print("这个列表推导式的作用：")
    print("• 遍历所有名称")
    print("• 使用 count() 方法统计每个名称出现的次数")
    print("• 只保留出现次数 > 1 的名称（即重复的名称）")
    print("• 生成重复名称的列表")
    
    print("\n💡 示例演示：")
    print("-" * 30)
    names = ['calculator', 'web_search', 'calculator', 'data_analyst', 'web_search']
    duplicates = [name for name in names if names.count(name) > 1]
    print(f"原始名称: {names}")
    print(f"重复名称: {list(set(duplicates))}")  # 去重显示
    print(f"错误信息: Each tool or managed_agent should have a unique name! You passed these duplicate names: {list(set(duplicates))}")

def explain_conflict_scenarios():
    """
    解释可能出现的冲突场景
    """
    print("\n" + "=" * 80)
    print("⚠️ 可能的冲突场景分析")
    print("=" * 80)
    
    print("\n🔥 场景1：工具之间的名称冲突")
    print("-" * 40)
    print("""
    tools = [
        Calculator(name="math_tool"),
        AdvancedCalculator(name="math_tool")  # 冲突！
    ]
    
    问题：LLM 不知道应该调用哪个 math_tool
    结果：可能随机选择，导致不可预测的行为
    """)
    
    print("\n🔥 场景2：工具与子 Agent 的名称冲突")
    print("-" * 40)
    print("""
    tools = [WebSearchTool(name="search")]
    managed_agents = [SearchAgent(name="search")]  # 冲突！
    
    问题：LLM 分不清是调用工具还是子 Agent
    结果：可能调用错误的组件
    """)
    
    print("\n🔥 场景3：子 Agent 与自身名称冲突")
    print("-" * 40)
    print("""
    # 子 Agent 的配置
    sub_agent = MultiStepAgent(
        name="analyzer",
        managed_agents=[
            DataAgent(name="analyzer")  # 与自身名称冲突！
        ]
    )
    
    问题：递归调用或调用混淆
    结果：可能导致无限循环或错误调用
    """)

def demonstrate_real_world_impact():
    """
    演示现实世界中的影响
    """
    print("\n" + "=" * 80)
    print("🌍 现实世界影响分析")
    print("=" * 80)
    
    print("\n💥 没有名称校验的后果：")
    print("-" * 40)
    print("1. **调试噩梦**：")
    print("   • 错误日志显示调用了 'search'，但不知道是哪个 search")
    print("   • 难以追踪问题的根源")
    
    print("\n2. **不一致的行为**：")
    print("   • 同样的输入可能产生不同的输出")
    print("   • 用户体验变得不可预测")
    
    print("\n3. **性能问题**：")
    print("   • 可能调用了性能较差的工具版本")
    print("   • 资源使用不优化")
    
    print("\n4. **安全风险**：")
    print("   • 可能调用了权限更高的工具")
    print("   • 意外的功能执行")
    
    print("\n✅ 有名称校验的好处：")
    print("-" * 40)
    print("1. **早期发现问题**：在初始化时就发现冲突")
    print("2. **清晰的错误信息**：明确指出哪些名称重复了")
    print("3. **强制最佳实践**：要求开发者使用唯一名称")
    print("4. **提高系统可靠性**：避免运行时的不确定行为")

def show_best_practices():
    """
    展示命名最佳实践
    """
    print("\n" + "=" * 80)
    print("💡 命名最佳实践")
    print("=" * 80)
    
    print("\n🎯 推荐的命名策略：")
    print("-" * 40)
    print("1. **描述性命名**：")
    print("   • 好：'web_search_tool', 'data_analysis_agent'")
    print("   • 差：'tool1', 'agent'")
    
    print("\n2. **功能前缀**：")
    print("   • 工具：'tool_calculator', 'tool_search'")
    print("   • Agent：'agent_analyst', 'agent_reviewer'")
    
    print("\n3. **领域后缀**：")
    print("   • 'calculator_math', 'calculator_finance'")
    print("   • 'search_web', 'search_database'")
    
    print("\n4. **版本标识**：")
    print("   • 'calculator_v1', 'calculator_v2'")
    print("   • 'search_basic', 'search_advanced'")
    
    print("\n⚠️ 避免的命名模式：")
    print("-" * 40)
    print("• 过于通用：'tool', 'agent', 'helper'")
    print("• 数字编号：'tool1', 'tool2'（除非有明确版本含义）")
    print("• 缩写过度：'calc', 'srch'（可读性差）")
    print("• 特殊字符：'tool@1', 'agent#2'（可能引起解析问题）")

def main():
    """
    主函数：运行所有解释和演示
    """
    explain_name_uniqueness_importance()
    analyze_validation_code()
    demonstrate_duplicate_detection()
    show_error_message_generation()
    explain_conflict_scenarios()
    demonstrate_real_world_impact()
    show_best_practices()
    
    print("\n" + "=" * 80)
    print("📚 总结")
    print("=" * 80)
    print("名称唯一性校验的价值：")
    print("1. **防止运行时冲突**：避免 LLM 调用混淆")
    print("2. **提高系统可靠性**：确保行为的可预测性")
    print("3. **改善开发体验**：早期发现问题，清晰的错误信息")
    print("4. **强制最佳实践**：推动良好的命名习惯")
    print("\n这是一个典型的'防御性编程'实践，")
    print("通过简单的校验避免了复杂的运行时问题。")

if __name__ == "__main__":
    main()