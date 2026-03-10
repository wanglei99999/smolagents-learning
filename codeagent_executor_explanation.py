#!/usr/bin/env python3
"""
CodeAgent Python 执行器机制详解

这个文件详细解释了 CodeAgent 中 Python 执行器的工作原理，
包括为什么需要发送变量和工具，以及这种设计的优势。
"""

def explain_codeagent_concept():
    """
    解释 CodeAgent 的基本概念
    """
    print("=" * 80)
    print("🐍 CodeAgent Python 执行器机制详解")
    print("=" * 80)
    
    print("\n📋 什么是 CodeAgent？")
    print("-" * 50)
    print("• **CodeAgent** 是一种特殊的 Agent，能够生成和执行 Python 代码")
    print("• 与普通 Agent 不同，它不仅能调用预定义的工具，还能写代码解决问题")
    print("• 拥有独立的 Python 执行环境（沙箱），确保代码执行的安全性")
    print("• 可以进行数据分析、计算、可视化等复杂任务")
    
    print("\n🎯 为什么需要 Python 执行器？")
    print("-" * 50)
    print("1. **动态代码执行**：LLM 生成的 Python 代码需要在安全环境中执行")
    print("2. **变量共享**：代码执行时需要访问 Agent 的状态变量")
    print("3. **工具集成**：Python 代码中需要能够调用 Agent 的工具函数")
    print("4. **结果返回**：执行结果需要返回给 Agent 继续处理")

def explain_variable_sending():
    """
    解释发送变量的机制
    """
    print("\n" + "=" * 80)
    print("📦 发送变量机制：send_variables()")
    print("=" * 80)
    
    print("\n🔍 代码分析：")
    print("-" * 30)
    print("self.python_executor.send_variables(variables=self.state)")
    
    print("\n💡 作用解释：")
    print("-" * 30)
    print("• 将 Agent 的状态字典 (self.state) 发送到 Python 执行环境")
    print("• 这些变量在 Python 代码中可以直接使用，无需重新定义")
    print("• 包括用户通过 additional_args 传入的数据")
    
    print("\n🎯 实际场景示例：")
    print("-" * 30)
    print("""
    # 用户调用
    import pandas as pd
    df = pd.read_csv("sales_data.csv")
    
    agent.run(
        "分析销售数据并生成图表", 
        additional_args={"sales_data": df, "year": 2024}
    )
    
    # Agent 内部：self.state 现在包含
    # {
    #     "sales_data": <pandas.DataFrame>,
    #     "year": 2024
    # }
    
    # 发送到 Python 执行器后，LLM 生成的代码可以直接使用：
    # 
    # import matplotlib.pyplot as plt
    # 
    # # 直接使用变量，无需重新定义！
    # filtered_data = sales_data[sales_data['year'] == year]
    # plt.plot(filtered_data['month'], filtered_data['revenue'])
    # plt.title(f'{year}年销售趋势')
    # plt.show()
    """)

def explain_tool_sending():
    """
    解释发送工具的机制
    """
    print("\n" + "=" * 80)
    print("🛠️ 发送工具机制：send_tools()")
    print("=" * 80)
    
    print("\n🔍 代码分析：")
    print("-" * 30)
    print("self.python_executor.send_tools({**self.tools, **self.managed_agents})")
    
    print("\n💡 作用解释：")
    print("-" * 30)
    print("• 将 Agent 的所有工具和子 Agent 发送到 Python 执行环境")
    print("• 使用字典合并语法 {**dict1, **dict2} 将两个字典合并")
    print("• Python 代码中可以直接调用这些工具函数")
    print("• 实现了工具调用的无缝集成")
    
    print("\n🎯 实际场景示例：")
    print("-" * 30)
    print("""
    # Agent 配置了这些工具
    tools = [
        WebSearchTool(name="web_search"),
        CalculatorTool(name="calculator"),
        EmailTool(name="send_email")
    ]
    
    # 发送到 Python 执行器后，LLM 生成的代码可以直接调用：
    #
    # # 搜索最新信息
    # search_result = web_search("Python 3.12 新特性")
    # 
    # # 进行计算
    # result = calculator("25 * 4 + 10")
    # 
    # # 发送邮件
    # send_email(
    #     to="user@example.com",
    #     subject="分析报告",
    #     body=f"计算结果：{result}"
    # )
    """)

def demonstrate_workflow():
    """
    演示完整的工作流程
    """
    print("\n" + "=" * 80)
    print("🔄 完整工作流程演示")
    print("=" * 80)
    
    print("\n📝 步骤1：用户调用")
    print("-" * 30)
    print("""
    import pandas as pd
    from smolagents import CodeAgent
    
    # 准备数据
    df = pd.DataFrame({
        'product': ['A', 'B', 'C'],
        'sales': [100, 200, 150]
    })
    
    # 创建 CodeAgent
    agent = CodeAgent(
        model=model,
        tools=[WebSearchTool(), CalculatorTool()]
    )
    
    # 执行任务
    result = agent.run(
        "分析产品销售数据，计算总销售额，并搜索行业平均水平进行对比",
        additional_args={"product_data": df}
    )
    """)
    
    print("\n📝 步骤2：Agent 内部处理")
    print("-" * 30)
    print("""
    # 在 run() 方法中：
    
    # 1. 更新状态
    self.state.update({"product_data": df})
    
    # 2. 发送变量到 Python 执行器
    self.python_executor.send_variables(variables={
        "product_data": <DataFrame>
    })
    
    # 3. 发送工具到 Python 执行器
    self.python_executor.send_tools({
        "web_search": <WebSearchTool>,
        "calculator": <CalculatorTool>
    })
    """)
    
    print("\n📝 步骤3：LLM 生成代码")
    print("-" * 30)
    print("""
    # LLM 理解任务后生成 Python 代码：
    
    # 分析产品销售数据
    total_sales = product_data['sales'].sum()
    print(f"总销售额: {total_sales}")
    
    # 计算平均销售额
    avg_sales = calculator(f"{total_sales} / {len(product_data)}")
    print(f"平均销售额: {avg_sales}")
    
    # 搜索行业数据
    industry_data = web_search("产品销售行业平均水平 2024")
    print(f"行业信息: {industry_data}")
    
    # 生成对比分析
    analysis = f"我们的平均销售额是 {avg_sales}，行业情况：{industry_data}"
    print(analysis)
    """)
    
    print("\n📝 步骤4：代码执行")
    print("-" * 30)
    print("""
    # Python 执行器执行代码：
    # 1. 直接访问 product_data 变量（来自 send_variables）
    # 2. 调用 calculator 函数（来自 send_tools）
    # 3. 调用 web_search 函数（来自 send_tools）
    # 4. 返回执行结果给 Agent
    """)

def explain_security_and_isolation():
    """
    解释安全性和隔离机制
    """
    print("\n" + "=" * 80)
    print("🔒 安全性和隔离机制")
    print("=" * 80)
    
    print("\n🛡️ 安全特性：")
    print("-" * 30)
    print("1. **沙箱执行**：代码在隔离的环境中运行")
    print("2. **受限权限**：限制文件系统访问和网络操作")
    print("3. **资源限制**：限制内存和 CPU 使用")
    print("4. **超时控制**：防止无限循环或长时间运行")
    
    print("\n🔄 数据流向：")
    print("-" * 30)
    print("""
    Agent 主进程          Python 执行器（沙箱）
    ┌─────────────┐      ┌─────────────────────┐
    │ self.state  │ ──→  │ 全局变量空间        │
    │ self.tools  │ ──→  │ 可调用函数          │
    │ managed_agents ──→  │ 可调用函数          │
    └─────────────┘      └─────────────────────┘
                              │
                              ▼
                         ┌─────────────────────┐
                         │ 执行 LLM 生成的代码 │
                         └─────────────────────┘
                              │
                              ▼
                         ┌─────────────────────┐
                         │ 返回执行结果        │
                         └─────────────────────┘
    """)

def explain_advantages():
    """
    解释这种设计的优势
    """
    print("\n" + "=" * 80)
    print("🎯 设计优势分析")
    print("=" * 80)
    
    print("\n✅ 1. 无缝集成")
    print("-" * 30)
    print("• LLM 生成的代码可以直接使用变量和工具")
    print("• 无需复杂的序列化/反序列化过程")
    print("• 代码更简洁，执行更高效")
    
    print("\n✅ 2. 灵活性")
    print("-" * 30)
    print("• 支持任意复杂的数据结构（DataFrame、图片、模型等）")
    print("• 可以组合使用多个工具")
    print("• 支持复杂的计算和分析逻辑")
    
    print("\n✅ 3. 安全性")
    print("-" * 30)
    print("• 代码在隔离环境中执行")
    print("• 受控的资源访问")
    print("• 可以监控和限制执行过程")
    
    print("\n✅ 4. 可扩展性")
    print("-" * 30)
    print("• 新工具自动可用于代码执行")
    print("• 支持动态添加变量和工具")
    print("• 便于调试和监控")

def show_comparison():
    """
    对比不同的实现方式
    """
    print("\n" + "=" * 80)
    print("🔄 实现方式对比")
    print("=" * 80)
    
    print("\n❌ 方案1：每次传递参数")
    print("-" * 30)
    print("""
    # 每次执行代码时都要传递参数
    result = python_executor.execute(
        code="df.sum()",
        variables={"df": dataframe},
        tools={"calculator": calc_tool}
    )
    
    问题：
    • 每次调用都有序列化开销
    • 代码冗长，容易出错
    • 难以处理复杂的数据结构
    """)
    
    print("\n❌ 方案2：全局导入")
    print("-" * 30)
    print("""
    # 在执行环境中预先导入所有可能的工具
    exec("from agent_tools import *")
    exec(llm_generated_code)
    
    问题：
    • 命名空间污染
    • 安全风险
    • 无法动态调整工具集
    """)
    
    print("\n✅ 方案3：预发送机制（当前方案）")
    print("-" * 30)
    print("""
    # 在任务开始前一次性发送所有需要的变量和工具
    python_executor.send_variables(variables=self.state)
    python_executor.send_tools({**self.tools, **self.managed_agents})
    
    优势：
    • 一次设置，多次使用
    • 高效的执行性能
    • 清晰的变量和工具管理
    • 安全的隔离执行
    """)

def main():
    """
    主函数：运行所有解释和演示
    """
    explain_codeagent_concept()
    explain_variable_sending()
    explain_tool_sending()
    demonstrate_workflow()
    explain_security_and_isolation()
    explain_advantages()
    show_comparison()
    
    print("\n" + "=" * 80)
    print("📚 总结")
    print("=" * 80)
    print("CodeAgent 的 Python 执行器机制：")
    print("1. **变量发送**：让代码直接访问 Agent 状态")
    print("2. **工具发送**：让代码直接调用 Agent 工具")
    print("3. **安全执行**：在隔离环境中运行代码")
    print("4. **无缝集成**：LLM 生成的代码可以直接使用资源")
    print("\n这种设计实现了 Agent 与代码执行环境的完美融合，")
    print("让 AI 能够像人类程序员一样编写和执行代码！")

if __name__ == "__main__":
    main()