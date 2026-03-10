"""
理解嵌套键验证和 assert 语句

这个示例展示：
1. 什么是嵌套字典
2. 如何检查嵌套键
3. assert 语句如何工作
4. 完整的验证流程
"""

# ========== 第一部分：理解 assert ==========

print("=" * 60)
print("第一部分：理解 assert 语句")
print("=" * 60)

# 例子1：assert 成功（条件为真）
print("\n例子1：assert 成功")
age = 20
try:
    assert age >= 18, "年龄必须大于等于18岁"
    print(f"✅ 断言通过：age={age} >= 18")
except AssertionError as e:
    print(f"❌ 断言失败：{e}")

# 例子2：assert 失败（条件为假）
print("\n例子2：assert 失败")
age = 15
try:
    assert age >= 18, "年龄必须大于等于18岁"
    print(f"✅ 断言通过：age={age} >= 18")
except AssertionError as e:
    print(f"❌ 断言失败：{e}")

# 例子3：assert 等价于 if + raise
print("\n例子3：assert 的等价写法")
age = 15

# 使用 assert
try:
    assert age >= 18, "年龄必须大于等于18岁"
except AssertionError as e:
    print(f"使用 assert: {e}")

# 等价的 if 写法
try:
    if not (age >= 18):
        raise AssertionError("年龄必须大于等于18岁")
except AssertionError as e:
    print(f"使用 if+raise: {e}")


# ========== 第二部分：理解嵌套字典 ==========

print("\n" + "=" * 60)
print("第二部分：理解嵌套字典")
print("=" * 60)

# 定义嵌套字典
EMPTY_PROMPT_TEMPLATES = {
    "system_prompt": "这是一个字符串",  # 不是字典
    "planning": {                      # 这是字典（嵌套）
        "initial_plan": "",
        "update_plan_pre_messages": "",
        "update_plan_post_messages": ""
    },
    "managed_agent": {                 # 这是字典（嵌套）
        "task": "",
        "report": ""
    },
    "final_answer": {                  # 这是字典（嵌套）
        "pre_messages": "",
        "post_messages": ""
    }
}

print("\n遍历 EMPTY_PROMPT_TEMPLATES：")
for key, value in EMPTY_PROMPT_TEMPLATES.items():
    if isinstance(value, dict):
        print(f"  {key}: 是字典，包含子键 {list(value.keys())}")
    else:
        print(f"  {key}: 是字符串，不需要检查子键")


# ========== 第三部分：检查嵌套键（完整模板） ==========

print("\n" + "=" * 60)
print("第三部分：检查嵌套键（完整模板）")
print("=" * 60)

# 用户提供的完整模板
prompt_templates_complete = {
    "system_prompt": "自定义系统提示词",
    "planning": {
        "initial_plan": "自定义初始计划",
        "update_plan_pre_messages": "自定义更新前",
        "update_plan_post_messages": "自定义更新后"
    },
    "managed_agent": {
        "task": "自定义任务",
        "report": "自定义报告"
    },
    "final_answer": {
        "pre_messages": "自定义前置消息",
        "post_messages": "自定义后置消息"
    }
}

print("\n开始验证...")
try:
    for key, value in EMPTY_PROMPT_TEMPLATES.items():
        print(f"\n检查 '{key}':")
        
        if isinstance(value, dict):
            print(f"  → '{key}' 是字典，需要检查子键")
            
            for subkey in value.keys():
                print(f"    检查子键 '{subkey}'...", end=" ")
                
                # 这就是原代码中的 assert 语句
                assert key in prompt_templates_complete.keys() and \
                       (subkey in prompt_templates_complete[key].keys()), \
                       f"缺少 {subkey} under {key}"
                
                print("✅")
        else:
            print(f"  → '{key}' 不是字典，跳过")
    
    print("\n✅ 所有嵌套键验证通过！")
    
except AssertionError as e:
    print(f"\n❌ 验证失败：{e}")


# ========== 第四部分：检查嵌套键（不完整模板） ==========

print("\n" + "=" * 60)
print("第四部分：检查嵌套键（不完整模板）")
print("=" * 60)

# 用户提供的不完整模板（planning 缺少子键）
prompt_templates_incomplete = {
    "system_prompt": "自定义系统提示词",
    "planning": {
        "initial_plan": "自定义初始计划",
        # ❌ 缺少 "update_plan_pre_messages" 和 "update_plan_post_messages"
    },
    "managed_agent": {
        "task": "自定义任务",
        "report": "自定义报告"
    },
    "final_answer": {
        "pre_messages": "自定义前置消息",
        "post_messages": "自定义后置消息"
    }
}

print("\n开始验证...")
try:
    for key, value in EMPTY_PROMPT_TEMPLATES.items():
        print(f"\n检查 '{key}':")
        
        if isinstance(value, dict):
            print(f"  → '{key}' 是字典，需要检查子键")
            
            for subkey in value.keys():
                print(f"    检查子键 '{subkey}'...", end=" ")
                
                # 这就是原代码中的 assert 语句
                assert key in prompt_templates_incomplete.keys() and \
                       (subkey in prompt_templates_incomplete[key].keys()), \
                       f"缺少 {subkey} under {key}"
                
                print("✅")
        else:
            print(f"  → '{key}' 不是字典，跳过")
    
    print("\n✅ 所有嵌套键验证通过！")
    
except AssertionError as e:
    print(f"\n❌ 验证失败：{e}")


# ========== 第五部分：逐步分解 assert 条件 ==========

print("\n" + "=" * 60)
print("第五部分：逐步分解 assert 条件")
print("=" * 60)

key = "planning"
subkey = "update_plan_pre_messages"

print(f"\n检查：{subkey} under {key}")
print(f"用户模板：{prompt_templates_incomplete}")

# 条件1：key 在用户模板中吗？
condition1 = key in prompt_templates_incomplete.keys()
print(f"\n条件1：'{key}' in prompt_templates.keys()")
print(f"  结果：{condition1}")
print(f"  解释：用户模板{'包含' if condition1 else '不包含'} '{key}' 键")

# 条件2：subkey 在用户模板的 key 字典中吗？
if condition1:
    condition2 = subkey in prompt_templates_incomplete[key].keys()
    print(f"\n条件2：'{subkey}' in prompt_templates['{key}'].keys()")
    print(f"  结果：{condition2}")
    print(f"  解释：用户模板的 '{key}' {'包含' if condition2 else '不包含'} '{subkey}' 子键")
    
    # 最终条件：两个条件都为真
    final_condition = condition1 and condition2
    print(f"\n最终条件：condition1 AND condition2")
    print(f"  结果：{final_condition}")
    print(f"  解释：{'✅ 验证通过' if final_condition else '❌ 验证失败'}")


# ========== 第六部分：为什么需要两个条件？ ==========

print("\n" + "=" * 60)
print("第六部分：为什么需要两个条件？")
print("=" * 60)

print("""
assert key in prompt_templates.keys() and (subkey in prompt_templates[key].keys())
       ↑ 条件1                              ↑ 条件2

条件1：key in prompt_templates.keys()
  - 检查顶层键是否存在
  - 例如：检查 'planning' 是否在用户模板中
  - 如果不存在，条件2 会报错（KeyError）

条件2：subkey in prompt_templates[key].keys()
  - 检查子键是否存在
  - 例如：检查 'initial_plan' 是否在 planning 字典中
  - 只有条件1为真时才能安全执行

为什么用 AND？
  - 两个条件都必须为真
  - 如果条件1为假，条件2不会执行（短路求值）
  - 避免 KeyError 异常
""")


# ========== 第七部分：实际场景模拟 ==========

print("=" * 60)
print("第七部分：实际场景模拟")
print("=" * 60)

def validate_nested_keys(prompt_templates):
    """完整的嵌套键验证函数"""
    print("\n开始验证嵌套键...")
    
    for key, value in EMPTY_PROMPT_TEMPLATES.items():
        if isinstance(value, dict):
            print(f"\n检查 '{key}' 的子键：")
            
            for subkey in value.keys():
                # 检查条件1
                if key not in prompt_templates.keys():
                    raise AssertionError(
                        f"缺少顶层键 '{key}'"
                    )
                
                # 检查条件2
                if subkey not in prompt_templates[key].keys():
                    raise AssertionError(
                        f"缺少子键 '{subkey}' under '{key}'"
                    )
                
                print(f"  ✅ {subkey}")
    
    print("\n✅ 所有嵌套键验证通过！")

# 测试1：完整模板
print("\n测试1：完整模板")
try:
    validate_nested_keys(prompt_templates_complete)
except AssertionError as e:
    print(f"❌ {e}")

# 测试2：不完整模板
print("\n测试2：不完整模板")
try:
    validate_nested_keys(prompt_templates_incomplete)
except AssertionError as e:
    print(f"❌ {e}")


# ========== 总结 ==========

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
1. assert 语句的作用：
   - 检查条件是否为真
   - 如果为假，抛出 AssertionError
   - 用于开发时的前置条件检查

2. 嵌套键验证的目的：
   - 确保用户提供的模板结构完整
   - 不仅检查顶层键，还检查子键
   - 防止运行时 KeyError

3. 为什么需要两个条件：
   - 条件1：确保顶层键存在
   - 条件2：确保子键存在
   - 使用 AND 避免 KeyError

4. 实际应用：
   - 在 Agent 初始化时验证
   - 提前发现配置错误
   - 提供清晰的错误信息

记忆技巧：
- assert = "断言" = "我确信这是真的"
- 嵌套验证 = "检查文件夹里的文件"
- 两个条件 = "先检查文件夹存在，再检查文件存在"
""")
