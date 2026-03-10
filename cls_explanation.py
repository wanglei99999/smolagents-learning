#!/usr/bin/env python3
"""
cls 命名约定详解

这个文件详细解释了 Python 中 cls 的含义和使用场景，
以及它与 class、self 等概念的区别。
"""

def explain_cls_naming():
    """
    解释 cls 命名约定
    """
    print("=" * 80)
    print("🔤 cls 命名约定详解")
    print("=" * 80)
    
    print("\n📋 cls 是什么？")
    print("-" * 50)
    print("• cls 是 'class' 的缩写")
    print("• 在 Python 中是一个命名约定（convention），不是关键字")
    print("• 通常用于表示'类'或'类型'")
    print("• 类似于 self 表示'实例'，cls 表示'类'")
    
    print("\n🎯 常见使用场景：")
    print("-" * 50)
    print("1. **类方法参数**：@classmethod 装饰的方法的第一个参数")
    print("2. **类型变量**：表示一个类或类型对象")
    print("3. **循环变量**：遍历类列表时的临时变量名")

def explain_cls_in_classmethod():
    """
    解释 cls 在类方法中的使用
    """
    print("\n" + "=" * 80)
    print("🏗️ cls 在类方法中的使用")
    print("=" * 80)
    
    print("\n📝 类方法示例：")
    print("-" * 30)
    print("""
    class MyClass:
        count = 0
        
        @classmethod
        def create_instance(cls):
            # cls 代表 MyClass 这个类本身
            cls.count += 1
            return cls()  # 等同于 MyClass()
    """)
    
    print("\n🔍 关键点：")
    print("• cls 是类方法的第一个参数")
    print("• cls 指向类本身，不是实例")
    print("• 可以通过 cls 访问类属性和创建实例")
    print("• 类似于其他语言中的 'static' 方法")

def explain_cls_as_type_variable():
    """
    解释 cls 作为类型变量的使用
    """
    print("\n" + "=" * 80)
    print("📦 cls 作为类型变量")
    print("=" * 80)
    
    print("\n📝 在回调注册中的使用：")
    print("-" * 30)
    print("""
    # 在 _setup_step_callbacks 中
    for step_cls, callbacks in step_callbacks.items():
        #   ^^^^^^^^
        #   这里的 step_cls 表示"步骤类型"
        
        for callback in callbacks:
            self.step_callbacks.register(step_cls, callback)
    """)
    
    print("\n🔍 含义解析：")
    print("• step_cls：步骤类（step class）的缩写")
    print("• 表示一个类对象，如 ActionStep、PlanningStep")
    print("• 不是类的实例，而是类本身")
    print("• 用于类型匹配和注册")

def demonstrate_cls_usage():
    """
    演示 cls 的实际使用
    """
    print("\n" + "=" * 80)
    print("💡 实际使用演示")
    print("=" * 80)
    
    print("\n🎯 示例1：类方法中的 cls")
    print("-" * 30)
    
    class Counter:
        count = 0
        
        @classmethod
        def increment(cls):
            """cls 代表 Counter 类本身"""
            cls.count += 1
            print(f"当前计数: {cls.count}")
        
        @classmethod
        def create(cls, initial_value=0):
            """使用 cls 创建实例"""
            instance = cls()  # 等同于 Counter()
            cls.count = initial_value
            return instance
    
    print("Counter.increment()  # cls 指向 Counter 类")
    Counter.increment()
    print("Counter.increment()")
    Counter.increment()
    
    print("\n🎯 示例2：类型变量中的 cls")
    print("-" * 30)
    
    # 模拟步骤类
    class MemoryStep:
        pass
    
    class ActionStep(MemoryStep):
        pass
    
    class PlanningStep(MemoryStep):
        pass
    
    # 模拟回调配置
    step_callbacks = {
        ActionStep: lambda step: print("Action callback"),
        PlanningStep: lambda step: print("Planning callback")
    }
    
    print("遍历步骤类型和回调：")
    for step_cls, callbacks in step_callbacks.items():
        #   ^^^^^^^^ 这里的 step_cls 是一个类对象
        print(f"  步骤类型: {step_cls.__name__}")
        print(f"  是否是类: {isinstance(step_cls, type)}")
        print(f"  回调函数: {callbacks}")

def explain_cls_vs_self():
    """
    对比 cls 和 self 的区别
    """
    print("\n" + "=" * 80)
    print("🔄 cls vs self 对比")
    print("=" * 80)
    
    print("\n📊 对比表：")
    print("-" * 50)
    print("| 特性       | self              | cls                |")
    print("|-----------|-------------------|--------------------|")
    print("| 含义      | 实例对象          | 类对象             |")
    print("| 使用场景  | 实例方法          | 类方法             |")
    print("| 装饰器    | 无需装饰器        | @classmethod       |")
    print("| 访问范围  | 实例属性+类属性   | 类属性             |")
    print("| 创建实例  | 不能              | 可以               |")
    
    print("\n📝 代码示例：")
    print("-" * 30)
    print("""
    class Example:
        class_var = "类变量"
        
        def __init__(self):
            self.instance_var = "实例变量"
        
        def instance_method(self):
            # self 指向实例对象
            print(self.instance_var)  # 访问实例变量
            print(self.class_var)     # 也能访问类变量
        
        @classmethod
        def class_method(cls):
            # cls 指向类本身
            print(cls.class_var)      # 访问类变量
            # print(cls.instance_var) # 错误！无法访问实例变量
            return cls()              # 可以创建实例
    """)

def explain_in_callback_context():
    """
    在回调上下文中解释 cls
    """
    print("\n" + "=" * 80)
    print("🎯 在回调系统中的 cls")
    print("=" * 80)
    
    print("\n📝 完整的上下文：")
    print("-" * 30)
    print("""
    # 用户配置回调
    step_callbacks = {
        ActionStep: [callback1, callback2],      # ActionStep 是一个类
        PlanningStep: callback3,                 # PlanningStep 是一个类
        FinalAnswerStep: [callback4]             # FinalAnswerStep 是一个类
    }
    
    # 在 _setup_step_callbacks 中处理
    for step_cls, callbacks in step_callbacks.items():
        # step_cls 依次是：ActionStep, PlanningStep, FinalAnswerStep
        # 这些都是类对象，不是实例
        
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        
        for callback in callbacks:
            # 注册：当遇到 step_cls 类型的步骤时，调用 callback
            self.step_callbacks.register(step_cls, callback)
    """)
    
    print("\n🔍 关键理解：")
    print("-" * 30)
    print("• step_cls 是一个类对象（如 ActionStep 类本身）")
    print("• 不是类的实例（不是 ActionStep() 创建的对象）")
    print("• 用于类型匹配：当执行步骤是这个类型时，触发回调")
    print("• 支持继承：子类的步骤也会触发父类注册的回调")

def show_type_checking():
    """
    展示类型检查
    """
    print("\n" + "=" * 80)
    print("🔬 类型检查演示")
    print("=" * 80)
    
    class ActionStep:
        pass
    
    # step_cls 是一个类
    step_cls = ActionStep
    
    print(f"step_cls 的值: {step_cls}")
    print(f"step_cls 的类型: {type(step_cls)}")
    print(f"step_cls 是否是 type: {isinstance(step_cls, type)}")
    print(f"step_cls 的名称: {step_cls.__name__}")
    
    # 创建实例
    step_instance = step_cls()  # 等同于 ActionStep()
    print(f"\nstep_instance 的类型: {type(step_instance)}")
    print(f"step_instance 是 ActionStep 的实例: {isinstance(step_instance, ActionStep)}")
    print(f"step_instance 是 ActionStep 类本身: {step_instance is ActionStep}")

def main():
    """
    主函数：运行所有解释和演示
    """
    explain_cls_naming()
    explain_cls_in_classmethod()
    explain_cls_as_type_variable()
    demonstrate_cls_usage()
    explain_cls_vs_self()
    explain_in_callback_context()
    show_type_checking()
    
    print("\n" + "=" * 80)
    print("📚 总结")
    print("=" * 80)
    print("cls 的含义：")
    print("1. **命名约定**：class 的缩写，表示'类'")
    print("2. **类方法参数**：@classmethod 的第一个参数")
    print("3. **类型变量**：表示一个类对象（不是实例）")
    print("4. **在回调中**：step_cls 表示步骤类型（如 ActionStep）")
    print("\n记住：cls 指向类本身，self 指向实例对象！")

if __name__ == "__main__":
    main()