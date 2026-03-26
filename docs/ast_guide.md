# Python AST（抽象语法树）学习指南

> 本文档是为理解 `local_python_executor.py` 而编写的 AST 基础教程。
> smolagents 的沙箱执行器不使用 `exec()`，而是用 `ast.parse()` 把代码解析成语法树，然后逐节点模拟执行。

## 1. AST 是什么

AST = Abstract Syntax Tree（抽象语法树）。

Python 执行代码前，会先把代码字符串解析成一棵树。每个语法结构（赋值、函数调用、if 判断等）都变成树上的一个节点。

```python
import ast

code = "x = 1 + 2"
tree = ast.parse(code)
print(ast.dump(tree, indent=2))
```

输出：

```
Module(
  body=[
    Assign(
      targets=[Name(id='x')],
      value=BinOp(
        left=Constant(value=1),
        op=Add(),
        right=Constant(value=2)
      )
    )
  ]
)
```

树形结构：

```
Module                        ← 根节点（整个代码文件）
└── body: [                   ← 顶层语句列表
      Assign                  ← 赋值节点
      ├── targets: [Name("x")]  ← 左边：变量名 x
      └── value: BinOp          ← 右边：二元运算
          ├── left: Constant(1)  ← 左操作数
          ├── op: Add()          ← 运算符
          └── right: Constant(2) ← 右操作数
    ]
```

**关键概念：`ast.parse()` 不执行任何代码，只分析语法结构。**

## 2. 核心节点类型速查表

按在 `local_python_executor.py` 中出现的频率排序：

| 节点类型 | Python 代码 | AST 表示 | 处理函数 |
|---------|------------|---------|---------|
| `ast.Constant` | `42`, `"hello"`, `True` | `Constant(value=42)` | 直接返回 `expression.value` |
| `ast.Name` | `x`, `result` | `Name(id='x')` | `evaluate_name` → 从 state 查值 |
| `ast.Assign` | `x = 10` | `Assign(targets=[Name('x')], value=Constant(10))` | `evaluate_assign` |
| `ast.Call` | `len(data)` | `Call(func=Name('len'), args=[Name('data')])` | `evaluate_call` |
| `ast.Attribute` | `obj.name` | `Attribute(value=Name('obj'), attr='name')` | `evaluate_attribute` |
| `ast.BinOp` | `x + y` | `BinOp(left=Name('x'), op=Add(), right=Name('y'))` | `evaluate_binop` |
| `ast.Compare` | `x > 5` | `Compare(left=Name('x'), ops=[Gt()], comparators=[Constant(5)])` | `evaluate_condition` |
| `ast.If` | `if x > 0: ...` | `If(test=Compare(...), body=[...], orelse=[...])` | `evaluate_if` |
| `ast.For` | `for i in range(10): ...` | `For(target=Name('i'), iter=Call(...), body=[...])` | `evaluate_for` |
| `ast.While` | `while x > 0: ...` | `While(test=Compare(...), body=[...])` | `evaluate_while` |
| `ast.FunctionDef` | `def add(a, b): ...` | `FunctionDef(name='add', args=..., body=[...])` | `evaluate_function_def` → `create_function` |
| `ast.Return` | `return x` | `Return(value=Name('x'))` | `raise ReturnException(value)` |
| `ast.Import` | `import json` | `Import(names=[alias(name='json')])` | `evaluate_import` |
| `ast.ImportFrom` | `from math import sqrt` | `ImportFrom(module='math', names=[alias(name='sqrt')])` | `evaluate_import` |
| `ast.Expr` | `print("hi")` (单独一行) | `Expr(value=Call(...))` | 剥壳 → `evaluate_ast(expression.value)` |
| `ast.AugAssign` | `x += 1` | `AugAssign(target=Name('x'), op=Add(), value=Constant(1))` | `evaluate_augassign` |
| `ast.Subscript` | `data[0]` | `Subscript(value=Name('data'), slice=Constant(0))` | `evaluate_subscript` |

## 3. 逐个详解

### 3.1 ast.Constant — 字面量

最简单的节点，没有子节点，直接包含值。

```python
42          → Constant(value=42)
"hello"     → Constant(value='hello')
True        → Constant(value=True)
None        → Constant(value=None)
3.14        → Constant(value=3.14)
```

沙箱处理方式：
```python
elif isinstance(expression, ast.Constant):
    return expression.value    # 直接返回，不需要任何计算
```

### 3.2 ast.Name — 变量名

只存了变量名字符串，不存值。值要去 `state` 字典里查。

```python
x           → Name(id='x')
result      → Name(id='result')
web_search  → Name(id='web_search')
```

沙箱处理方式（`evaluate_name`）：
```python
# 简化版逻辑
if name in state:
    return state[name]
elif name in static_tools:
    return static_tools[name]
elif name in custom_tools:
    return custom_tools[name]
else:
    raise InterpreterError(f"未定义的变量: {name}")
```

### 3.3 ast.Assign — 赋值

```python
x = 10
# Assign(
#   targets=[Name(id='x')],
#   value=Constant(value=10)
# )

a, b = 1, 2
# Assign(
#   targets=[Tuple(elts=[Name('a'), Name('b')])],
#   value=Tuple(elts=[Constant(1), Constant(2)])
# )

x = y = 10    # 多目标赋值
# Assign(
#   targets=[Name('x'), Name('y')],    ← targets 有两个元素
#   value=Constant(10)
# )
```

`targets` 是列表（因为可以 `x = y = 10`），`value` 是右边的表达式。

### 3.4 ast.Call — 函数调用（最重要）

```python
len(data)
# Call(
#   func=Name(id='len'),           ← 被调用的函数
#   args=[Name(id='data')],        ← 位置参数
#   keywords=[]                     ← 关键字参数
# )

print("hello", end="")
# Call(
#   func=Name(id='print'),
#   args=[Constant(value='hello')],
#   keywords=[keyword(arg='end', value=Constant(value=''))]
# )
```

方法调用是 Call + Attribute 的组合：
```python
data.items()
# Call(
#   func=Attribute(                 ← 函数是一个属性访问
#     value=Name(id='data'),        ← 对象
#     attr='items'                  ← 方法名
#   ),
#   args=[],
#   keywords=[]
# )
```

链式调用是 Call 嵌套 Call：
```python
get_func()("hello")
# Call(
#   func=Call(                      ← 函数本身是另一个调用的结果
#     func=Name(id='get_func'),
#     args=[]
#   ),
#   args=[Constant(value='hello')]
# )
```

### 3.5 ast.Attribute — 属性访问

```python
obj.name     → Attribute(value=Name('obj'), attr='name')
data.items   → Attribute(value=Name('data'), attr='items')
self.count   → Attribute(value=Name('self'), attr='count')
```

`value` 是点号左边的对象，`attr` 是点号右边的属性名（字符串）。

链式属性访问是嵌套的：
```python
a.b.c
# Attribute(
#   value=Attribute(              ← a.b
#     value=Name('a'),
#     attr='b'
#   ),
#   attr='c'                      ← .c
# )
```

### 3.6 ast.BinOp — 二元运算

```python
x + y    → BinOp(left=Name('x'), op=Add(), right=Name('y'))
a * 2    → BinOp(left=Name('a'), op=Mult(), right=Constant(2))
s // 2   → BinOp(left=Name('s'), op=FloorDiv(), right=Constant(2))
```

运算符类型：
| Python | AST op |
|--------|--------|
| `+` | `Add()` |
| `-` | `Sub()` |
| `*` | `Mult()` |
| `/` | `Div()` |
| `//` | `FloorDiv()` |
| `%` | `Mod()` |
| `**` | `Pow()` |
| `&` | `BitAnd()` |
| `\|` | `BitOr()` |
| `^` | `BitXor()` |

复合表达式是嵌套的：
```python
1 + 2 * 3
# BinOp(
#   left=Constant(1),
#   op=Add(),
#   right=BinOp(              ← 先算 2 * 3（优先级）
#     left=Constant(2),
#     op=Mult(),
#     right=Constant(3)
#   )
# )
```

### 3.7 ast.Compare — 比较运算

```python
x > 5
# Compare(
#   left=Name('x'),
#   ops=[Gt()],
#   comparators=[Constant(5)]
# )

1 < x < 10    # 链式比较是一个节点！
# Compare(
#   left=Constant(1),
#   ops=[Lt(), Lt()],                    ← 两个运算符
#   comparators=[Name('x'), Constant(10)] ← 两个比较对象
# )
```

比较运算符：`Eq`, `NotEq`, `Lt`, `LtE`, `Gt`, `GtE`, `Is`, `IsNot`, `In`, `NotIn`

### 3.8 ast.If — if 语句

```python
if x > 0:
    y = 1
elif x == 0:
    y = 0
else:
    y = -1
```

```
If(
  test=Compare(Name('x'), [Gt()], [Constant(0)]),   ← 条件
  body=[Assign(Name('y'), Constant(1))],             ← if 分支
  orelse=[                                            ← else 分支
    If(                                               ← elif 其实是嵌套的 If！
      test=Compare(Name('x'), [Eq()], [Constant(0)]),
      body=[Assign(Name('y'), Constant(0))],
      orelse=[Assign(Name('y'), Constant(-1))]        ← 最终的 else
    )
  ]
)
```

**重要：`elif` 在 AST 中不是独立节点，而是外层 If 的 `orelse` 里嵌套了另一个 If。**

### 3.9 ast.For — for 循环

```python
for i in range(10):
    print(i)
```

```
For(
  target=Name('i'),                              ← 循环变量
  iter=Call(Name('range'), [Constant(10)]),       ← 可迭代对象
  body=[Expr(Call(Name('print'), [Name('i')]))],  ← 循环体
  orelse=[]                                       ← for...else 的 else 分支
)
```

### 3.10 ast.FunctionDef — 函数定义

```python
def add(a, b=0):
    return a + b
```

```
FunctionDef(
  name='add',                                    ← 函数名
  args=arguments(
    args=[arg(arg='a'), arg(arg='b')],           ← 参数列表
    defaults=[Constant(0)]                        ← 默认值（从右往左对齐）
  ),
  body=[                                          ← 函数体
    Return(
      value=BinOp(Name('a'), Add(), Name('b'))
    )
  ],
  decorator_list=[]                               ← 装饰器列表
)
```

### 3.11 ast.Expr — 表达式语句（包装壳）

当一个表达式单独占一行时（不是赋值的一部分），外面会包一层 `Expr`：

```python
print("hello")     → Expr(value=Call(Name('print'), [Constant('hello')]))
42                  → Expr(value=Constant(42))
```

`Expr` 本身不做任何事，沙箱处理时直接剥壳：
```python
elif isinstance(expression, ast.Expr):
    return evaluate_ast(expression.value, ...)  # 处理里面的内容
```

### 3.12 ast.Import / ast.ImportFrom

```python
import json
# Import(names=[alias(name='json', asname=None)])

import json as j
# Import(names=[alias(name='json', asname='j')])

from math import sqrt, ceil
# ImportFrom(
#   module='math',
#   names=[alias(name='sqrt'), alias(name='ceil')]
# )

from json import *
# ImportFrom(module='json', names=[alias(name='*')])
```

### 3.13 ast.Return / ast.Break / ast.Continue

```python
return x + 1   → Return(value=BinOp(...))
return          → Return(value=None)
break           → Break()          ← 没有子节点
continue        → Continue()       ← 没有子节点
```

沙箱中这三个都用异常模拟：
```python
ast.Return   → raise ReturnException(value)
ast.Break    → raise BreakException()
ast.Continue → raise ContinueException()
```

### 3.14 ast.JoinedStr / ast.FormattedValue — f-string

```python
f"hello {name}, age={age:.1f}"
```

```
JoinedStr(values=[                          ← f-string 是多个部分拼接
  Constant(value='hello '),                 ← 纯文本部分
  FormattedValue(                           ← {name} 部分
    value=Name(id='name'),
    format_spec=None
  ),
  Constant(value=', age='),                 ← 纯文本部分
  FormattedValue(                           ← {age:.1f} 部分
    value=Name(id='age'),
    format_spec=JoinedStr(values=[Constant('.1f')])  ← 格式说明
  )
])
```

## 4. 递归执行的完整示例

```python
result = web_search("hello")
```

AST 结构：
```
Assign
├── targets: [Name(id='result')]
└── value: Call
    ├── func: Name(id='web_search')
    ├── args: [Constant(value='hello')]
    └── keywords: []
```

`evaluate_ast` 的递归执行过程：

```
evaluate_ast(Assign)                          ← 第1层：赋值
│
├── evaluate_assign()
│   ├── evaluate_ast(Call)                    ← 第2层：先算右边的函数调用
│   │   ├── evaluate_call()
│   │   │   ├── evaluate_ast(Name('web_search'))  ← 第3层：找函数
│   │   │   │   └── 在 static_tools 中找到 → 返回 <Tool 对象>
│   │   │   │
│   │   │   ├── evaluate_ast(Constant('hello'))   ← 第3层：算参数
│   │   │   │   └── 直接返回 "hello"
│   │   │   │
│   │   │   └── 调用 web_search("hello") → 返回 "搜索结果..."
│   │   │
│   │   └── 返回 "搜索结果..."
│   │
│   └── set_value(Name('result'), "搜索结果...", state)
│       └── state["result"] = "搜索结果..."
│
└── 返回 "搜索结果..."
```

## 5. 与 evaluate_ast 调度器的对应关系

`evaluate_ast` 就是一个大 `if/elif` 链，每个分支处理一种节点类型：

```python
def evaluate_ast(expression, state, static_tools, custom_tools, authorized_imports):
    # 第1组：赋值
    if isinstance(expression, ast.Assign):     → evaluate_assign()
    elif isinstance(expression, ast.AugAssign): → evaluate_augassign()

    # 第2组：函数调用
    elif isinstance(expression, ast.Call):      → evaluate_call()

    # 第3组：字面量和容器
    elif isinstance(expression, ast.Constant):  → return expression.value
    elif isinstance(expression, ast.List):      → 递归算每个元素
    elif isinstance(expression, ast.Dict):      → 递归算每个 key/value

    # 第4组：运算符
    elif isinstance(expression, ast.BinOp):     → evaluate_binop()
    elif isinstance(expression, ast.Compare):   → evaluate_condition()

    # 第5组：控制流
    elif isinstance(expression, ast.If):        → evaluate_if()
    elif isinstance(expression, ast.For):       → evaluate_for()
    elif isinstance(expression, ast.Return):    → raise ReturnException(value)
    elif isinstance(expression, ast.Break):     → raise BreakException()

    # 第6组：函数/类定义
    elif isinstance(expression, ast.FunctionDef): → evaluate_function_def()

    # 第7组：变量和属性
    elif isinstance(expression, ast.Name):      → evaluate_name() → 从 state 查值
    elif isinstance(expression, ast.Attribute): → evaluate_attribute()

    # 第8组：import
    elif isinstance(expression, (ast.Import, ast.ImportFrom)): → evaluate_import()

    # 兜底
    else: → raise InterpreterError("不支持的语法")
```

## 6. 动手实验

在 Python 中可以随时用 `ast.dump` 查看任何代码的 AST 结构：

```python
import ast

# 查看任意代码的 AST
code = """
for i in range(10):
    if i % 2 == 0:
        print(i)
"""
print(ast.dump(ast.parse(code), indent=2))
```

也可以用 `ast.unparse` 把 AST 还原成代码（Python 3.9+）：

```python
tree = ast.parse("x = 1 + 2")
print(ast.unparse(tree))  # 输出: x = 1 + 2
```

推荐在线工具：在浏览器中可视化 AST 树结构
- https://astexplorer.net（选择 Python 语言）
