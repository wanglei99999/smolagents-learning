#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# local_python_executor.py —— CodeAgent 的本地 Python 沙箱执行器
#
# 这是 smolagents 安全执行 LLM 生成代码的核心模块。
# 不同于直接 exec()，本文件实现了一个基于 AST 的受限 Python 解释器，
# 能够在不启动任何子进程的情况下，在当前进程内安全执行受限代码。
#
# 安全机制（多层防护）：
#
#   1. 导入白名单（authorized_imports）
#      - 只有在白名单中的模块才能被 import
#      - 默认白名单：BASE_BUILTIN_MODULES（collections, json, math, datetime 等）
#      - 用户通过 additional_authorized_imports 扩展
#      - additional_authorized_imports=["*"] 则允许所有导入（危险！生产环境禁用）
#
#   2. 危险模块黑名单（DANGEROUS_MODULES）
#      - os, sys, subprocess, socket, pathlib 等文件系统/网络/进程相关模块被禁止
#
#   3. Dunder 方法限制（ALLOWED_DUNDER_METHODS）
#      - 只允许 __init__, __str__, __repr__
#      - 防止通过 __class__.__bases__ 等方式逃逸沙箱
#
#   4. 操作计数器（MAX_OPERATIONS = 10,000,000）
#      - 限制总操作数，防止无限循环消耗资源
#
#   5. 执行超时（MAX_EXECUTION_TIME_SECONDS = 30）
#      - 使用线程池实现超时（兼容 Windows 和非主线程调用，不依赖 signal）
#      - 超时后抛出 ExecutionTimeoutError
#
#   6. print 重定向
#      - 代码中的 print() 输出被重定向到 PrintContainer
#      - 不会真正打印到终端，而是收集到 logs 中返回给 Agent
#
# 核心执行流程（evaluate_python_code → LocalPythonExecutor.__call__）：
#   代码字符串 → ast.parse() → 逐节点 evaluate_ast() → 返回 CodeOutput
#
# final_answer 的特殊处理：
#   - 当代码调用 final_answer(xxx) 时，抛出 FinalAnswerException
#   - 这个异常被捕获后，设置 is_final_answer=True，终止 ReAct 循环
#   - 使用 BaseException 而非 Exception，防止被代码中的 except Exception 误捕获
#
# 状态持久化：
#   - LocalPythonExecutor 在多步执行之间保持 self.state（变量存储）
#   - 每步新产生的变量会留在 state 中，供后续步骤使用
#   - Agent 的 additional_args 也通过 send_variables() 注入 state
# =============================================================================

import ast
import builtins
import difflib
import inspect
import logging
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from functools import wraps
from importlib import import_module
from importlib.util import find_spec
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Any

from .tools import Tool
from .utils import BASE_BUILTIN_MODULES, truncate_content


logger = logging.getLogger(__name__)


class InterpreterError(ValueError):
    """
    An error raised when the interpreter cannot evaluate a Python expression, due to syntax error or unsupported
    operations.
    """

    pass


ERRORS = {
    name: getattr(builtins, name)
    for name in dir(builtins)
    if isinstance(getattr(builtins, name), type) and issubclass(getattr(builtins, name), BaseException)
}

DEFAULT_MAX_LEN_OUTPUT = 50000   # print 输出的最大长度（字符数），超出则截断
MAX_OPERATIONS = 10000000        # 最大操作计数（防止无限循环）
MAX_WHILE_ITERATIONS = 1000000   # while 循环的最大迭代次数
MAX_EXECUTION_TIME_SECONDS = 30  # 单次代码执行的最大时间（秒）
ALLOWED_DUNDER_METHODS = ["__init__", "__str__", "__repr__"]  # 允许调用的魔法方法白名单


def custom_print(*args):
    """替换 print()：LLM 生成的代码中的 print 不打印到终端，
    实际输出被重定向到 state["_print_outputs"]（见 evaluate_python_code 中的实现）。
    """
    return None


def nodunder_getattr(obj, name, default=None):
    """安全版 getattr：禁止访问双下划线（dunder）属性，防止沙箱逃逸。
    例如：obj.__class__.__bases__ 这类访问会被拦截。
    """
    if name.startswith("__") and name.endswith("__"):
        raise InterpreterError(f"Forbidden access to dunder attribute: {name}")
    return getattr(obj, name, default)


# 沙箱中默认可用的 Python 内置函数集合
# LLM 生成的代码只能调用这些函数（以及白名单模块中的函数）
# 注意：print 被替换为 custom_print（输出重定向到日志）
# 注意：getattr 被替换为 nodunder_getattr（防止访问危险属性）
BASE_PYTHON_TOOLS = {
    "print": custom_print,
    "isinstance": isinstance,
    "range": range,
    "float": float,
    "int": int,
    "bool": bool,
    "str": str,
    "set": set,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "degrees": math.degrees,
    "radians": math.radians,
    "pow": pow,
    "sqrt": math.sqrt,
    "len": len,
    "sum": sum,
    "max": max,
    "min": min,
    "abs": abs,
    "enumerate": enumerate,
    "zip": zip,
    "reversed": reversed,
    "sorted": sorted,
    "all": all,
    "any": any,
    "map": map,
    "filter": filter,
    "ord": ord,
    "chr": chr,
    "next": next,
    "iter": iter,
    "divmod": divmod,
    "callable": callable,
    "getattr": nodunder_getattr,
    "hasattr": hasattr,
    "setattr": setattr,
    "issubclass": issubclass,
    "type": type,
    "complex": complex,
}

# 危险模块黑名单（不完整，仅列举最典型的危险模块）
# 即使这些模块在 authorized_imports 中，也不应允许导入
# 注意：这个黑名单不是主要防护手段，主要防护靠 authorized_imports 白名单
DANGEROUS_MODULES = [
    "builtins",
    "io",
    "multiprocessing",
    "os",
    "pathlib",
    "pty",
    "shutil",
    "socket",
    "subprocess",
    "sys",
]

DANGEROUS_FUNCTIONS = [
    "builtins.compile",
    "builtins.eval",
    "builtins.exec",
    "builtins.globals",
    "builtins.locals",
    "builtins.__import__",
    "os.popen",
    "os.system",
    "posix.system",
]


def check_safer_result(result: Any, static_tools: dict[str, Callable] = None, authorized_imports: list[str] = None):
    """
    Checks if a result is safer according to authorized imports and static tools.

    Args:
        result (Any): The result to check.
        static_tools (dict[str, Callable]): Dictionary of static tools.
        authorized_imports (list[str]): List of authorized imports.

    Raises:
        InterpreterError: If the result is not safe
    """
    if isinstance(result, ModuleType):
        if not check_import_authorized(result.__name__, authorized_imports):
            raise InterpreterError(f"Forbidden access to module: {result.__name__}")
    elif isinstance(result, dict) and result.get("__spec__"):
        if not check_import_authorized(result["__name__"], authorized_imports):
            raise InterpreterError(f"Forbidden access to module: {result['__name__']}")
    elif isinstance(result, (FunctionType, BuiltinFunctionType)):
        for qualified_function_name in DANGEROUS_FUNCTIONS:
            module_name, function_name = qualified_function_name.rsplit(".", 1)
            if (
                (static_tools is None or function_name not in static_tools)
                and result.__name__ == function_name
                and result.__module__ == module_name
            ):
                raise InterpreterError(f"Forbidden access to function: {function_name}")


def safer_eval(func: Callable):
    """
    Decorator to enhance the security of an evaluation function by checking its return value.

    Args:
        func (Callable): Evaluation function to be made safer.

    Returns:
        Callable: Safer evaluation function with return value check.
    """

    @wraps(func)
    def _check_return(
        expression,
        state,
        static_tools,
        custom_tools,
        authorized_imports=BASE_BUILTIN_MODULES,
    ):
        result = func(expression, state, static_tools, custom_tools, authorized_imports=authorized_imports)
        check_safer_result(result, static_tools, authorized_imports)
        return result

    return _check_return


def safer_func(
    func: Callable,
    static_tools: dict[str, Callable] = BASE_PYTHON_TOOLS,
    authorized_imports: list[str] = BASE_BUILTIN_MODULES,
):
    """
    Decorator to enhance the security of a function call by checking its return value.

    Args:
        func (Callable): Function to be made safer.
        static_tools (dict[str, Callable]): Dictionary of static tools.
        authorized_imports (list[str]): List of authorized imports.

    Returns:
        Callable: Safer function with return value check.
    """
    # If the function is a type, return it directly without wrapping
    if isinstance(func, type):
        return func

    @wraps(func)
    def _check_return(*args, **kwargs):
        result = func(*args, **kwargs)
        check_safer_result(result, static_tools, authorized_imports)
        return result

    return _check_return


class PrintContainer:
    """print 输出的收集容器。

    沙箱中的 print() 不会打印到终端，而是把内容追加到这个容器中。
    执行结束后，容器中的内容作为 CodeOutput.logs 返回给 Agent。

    使用方式（在 evaluate_python_code 中）：
        state["_print_outputs"] = PrintContainer()
        # LLM 代码中的 print("hello") 会触发：
        #   state["_print_outputs"] += "hello"
    """

    def __init__(self):
        self.value = ""

    def append(self, text):
        """追加文本内容"""
        self.value += text
        return self

    def __iadd__(self, other):
        """实现 += 运算符，支持 state["_print_outputs"] += "text" """
        self.value += str(other)
        return self

    def __str__(self):
        """返回收集到的全部 print 输出"""
        return self.value

    def __repr__(self):
        return f"PrintContainer({self.value})"

    def __len__(self):
        """支持 len()，用于检查输出是否为空"""
        return len(self.value)


class BreakException(Exception):
    """模拟 break 语句：在 evaluate_for / evaluate_while 中，
    遇到 break 时抛出此异常，由外层循环的 except 捕获来跳出循环。"""
    pass


class ContinueException(Exception):
    """模拟 continue 语句：抛出后被循环体的 except 捕获，跳到下一次迭代。"""
    pass


class ReturnException(Exception):
    """模拟 return 语句：在 create_function 创建的函数体中，
    遇到 return 时抛出此异常，携带返回值，由函数调用处捕获。"""
    def __init__(self, value):
        self.value = value


class ExecutionTimeoutError(Exception):
    """代码执行超时异常。当执行时间超过 MAX_EXECUTION_TIME_SECONDS（默认30秒）时抛出。
    由 timeout() 装饰器中的 ThreadPoolExecutor 触发。"""
    pass


def timeout(timeout_seconds: int):
    """执行超时装饰器：限制函数的最大执行时间。

    实现方式：用 ThreadPoolExecutor 在子线程中执行函数，主线程等待指定秒数。
    - 跨平台：Windows 也能用（不依赖 signal）
    - 线程安全：可以在非主线程中调用（signal 只能在主线程用）

    注意：超时后子线程不会被强制杀死（Python 无法安全杀线程），
    它会继续在后台运行直到完成，但调用方已经收到 TimeoutError 继续往下走了。

    Args:
        timeout_seconds: 最大允许执行时间（秒）
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a new ThreadPoolExecutor for each call to avoid threading issues
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    result = future.result(timeout=timeout_seconds)
                    return result
                except FuturesTimeoutError:
                    raise ExecutionTimeoutError(
                        f"Code execution exceeded the maximum execution time of {timeout_seconds} seconds"
                    )

        return wrapper

    return decorator


def get_iterable(obj):
    if isinstance(obj, list):
        return obj
    elif hasattr(obj, "__iter__"):
        return list(obj)
    else:
        raise InterpreterError("Object is not iterable")


def fix_final_answer_code(code: str) -> str:
    """
    Sometimes an LLM can try to assign a variable to final_answer, which would break the final_answer() tool.
    This function fixes this behaviour by replacing variable assignments to final_answer with final_answer_variable,
    while preserving function calls to final_answer().
    """
    # First, find if there's a direct assignment to final_answer
    # Use word boundary and negative lookbehind to ensure it's not an object attribute
    assignment_pattern = r"(?<!\.)(?<!\w)\bfinal_answer\s*="
    if "final_answer(" not in code or not re.search(assignment_pattern, code):
        # If final_answer tool is not called in this blob, then doing the replacement is hazardous because it could false the model's memory for next steps.
        # Let's not modify the code and leave the subsequent assignment error happen.
        return code

    # Pattern for replacing variable assignments
    # Looks for 'final_answer' followed by '=' with optional whitespace
    # Negative lookbehind ensures we don't match object attributes
    assignment_regex = r"(?<!\.)(?<!\w)(\bfinal_answer)(\s*=)"
    code = re.sub(assignment_regex, r"final_answer_variable\2", code)

    # Pattern for replacing variable usage but not function calls
    # Negative lookahead (?!\s*\() ensures we don't match function calls
    # Negative lookbehind (?<!\.|\w) ensures we don't match object methods or other variables
    variable_regex = r"(?<!\.)(?<!\w)(\bfinal_answer\b)(?!\s*\()"
    code = re.sub(variable_regex, "final_answer_variable", code)
    return code


def build_import_tree(authorized_imports: list[str]) -> dict[str, Any]:
    tree = {}
    for import_path in authorized_imports:
        parts = import_path.split(".")
        current = tree
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]
    return tree


def check_import_authorized(import_to_check: str, authorized_imports: list[str]) -> bool:
    current_node = build_import_tree(authorized_imports)
    for part in import_to_check.split("."):
        if "*" in current_node:
            return True
        if part not in current_node:
            return False
        current_node = current_node[part]
    return True

#属性访问解释器
def evaluate_attribute(
    expression: ast.Attribute,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    #这个叫拦截dunder属性，dunder属性是指双下划线开头的属性。用来防止危险方法访问
    if expression.attr.startswith("__") and expression.attr.endswith("__"):
        raise InterpreterError(f"Forbidden access to dunder attribute: {expression.attr}")
    value = evaluate_ast(expression.value, state, static_tools, custom_tools, authorized_imports)
    return getattr(value, expression.attr)

#处理一元运算符的
def evaluate_unaryop(
    expression: ast.UnaryOp,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    operand = evaluate_ast(expression.operand, state, static_tools, custom_tools, authorized_imports)
    if isinstance(expression.op, ast.USub):
        return -operand
    elif isinstance(expression.op, ast.UAdd):
        return operand
    elif isinstance(expression.op, ast.Not):
        return not operand
    elif isinstance(expression.op, ast.Invert):
        return ~operand
    else:
        raise InterpreterError(f"Unary operation {expression.op.__class__.__name__} is not supported.")

#处理匿名表达式的
def evaluate_lambda(
    lambda_expression: ast.Lambda,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Callable:
    args = [arg.arg for arg in lambda_expression.args.args]

    def lambda_func(*values: Any) -> Any:
        new_state = state.copy()
        for arg, value in zip(args, values):
            new_state[arg] = value
        return evaluate_ast(
            lambda_expression.body,
            new_state,
            static_tools,
            custom_tools,
            authorized_imports,
        )

    return lambda_func


def evaluate_while(
    while_loop: ast.While,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> None:
    iterations = 0
    while evaluate_ast(while_loop.test, state, static_tools, custom_tools, authorized_imports):
        for node in while_loop.body:
            try:
                evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
            except BreakException:
                return None
            except ContinueException:
                break
        iterations += 1
        if iterations > MAX_WHILE_ITERATIONS:
            raise InterpreterError(f"Maximum number of {MAX_WHILE_ITERATIONS} iterations in While loop exceeded")
    return None


def create_function(
    func_def: ast.FunctionDef,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Callable:
    """在沙箱中创建函数 —— 返回一个闭包，调用时用 AST 解释器执行函数体。

    不使用 exec/compile，而是把函数体的 AST 节点保存在闭包中，
    每次调用时创建局部作用域（func_state），然后逐语句 evaluate_ast 执行。

    return 语句通过 ReturnException 异常实现：
        evaluate_ast 遇到 ast.Return → raise ReturnException(value)
        → 这里的 except ReturnException 捕获 → 拿到返回值
    """
    source_code = ast.unparse(func_def)  # 保存源码（用于调试和 __source__ 属性）

    def new_func(*args: Any, **kwargs: Any) -> Any:
        # ===== 创建函数的局部作用域 =====
        # 浅拷贝外层 state，函数内部能看到外层变量，但赋值不会影响外层
        func_state = state.copy()

        # ===== 参数绑定 =====
        arg_names = [arg.arg for arg in func_def.args.args]  # 形参名列表，如 ["data", "count"]

        # 计算默认值（def func(x, y=10) 中的 10）
        default_values = [
            evaluate_ast(d, state, static_tools, custom_tools, authorized_imports) for d in func_def.args.defaults
        ]

        # 默认值从右往左对齐：def func(a, b, c=1, d=2) → defaults = {"c": 1, "d": 2}
        defaults = dict(zip(arg_names[-len(default_values) :], default_values))

        # 绑定位置参数：func("hello", 5) → func_state["data"] = "hello", func_state["count"] = 5
        # arg_name是名字，args是实际参数
        for name, value in zip(arg_names, args):
            func_state[name] = value

        # 绑定关键字参数：func(count=5) → func_state["count"] = 5
        for name, value in kwargs.items():
            func_state[name] = value

        # 处理 *args 可变参数
        if func_def.args.vararg:
            vararg_name = func_def.args.vararg.arg
            func_state[vararg_name] = args

        # 处理 **kwargs 可变关键字参数
        if func_def.args.kwarg:
            kwarg_name = func_def.args.kwarg.arg
            func_state[kwarg_name] = kwargs

        # 未传入的参数使用默认值
        for name, value in defaults.items():
            if name not in func_state:
                func_state[name] = value

        # 如果是方法调用（第一个参数是 self），设置 self 和 __class__
        if func_def.args.args and func_def.args.args[0].arg == "self":
            if args:
                func_state["self"] = args[0]
                func_state["__class__"] = args[0].__class__

        # ===== 执行函数体 =====
        result = None
        try:
            #执行函数体，放到我们的evaluate_ast系统中
            for stmt in func_def.body:
                result = evaluate_ast(stmt, func_state, static_tools, custom_tools, authorized_imports)
        except ReturnException as e:
            # return 语句触发的异常 → 拿到返回值
            result = e.value

        # __init__ 按 Python 规范必须返回 None
        if func_def.name == "__init__":
            return None

        return result

    # 保存元信息（用于调试、序列化等）
    new_func.__ast__ = func_def
    new_func.__source__ = source_code
    new_func.__name__ = func_def.name
    #返回一个可调用对象
    return new_func


def evaluate_function_def(
    func_def: ast.FunctionDef,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Callable:
    """处理 def 语句：创建函数对象并存入 custom_tools。

    存入 custom_tools（而非 state）意味着：
        - 后续代码可以通过 evaluate_call 的查找链找到它
        - 可以被重新 def 覆盖（custom_tools 可覆盖，static_tools 不可）
    """
    custom_tools[func_def.name] = create_function(func_def, state, static_tools, custom_tools, authorized_imports)
    return custom_tools[func_def.name]


def evaluate_class_def(
    class_def: ast.ClassDef,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> type:
    # class Person(Base): ...
    # class_def.name 就是 "Person"
    class_name = class_def.name

    # 先把父类节点递归求值成真正的类对象。
    # 例如 class Dog(Animal): ... 这里会把 Name("Animal") 解析成 Animal 类本身。
    bases = [evaluate_ast(base, state, static_tools, custom_tools, authorized_imports) for base in class_def.bases]

    # 决定用哪个 metaclass 来创建最终的类对象。
    # 默认所有普通类都由 type 创建；如果父类里有自定义 metaclass，就沿用它。
    metaclass = type
    for base in bases:
        base_metaclass = type(base)
        if base_metaclass is not type:
            metaclass = base_metaclass
            break

    # 类体在真正变成类之前，会先执行到一个“类命名空间”里。
    # 大多数时候这个命名空间就是 dict；有些 metaclass 会通过 __prepare__ 自定义它。
    if hasattr(metaclass, "__prepare__"):
        class_dict = metaclass.__prepare__(class_name, bases)
    else:
        class_dict = {}

    # 逐条处理类体里的语句，把方法、类属性、注解等都收集到 class_dict 中。
    for stmt in class_def.body:
        if isinstance(stmt, ast.FunctionDef):
            # def method(...): ...
            # 方法先按普通函数去创建，随后作为类属性挂到 class_dict 上。
            class_dict[stmt.name] = evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
        elif isinstance(stmt, ast.AnnAssign):
            # 处理带类型注解的类属性，例如:
            #   age: int = 18
            #   name: str
            if stmt.value:
                value = evaluate_ast(stmt.value, state, static_tools, custom_tools, authorized_imports)
            target = stmt.target
            # 按左侧目标的不同形态分别处理。
            if isinstance(target, ast.Name):
                # 简单类属性注解: x: int = 1
                annotation = evaluate_ast(stmt.annotation, state, static_tools, custom_tools, authorized_imports)
                #这里可以得到：class_dict["__annotations__"]["age"] = int
                class_dict.setdefault("__annotations__", {})[target.id] = annotation
                # 如果还带默认值，就把值也写入类命名空间。
                if stmt.value:
                    class_dict[target.id] = value
            elif isinstance(target, ast.Attribute):
                # 属性注解，例如 obj.attr: int = 1
                # 属性的内容不存dict，因为不是自己的值
                obj = evaluate_ast(target.value, class_dict, static_tools, custom_tools, authorized_imports)
                if stmt.value:
                    setattr(obj, target.attr, value)
            elif isinstance(target, ast.Subscript):
                # 下标注解，例如 mapping[key]: int = 1
                container = evaluate_ast(target.value, class_dict, static_tools, custom_tools, authorized_imports)
                index = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
                # If there's a value assignment, set the item
                if stmt.value:
                    container[index] = value
            else:
                raise InterpreterError(f"Unsupported AnnAssign target in class body: {type(target).__name__}")
        elif isinstance(stmt, ast.Assign):
            # 处理普通类属性赋值，例如:
            #   kind = "human"
            value = evaluate_ast(stmt.value, state, static_tools, custom_tools, authorized_imports)
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    class_dict[target.id] = value
                elif isinstance(target, ast.Attribute):
                    obj = evaluate_ast(target.value, class_dict, static_tools, custom_tools, authorized_imports)
                    setattr(obj, target.attr, value)
        elif isinstance(stmt, ast.Pass):
            # 空类体或占位 pass，直接跳过。
            pass
        elif (
            isinstance(stmt, ast.Expr)
            and stmt == class_def.body[0]
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            # 类体第一条如果是字符串常量，就把它当作类的 docstring。
            class_dict["__doc__"] = stmt.value.value
        else:
            # 为了保持解释器可控，类体里只支持上面列出的几类语句。
            raise InterpreterError(f"Unsupported statement in class body: {stmt.__class__.__name__}")

    # 最后调用 metaclass(name, bases, namespace) 真正造出类对象。
    new_class = metaclass(class_name, tuple(bases), class_dict)
    # 把类名注册到当前 state，后续代码才能直接使用 Person、MyClass 这样的名字。
    state[class_name] = new_class
    return new_class


def evaluate_annassign(
    annassign: ast.AnnAssign,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    # If there's a value to assign, evaluate it
    if annassign.value:
        value = evaluate_ast(annassign.value, state, static_tools, custom_tools, authorized_imports)
        # Set the value for the target
        set_value(annassign.target, value, state, static_tools, custom_tools, authorized_imports)
        return value
    # For declarations without values (x: int), just return None
    return None


def evaluate_augassign(
    expression: ast.AugAssign,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    """处理增量赋值：x += 1, count -= 1, data *= 2 等。

    流程：读取当前值 → 计算右边的值 → 根据运算符计算新值 → 写回。
    支持所有 Python 增量运算符：+=, -=, *=, /=, %=, **=, //=, &=, |=, ^=, <<=, >>=
    """
    def get_current_value(target: ast.AST) -> Any:
        """读取赋值目标的当前值（支持变量、索引、属性等）"""
        if isinstance(target, ast.Name):
            return state.get(target.id, 0)
        elif isinstance(target, ast.Subscript):
            obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
            key = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
            return obj[key]
        elif isinstance(target, ast.Attribute):
            obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
            return getattr(obj, target.attr)
        elif isinstance(target, ast.Tuple):
            return tuple(get_current_value(elt) for elt in target.elts)
        elif isinstance(target, ast.List):
            return [get_current_value(elt) for elt in target.elts]
        else:
            raise InterpreterError("AugAssign not supported for {type(target)} targets.")

    # 第1步：读取当前值
    current_value = get_current_value(expression.target)
    # 第2步：计算右边的值
    value_to_add = evaluate_ast(expression.value, state, static_tools, custom_tools, authorized_imports)

    # 第3步：根据运算符计算新值
    if isinstance(expression.op, ast.Add):          # +=
        if isinstance(current_value, list):
            if not isinstance(value_to_add, list):
                raise InterpreterError(f"Cannot add non-list value {value_to_add} to a list.")
            current_value += value_to_add
        else:
            current_value += value_to_add
    elif isinstance(expression.op, ast.Sub):        # -=
        current_value -= value_to_add
    elif isinstance(expression.op, ast.Mult):       # *=
        current_value *= value_to_add
    elif isinstance(expression.op, ast.Div):        # /=
        current_value /= value_to_add
    elif isinstance(expression.op, ast.Mod):        # %=
        current_value %= value_to_add
    elif isinstance(expression.op, ast.Pow):        # **=
        current_value **= value_to_add
    elif isinstance(expression.op, ast.FloorDiv):   # //=
        current_value //= value_to_add
    elif isinstance(expression.op, ast.BitAnd):     # &=
        current_value &= value_to_add
    elif isinstance(expression.op, ast.BitOr):      # |=
        current_value |= value_to_add
    elif isinstance(expression.op, ast.BitXor):     # ^=
        current_value ^= value_to_add
    elif isinstance(expression.op, ast.LShift):     # <<=
        current_value <<= value_to_add
    elif isinstance(expression.op, ast.RShift):     # >>=
        current_value >>= value_to_add
    else:
        raise InterpreterError(f"Operation {type(expression.op).__name__} is not supported.")

    # 第4步：把计算后的新值写回目标
    set_value(
        expression.target,
        current_value,
        state,
        static_tools,
        custom_tools,
        authorized_imports,
    )

    return current_value



def evaluate_boolop(
    node: ast.BoolOp,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    # Python 的 and/or 不是简单返回 True/False，而是返回参与运算的“原值”之一。
    # and: 遇到第一个 falsy 值就短路返回；如果都为 truthy，则返回最后一个值。
    # or: 遇到第一个 truthy 值就短路返回；如果都为 falsy，则返回最后一个值。
    is_short_circuit_value = (lambda x: not x) if isinstance(node.op, ast.And) else (lambda x: bool(x))

    # 按从左到右顺序依次求值，模拟 Python 的短路行为。
    for value in node.values:
        result = evaluate_ast(value, state, static_tools, custom_tools, authorized_imports)
        # 一旦遇到当前运算符的短路值，就立刻返回，后面的表达式不再执行。
        if is_short_circuit_value(result):
            return result

    # 如果整个链条都没有提前短路，结果就是最后一个被求值的原始值。
    return result


def evaluate_binop(
    binop: ast.BinOp,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    # 二元运算的通用套路是：
    # 1. 先递归求左操作数
    # 2. 再递归求右操作数
    # 3. 最后根据运算符类型执行真正的 Python 运算
    left_val = evaluate_ast(binop.left, state, static_tools, custom_tools, authorized_imports)
    right_val = evaluate_ast(binop.right, state, static_tools, custom_tools, authorized_imports)

    # 根据 AST 中记录的运算符节点类型，分发到对应的实际运算。
    if isinstance(binop.op, ast.Add):
        return left_val + right_val
    elif isinstance(binop.op, ast.Sub):
        return left_val - right_val
    elif isinstance(binop.op, ast.Mult):
        return left_val * right_val
    elif isinstance(binop.op, ast.Div):
        return left_val / right_val
    elif isinstance(binop.op, ast.Mod):
        return left_val % right_val
    elif isinstance(binop.op, ast.Pow):
        return left_val**right_val
    elif isinstance(binop.op, ast.FloorDiv):
        return left_val // right_val
    elif isinstance(binop.op, ast.BitAnd):
        return left_val & right_val
    elif isinstance(binop.op, ast.BitOr):
        return left_val | right_val
    elif isinstance(binop.op, ast.BitXor):
        return left_val ^ right_val
    elif isinstance(binop.op, ast.LShift):
        return left_val << right_val
    elif isinstance(binop.op, ast.RShift):
        return left_val >> right_val
    else:
        # 没有显式支持的二元运算一律视为未实现。
        raise NotImplementedError(f"Binary operation {type(binop.op).__name__} is not implemented.")


def evaluate_assign(
    assign: ast.Assign,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    """处理赋值语句：x = 10, a, b = func(), x = y = value

    流程：先计算右边的值（递归 evaluate_ast），再通过 set_value 存入 state。
    """
    # 第1步：计算等号右边的值
    result = evaluate_ast(assign.value, state, static_tools, custom_tools, authorized_imports)
    if len(assign.targets) == 1:
        # 常见情况：x = 10 或 a, b = 1, 2（单目标，可能是元组解包）
        target = assign.targets[0]
        set_value(target, result, state, static_tools, custom_tools, authorized_imports)
    else:
        # 多目标赋值：x = y = 10（同一个值赋给多个变量）
        expanded_values = []
        for tgt in assign.targets:
            if isinstance(tgt, ast.Starred):
                expanded_values.extend(result)
            else:
                expanded_values.append(result)

        for tgt, val in zip(assign.targets, expanded_values):
            set_value(tgt, val, state, static_tools, custom_tools, authorized_imports)
    # 返回赋值的值（用于 evaluate_python_code 中记录最后一个表达式的结果）
    return result


def set_value(
    target: ast.AST,
    value: Any,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> None:
    """把值存入正确的位置 —— 赋值语句的核心写入逻辑。

    根据赋值目标的类型分发：
        x = 10          → ast.Name → state["x"] = 10
        a, b = 1, 2     → ast.Tuple → 解包后递归 set_value
        data[0] = "hi"  → ast.Subscript → 修改容器元素
        obj.name = "x"  → ast.Attribute → 修改对象属性

    安全机制：如果赋值目标是 static_tools 中的工具名，拒绝赋值（防止 LLM 覆盖工具）。
    """
    if isinstance(target, ast.Name):
        # 变量赋值：x = 10
        # 安全检查：禁止覆盖 static_tools 中的工具（如 web_search, print, len 等）
        if target.id in static_tools:
            raise InterpreterError(f"Cannot assign to name '{target.id}': doing this would erase the existing tool!")
        state[target.id] = value
    elif isinstance(target, ast.Tuple):
        # 元组解包：a, b = 1, 2 → 递归处理每个元素
        if not isinstance(value, tuple):
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                value = tuple(value)  # 列表等可迭代对象转为元组
            else:
                raise InterpreterError("Cannot unpack non-tuple value")
        if len(target.elts) != len(value):
            raise InterpreterError("Cannot unpack tuple of wrong size")
        for i, elem in enumerate(target.elts):
            set_value(elem, value[i], state, static_tools, custom_tools, authorized_imports)
    elif isinstance(target, ast.Subscript):
        # 索引赋值：data[0] = "hello", dict["key"] = value
        obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
        key = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
        obj[key] = value
    elif isinstance(target, ast.Attribute):
        # 属性赋值：obj.name = "test"
        obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
        setattr(obj, target.attr, value)


def evaluate_call(
    call: ast.Call,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    """执行函数调用 —— 沙箱中最复杂的操作。

    处理所有形式的函数调用：
        web_search("hello")       → ast.Name 直接调用
        obj.method(x)             → ast.Attribute 方法调用
        get_func()("hello")      → ast.Call 链式调用
        (lambda x: x+1)(5)       → ast.Lambda 调用
        funcs[0]("hello")        → ast.Subscript(下表取值) 索引调用

    执行流程：1.找到函数 → 2.计算参数 → 3.安全检查 → 4.调用
    """

    # ===== 前置检查：call.func 必须是合法的可调用形式 =====
    if not isinstance(call.func, (ast.Call, ast.Lambda, ast.Attribute, ast.Name, ast.Subscript)):
        raise InterpreterError(f"This is not a correct function: {call.func}).")

    func, func_name = None, None

    # ===== 第1步：根据调用形式找到函数对象（func） =====
    # 因为call节点内部还包装着其它节点
    if isinstance(call.func, ast.Call):
        # 链式调用：get_func()("hello") → 先递归执行 get_func() 拿到返回的函数
        func = evaluate_ast(call.func, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(call.func, ast.Lambda):
        # lambda 调用：(lambda x: x+1)(5) → 先构造 lambda 函数对象
        func = evaluate_ast(call.func, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(call.func, ast.Attribute):
        # 方法调用：obj.method(x) → 先算出 obj，再从 obj 上取 method
        obj = evaluate_ast(call.func.value, state, static_tools, custom_tools, authorized_imports)
        func_name = call.func.attr
        if not hasattr(obj, func_name):
            raise InterpreterError(f"Object {obj} has no attribute {func_name}")
        func = getattr(obj, func_name)
    elif isinstance(call.func, ast.Name):
        # 直接调用：web_search("hello"), len(data) → 按优先级查找函数
        func_name = call.func.id
        if func_name in state:
            # 优先级1：state 中的变量（用户在代码中 def 的函数、import 的模块等）
            func = state[func_name]
        elif func_name in static_tools:
            # 优先级2：static_tools（Agent 工具 + BASE_PYTHON_TOOLS 内置安全函数）
            func = static_tools[func_name]
        elif func_name in custom_tools:
            # 优先级3：custom_tools（可被覆盖的工具）
            func = custom_tools[func_name]
        elif func_name in ERRORS:
            # 优先级4：Python 内置异常类（ValueError, TypeError 等，用于 raise）
            func = ERRORS[func_name]
        else:
            # 都找不到 → 报错：函数未授权
            raise InterpreterError(
                f"Forbidden function evaluation: '{call.func.id}' is not among the explicitly allowed tools or defined/imported in the preceding code"
            )
    elif isinstance(call.func, ast.Subscript):
        # 索引调用：funcs[0]("hello") → 先算出 funcs[0]，检查是否可调用
        func = evaluate_ast(call.func, state, static_tools, custom_tools, authorized_imports)
        if not callable(func):
            raise InterpreterError(f"This is not a correct function: {call.func}).")
        func_name = None

    # ===== 第2步：计算所有参数 =====

    # 位置参数
    args = []
    for arg in call.args:
        if isinstance(arg, ast.Starred):
            # *args 解包：func(*my_list) → 展开 my_list 的每个元素
            args.extend(evaluate_ast(arg.value, state, static_tools, custom_tools, authorized_imports))
        else:
            # 普通参数：递归计算值
            args.append(evaluate_ast(arg, state, static_tools, custom_tools, authorized_imports))

    # 关键字参数
    kwargs = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            # **kwargs 解包：func(**my_dict) → 展开字典
            starred_dict = evaluate_ast(keyword.value, state, static_tools, custom_tools, authorized_imports)
            if not isinstance(starred_dict, dict):
                raise InterpreterError(f"Cannot unpack non-dict value in **kwargs: {type(starred_dict).__name__}")
            kwargs.update(starred_dict)
        else:
            # 普通关键字参数：func(name="hello")
            kwargs[keyword.arg] = evaluate_ast(keyword.value, state, static_tools, custom_tools, authorized_imports)

    # ===== 第3步：调用函数（含特殊处理和安全检查） =====

    if func_name == "super":
        # 特殊处理 super()：需要从 state 中获取当前类和实例
        #三个super分别是 super(),super(class), super(class,obj)
        if not args:
            if "__class__" in state and "self" in state:
                return super(state["__class__"], state["self"])
            else:
                raise InterpreterError("super() needs at least one argument")
        cls = args[0]
        if not isinstance(cls, type):
            raise InterpreterError("super() argument 1 must be type")
        if len(args) == 1:
            return super(cls)
        elif len(args) == 2:
            instance = args[1]
            return super(cls, instance)
        else:
            raise InterpreterError("super() takes at most 2 arguments")
    elif func_name == "print":
        # 特殊处理 print()：不打印到终端，把内容写入 _print_outputs 收集器
        # 这就是 print 重定向的真正实现（custom_print 只是占位符）
        state["_print_outputs"] += " ".join(map(str, args)) + "\n"
        return None
    else:
        # ===== 通用路径：安全检查后调用 =====

        # 安全检查1：禁止调用未授权的内置函数
        # 比如 compile(), exec() 等危险函数，即使通过某种方式拿到了引用也不能调用
        if (inspect.getmodule(func) == builtins) and inspect.isbuiltin(func) and (func not in static_tools.values()):
            raise InterpreterError(
                f"Invoking a builtin function that has not been explicitly added as a tool is not allowed ({func_name})."
            )
        # 安全检查2：禁止调用 dunder 方法（__init__, __str__, __repr__ 除外）
        # 防止通过 obj.__class__.__bases__ 等方式逃逸沙箱
        if (
            hasattr(func, "__name__")
            and func.__name__.startswith("__")
            and func.__name__.endswith("__")
            and (func.__name__ not in static_tools)
            and (func.__name__ not in ALLOWED_DUNDER_METHODS)
        ):
            raise InterpreterError(f"Forbidden call to dunder function: {func.__name__}")

        # 通过所有安全检查 → 真正调用函数并返回结果
        return func(*args, **kwargs)


def evaluate_subscript(
    subscript: ast.Subscript,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    # 先计算方括号里的内容，比如:
    # data[0]        -> index 是 0
    # data["name"]   -> index 是 "name"
    # text[1:3]      -> index 是 slice(1, 3, None)
    # 再计算被索引的对象，比如:
    # data[0] 里的 data
    # info["x"] 里的 info
    index = evaluate_ast(subscript.slice, state, static_tools, custom_tools, authorized_imports)
    value = evaluate_ast(subscript.value, state, static_tools, custom_tools, authorized_imports)
    try:
        return value[index]
    # 下面几种错误都说明“索引失败了”
    # KeyError: 字典里没有这个 key
    # IndexError: 列表/元组下标越界
    # TypeError: 对不支持索引的对象用了 []
    except (KeyError, IndexError, TypeError) as e:
        error_message = f"Could not index {value} with '{index}': {type(e).__name__}: {e}"
        # 如果是“字典 + 字符串 key”的场景，
        # 顺便帮用户做一个近似匹配提示
        # 例如写成 data['nmae']，可能提示 data['name']
        if isinstance(index, str) and isinstance(value, Mapping):
            close_matches = difflib.get_close_matches(index, list(value.keys()))
            if len(close_matches) > 0:
                error_message += f". Maybe you meant one of these indexes instead: {str(close_matches)}"
         # 把底层 Python 异常包装成解释器自己的错误类型
        raise InterpreterError(error_message) from e


def evaluate_name(
    name: ast.Name,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    if name.id in state:
        return state[name.id]
    elif name.id in static_tools:
        return safer_func(static_tools[name.id], static_tools=static_tools, authorized_imports=authorized_imports)
    elif name.id in custom_tools:
        return custom_tools[name.id]
    elif name.id in ERRORS:
        return ERRORS[name.id]
    close_matches = difflib.get_close_matches(name.id, list(state.keys()))
    if len(close_matches) > 0:
        return state[close_matches[0]]
    raise InterpreterError(f"The variable `{name.id}` is not defined.")


def evaluate_condition(
    condition: ast.Compare,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> bool | object:
    # Compare 节点既能表示简单比较：
    #   x > 1
    # 也能表示链式比较：
    #   1 < x < 10
    # Python 的链式比较不是拆成两个互不相关的表达式，而是要按顺序逐段比较。
    result = True

    # 先求最左边的值，后面会不断把 right 滚动到下一轮的 left。
    left = evaluate_ast(condition.left, state, static_tools, custom_tools, authorized_imports)
    for i, (op, comparator) in enumerate(zip(condition.ops, condition.comparators)):
        op = type(op)

        # 当前比较链这一段的右操作数。
        right = evaluate_ast(comparator, state, static_tools, custom_tools, authorized_imports)
        if op == ast.Eq:
            current_result = left == right
        elif op == ast.NotEq:
            current_result = left != right
        elif op == ast.Lt:
            current_result = left < right
        elif op == ast.LtE:
            current_result = left <= right
        elif op == ast.Gt:
            current_result = left > right
        elif op == ast.GtE:
            current_result = left >= right
        elif op == ast.Is:
            current_result = left is right
        elif op == ast.IsNot:
            current_result = left is not right
        elif op == ast.In:
            current_result = left in right
        elif op == ast.NotIn:
            current_result = left not in right
        else:
            raise InterpreterError(f"Unsupported comparison operator: {op}")

        # 链式比较中只要有一段失败，整个比较立即为 False。
        if current_result is False:
            return False

        # 记录当前阶段比较结果。对于链式比较，所有阶段都必须成立。
        result = current_result if i == 0 else (result and current_result)

        # 链式比较的关键：
        #   1 < x < 10
        # 在比较完 1 < x 之后，下一轮要继续比较 x < 10，
        # 所以把当前的 right 变成下一轮的 left。
        left = right
    return result


def evaluate_if(
    if_statement: ast.If,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    # ast.If 对应:
    #   if cond:
    #       ...
    #   else:
    #       ...
    #
    # 解释器要先求条件，再决定执行 body 还是 orelse。
    result = None

    # 先计算 if 条件。
    test_result = evaluate_ast(if_statement.test, state, static_tools, custom_tools, authorized_imports)
    if test_result:
        # 条件为真时，顺序执行 if 分支。
        for line in if_statement.body:
            line_result = evaluate_ast(line, state, static_tools, custom_tools, authorized_imports)
            # 记录最近一个非 None 的语句结果，作为整个 if 语句的返回值候选。
            if line_result is not None:
                result = line_result
    else:
        # 条件为假时，顺序执行 else / elif 对应的 orelse 分支。
        for line in if_statement.orelse:
            line_result = evaluate_ast(line, state, static_tools, custom_tools, authorized_imports)
            if line_result is not None:
                result = line_result
    return result


def evaluate_for(
    for_loop: ast.For,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    # ast.For 对应:
    #   for target in iterable:
    #       ...
    #
    # 解释器需要先求值 iterable，再把每轮迭代值绑定到 target 上，最后执行循环体。
    result = None

    # 先计算 for ... in ... 里的迭代对象。
    iterator = evaluate_ast(for_loop.iter, state, static_tools, custom_tools, authorized_imports)
    for counter in iterator:
        # 把当前这一轮的值写入循环变量目标。
        # target 可能不只是简单名字，也可能是解包目标，例如:
        #   for a, b in pairs:
        set_value(
            for_loop.target,
            counter,
            state,
            static_tools,
            custom_tools,
            authorized_imports,
        )

        # 顺序执行循环体中的每条语句。
        for node in for_loop.body:
            try:
                line_result = evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
                if line_result is not None:
                    result = line_result
            except BreakException:
                # break: 立即结束整个 for 循环。
                return result
            except ContinueException:
                # continue: 结束当前这一轮，进入下一轮迭代。
                break
    return result


def _evaluate_comprehensions(
    comprehensions: list[ast.comprehension],
    evaluate_element: Callable[[dict[str, Any]], Any],
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Generator[Any, None, None]:
    """
    Recursively evaluate nested comprehensions and yields elements.

    Args:
        comprehensions (`list[ast.comprehension]`): Comprehensions to evaluate.
        evaluate_element (`Callable`): Function that evaluates the final element when comprehensions are exhausted.
        state (`dict[str, Any]`): Current evaluation state.
        static_tools (`dict[str, Callable]`): Static tools.
        custom_tools (`dict[str, Callable]`): Custom tools.
        authorized_imports (`list[str]`): Authorized imports.

    Yields:
        `Any`: Individual elements produced by the comprehension
    """
    # 这是推导式共享的递归“展开器”。
    # 它把:
    #   [x * 2 for x in xs if x > 0 for y in ys]
    # 这类结构拆成“逐层 for + if 过滤 + 最终元素求值”的递归过程。

    # 递归终点：如果没有剩余的 comprehension 生成器了，
    # 说明所有 for/if 条件都已经满足，此时真正计算最终元素并产出。
    if not comprehensions:
        yield evaluate_element(state)
        return

    # 每次只处理当前最外层的一个 comprehension，
    # 剩余层级交给递归继续展开。
    comprehension = comprehensions[0]
    iter_value = evaluate_ast(comprehension.iter, state, static_tools, custom_tools, authorized_imports)
    for value in iter_value:
        # 推导式每一轮都基于当前状态的副本执行，
        # 避免中间绑定的循环变量直接污染外层 state。
        new_state = state.copy()
        set_value(comprehension.target, value, new_state, static_tools, custom_tools, authorized_imports)

        # 这一层 comprehension 可能自带若干 if 过滤条件，
        # 只有全部满足，才继续进入下一层递归。
        if all(
            evaluate_ast(if_clause, new_state, static_tools, custom_tools, authorized_imports)
            for if_clause in comprehension.ifs
        ):
            # 当前层满足后，继续展开剩余 generators。
            yield from _evaluate_comprehensions(
                comprehensions[1:], evaluate_element, new_state, static_tools, custom_tools, authorized_imports
            )


def evaluate_listcomp(
    listcomp: ast.ListComp,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> list[Any]:
    # 列表推导式本质上就是：
    # 1. 用 _evaluate_comprehensions 逐个生成元素
    # 2. 最后收集成 list
    return list(
        _evaluate_comprehensions(
            listcomp.generators,
            lambda comp_state: evaluate_ast(listcomp.elt, comp_state, static_tools, custom_tools, authorized_imports),
            state,
            static_tools,
            custom_tools,
            authorized_imports,
        )
    )


def evaluate_setcomp(
    setcomp: ast.SetComp,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> set[Any]:
    # 集合推导式和列表推导式共享同一套展开逻辑，
    # 唯一差别只是最后把结果收集成 set。
    return set(
        _evaluate_comprehensions(
            setcomp.generators,
            lambda comp_state: evaluate_ast(setcomp.elt, comp_state, static_tools, custom_tools, authorized_imports),
            state,
            static_tools,
            custom_tools,
            authorized_imports,
        )
    )


def evaluate_dictcomp(
    dictcomp: ast.DictComp,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> dict[Any, Any]:
    # 字典推导式同样复用统一展开逻辑，
    # 只是最终元素不再是单个值，而是 (key, value) 二元组。
    return dict(
        _evaluate_comprehensions(
            dictcomp.generators,
            lambda comp_state: (
                evaluate_ast(dictcomp.key, comp_state, static_tools, custom_tools, authorized_imports),
                evaluate_ast(dictcomp.value, comp_state, static_tools, custom_tools, authorized_imports),
            ),
            state,
            static_tools,
            custom_tools,
            authorized_imports,
        )
    )


def evaluate_try(
    try_node: ast.Try,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> None:
    try:
        for stmt in try_node.body:
            evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
    except Exception as e:
        matched = False
        for handler in try_node.handlers:
            if handler.type is None or isinstance(
                e,
                evaluate_ast(handler.type, state, static_tools, custom_tools, authorized_imports),
            ):
                matched = True
                if handler.name:
                    state[handler.name] = e
                for stmt in handler.body:
                    evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
                break
        if not matched:
            raise e
    else:
        if try_node.orelse:
            for stmt in try_node.orelse:
                evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
    finally:
        if try_node.finalbody:
            for stmt in try_node.finalbody:
                evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)


def evaluate_raise(
    raise_node: ast.Raise,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> None:
    if raise_node.exc is not None:
        exc = evaluate_ast(raise_node.exc, state, static_tools, custom_tools, authorized_imports)
    else:
        exc = None
    if raise_node.cause is not None:
        cause = evaluate_ast(raise_node.cause, state, static_tools, custom_tools, authorized_imports)
    else:
        cause = None
    if exc is not None:
        if cause is not None:
            raise exc from cause
        else:
            raise exc
    else:
        raise InterpreterError("Re-raise is not supported without an active exception")


def evaluate_assert(
    assert_node: ast.Assert,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> None:
    test_result = evaluate_ast(assert_node.test, state, static_tools, custom_tools, authorized_imports)
    if not test_result:
        if assert_node.msg:
            msg = evaluate_ast(assert_node.msg, state, static_tools, custom_tools, authorized_imports)
            raise AssertionError(msg)
        else:
            # Include the failing condition in the assertion message
            test_code = ast.unparse(assert_node.test)
            raise AssertionError(f"Assertion failed: {test_code}")


def evaluate_with(
    with_node: ast.With,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> None:
    contexts = []
    for item in with_node.items:
        context_expr = evaluate_ast(item.context_expr, state, static_tools, custom_tools, authorized_imports)
        enter_result = context_expr.__enter__()
        contexts.append(context_expr)
        if item.optional_vars:
            state[item.optional_vars.id] = enter_result

    try:
        for stmt in with_node.body:
            evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
    except Exception as e:
        # exc_info tracks the active exception as we unwind (from innermost context manager)
        # Resetting it to (None, None, None) signals suppression to the remaining outer managers
        exc_info = (type(e), e, e.__traceback__)
        for context in reversed(contexts):
            try:
                if context.__exit__(*exc_info):
                    exc_info = (None, None, None)  # suppressed; outer CMs see no exception
            except Exception as exit_exc:
                exc_info = (type(exit_exc), exit_exc, exit_exc.__traceback__)  # new exc replaces active
        if exc_info[1] is not None:
            raise exc_info[1].with_traceback(exc_info[2])
    else:
        for context in reversed(contexts):
            context.__exit__(None, None, None)


def get_safe_module(raw_module, authorized_imports, visited=None):
    """创建模块的安全副本，递归处理嵌套的子模块。

    为什么需要这个？因为模块内部可能嵌套其他模块的引用。
    比如导入 numpy 后，numpy.os 可能指向 os 模块。
    这个函数递归遍历模块的所有属性，对子模块也做同样的处理。

    Args:
        raw_module: 原始导入的模块对象
        authorized_imports: 导入白名单
        visited: 已访问模块的 id 集合（防止循环引用导致无限递归）
    """
    # 不是模块对象（是函数、类等）→ 直接返回，无需处理
    if not isinstance(raw_module, ModuleType):
        return raw_module

    # 防止循环引用：A 模块引用 B，B 又引用 A → 用 visited 集合记录已处理的模块
    if visited is None:
        visited = set()

    module_id = id(raw_module)
    if module_id in visited:
        return raw_module  # 已经处理过，直接返回原始模块避免无限递归

    visited.add(module_id)

    # 创建一个新的空模块壳（同名但属性为空）
    safe_module = ModuleType(raw_module.__name__)

    # 逐个复制属性到安全副本，遇到子模块就递归处理
    for attr_name in dir(raw_module):
        try:
            attr_value = getattr(raw_module, attr_name)
        except (ImportError, AttributeError) as e:
            # 某些模块有懒加载属性，访问时可能报错 → 跳过
            logger.info(
                f"Skipping import error while copying {raw_module.__name__}.{attr_name}: {type(e).__name__} - {e}"
            )
            continue
        # 如果属性是子模块，递归创建安全副本
        if isinstance(attr_value, ModuleType):
            attr_value = get_safe_module(attr_value, authorized_imports, visited=visited)

        setattr(safe_module, attr_name, attr_value)

    return safe_module


def evaluate_import(expression, state, authorized_imports):
    """处理 import 语句 —— 沙箱安全的关键入口。

    两种形式：
        import json              → ast.Import，整个模块存入 state
        from math import sqrt    → ast.ImportFrom，只取模块中的指定名字

    安全流程：
        1. check_import_authorized() 检查模块是否在白名单中
        2. 真正导入模块
        3. get_safe_module() 创建安全副本
        4. 存入 state（后续代码通过 state 访问模块）
    """
    if isinstance(expression, ast.Import):
        # ===== import json / import json as j =====
        for alias in expression.names:
            # alias.name = "json", alias.asname = "j"（如果有 as）
            if check_import_authorized(alias.name, authorized_imports):
                raw_module = import_module(alias.name)  # 真正导入
                # 存入 state，key 是别名（有 as 用别名，没有用原名）
                state[alias.asname or alias.name] = get_safe_module(raw_module, authorized_imports)
            else:
                raise InterpreterError(
                    f"Import of {alias.name} is not allowed. Authorized imports are: {str(authorized_imports)}"
                )
        return None
    elif isinstance(expression, ast.ImportFrom):
        # ===== from math import sqrt / from json import * =====
        # expression.module = "math", expression.names = [alias(name="sqrt")]
        if check_import_authorized(expression.module, authorized_imports):
            raw_module = __import__(expression.module, fromlist=[alias.name for alias in expression.names])
            module = get_safe_module(raw_module, authorized_imports)
            if expression.names[0].name == "*":
                # from module import * → 导入所有公开名字
                if hasattr(module, "__all__"):
                    # 模块定义了 __all__ → 只导入 __all__ 中列出的名字
                    for name in module.__all__:
                        state[name] = getattr(module, name)
                else:
                    # 没有 __all__ → 导入所有不以 _ 开头的名字
                    for name in dir(module):
                        if not name.startswith("_"):
                            state[name] = getattr(module, name)
            else:
                # from math import sqrt, ceil → 逐个取出指定的名字
                for alias in expression.names:
                    if hasattr(module, alias.name):
                        state[alias.asname or alias.name] = getattr(module, alias.name)
                    else:
                        raise InterpreterError(f"Module {expression.module} has no attribute {alias.name}")
        else:
            raise InterpreterError(
                f"Import from {expression.module} is not allowed. Authorized imports are: {str(authorized_imports)}"
            )
        return None


def evaluate_generatorexp(
    genexp: ast.GeneratorExp,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Generator[Any]:
    def generator():
        for gen in genexp.generators:
            iter_value = evaluate_ast(gen.iter, state, static_tools, custom_tools, authorized_imports)
            for value in iter_value:
                new_state = state.copy()
                set_value(
                    gen.target,
                    value,
                    new_state,
                    static_tools,
                    custom_tools,
                    authorized_imports,
                )
                if all(
                    evaluate_ast(if_clause, new_state, static_tools, custom_tools, authorized_imports)
                    for if_clause in gen.ifs
                ):
                    yield evaluate_ast(
                        genexp.elt,
                        new_state,
                        static_tools,
                        custom_tools,
                        authorized_imports,
                    )

    return generator()


def evaluate_delete(
    delete_node: ast.Delete,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> None:
    """
    Evaluate a delete statement (del x, del x[y]).

    Args:
        delete_node: The AST Delete node to evaluate
        state: The current state dictionary
        static_tools: Dictionary of static tools
        custom_tools: Dictionary of custom tools
        authorized_imports: List of authorized imports
    """
    for target in delete_node.targets:
        if isinstance(target, ast.Name):
            # Handle simple variable deletion (del x)
            if target.id in state:
                del state[target.id]
            else:
                raise InterpreterError(f"Cannot delete name '{target.id}': name is not defined")
        elif isinstance(target, ast.Subscript):
            # Handle index/key deletion (del x[y])
            obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
            index = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
            try:
                del obj[index]
            except (TypeError, KeyError, IndexError) as e:
                raise InterpreterError(f"Cannot delete index/key: {str(e)}")
        else:
            raise InterpreterError(f"Deletion of {type(target).__name__} targets is not supported")


@safer_eval
def evaluate_ast(
    expression: ast.AST,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str] = BASE_BUILTIN_MODULES,
):
    """AST 解释器的核心调度器 —— 根据节点类型分发到对应的 evaluate_xxx 函数。

    这是一个递归函数：复杂的节点（如赋值、函数调用）内部会再次调用 evaluate_ast 处理子表达式。
    例如 `result = len(data)` 的执行过程：
        evaluate_ast(Assign)
          → evaluate_assign()
            → evaluate_ast(Call)          # 先算右边的 len(data)
              → evaluate_call()
                → evaluate_ast(Name)      # 算参数 data
                  → 从 state 中查找 data 的值
                → 调用 len(data的值)
            → state["result"] = 结果

    Args:
        expression: 一个 AST 节点（ast.Assign, ast.Call, ast.Name 等）
        state: 变量存储字典（跨步骤持久化）
        static_tools: 不可覆盖的工具函数（Agent 工具 + 内置安全函数）
        custom_tools: 可覆盖的工具函数
        authorized_imports: 导入白名单
    """
    # ===== 操作计数：防止无限循环 =====
    if state.setdefault("_operations_count", {"counter": 0})["counter"] >= MAX_OPERATIONS:
        raise InterpreterError(
            f"Reached the max number of operations of {MAX_OPERATIONS}. Maybe there is an infinite loop somewhere in the code, or you're just asking too many calculations."
        )
    state["_operations_count"]["counter"] += 1
    common_params = (state, static_tools, custom_tools, authorized_imports)

    # ===== 第1组：赋值语句 =====
    if isinstance(expression, ast.Assign):
        # x = 10, a, b = 1, 2 → 计算右边的值，存入 state
        return evaluate_assign(expression, *common_params)
    elif isinstance(expression, ast.AnnAssign):
        # x: int = 10 → 带类型注解的赋值
        return evaluate_annassign(expression, *common_params)
    elif isinstance(expression, ast.AugAssign):
        # x += 1, count -= 1 → 增量赋值
        return evaluate_augassign(expression, *common_params)

    # ===== 第2组：函数调用 =====
    elif isinstance(expression, ast.Call):
        # web_search("hello"), len(data), print(x) → 最复杂的一个
        return evaluate_call(expression, *common_params)

    # ===== 第3组：字面量和容器 =====
    elif isinstance(expression, ast.Constant):
        # 42, "hello", True, None → 直接返回值
        return expression.value
    elif isinstance(expression, ast.Tuple):
        # (1, 2, 3) → 递归算每个元素
        return tuple((evaluate_ast(elt, *common_params) for elt in expression.elts))
    elif isinstance(expression, ast.List):
        # [1, 2, 3] → 递归算每个元素
        return [evaluate_ast(elt, *common_params) for elt in expression.elts]
    elif isinstance(expression, ast.Dict):
        # {"a": 1} → 递归算每个 key 和 value
        keys = (evaluate_ast(k, *common_params) for k in expression.keys)
        values = (evaluate_ast(v, *common_params) for v in expression.values)
        return dict(zip(keys, values))
    elif isinstance(expression, ast.Set):
        # {1, 2, 3} → 递归算每个元素
        return set((evaluate_ast(elt, *common_params) for elt in expression.elts))

    # ===== 第4组：推导式和生成器 =====
    elif isinstance(expression, ast.GeneratorExp):
        # (x for x in range(10))
        return evaluate_generatorexp(expression, *common_params)
    elif isinstance(expression, ast.ListComp):
        # [x*2 for x in range(10)]
        return evaluate_listcomp(expression, *common_params)
    elif isinstance(expression, ast.DictComp):
        # {k: v for k, v in items}
        return evaluate_dictcomp(expression, *common_params)
    elif isinstance(expression, ast.SetComp):
        # {x for x in range(10)}
        return evaluate_setcomp(expression, *common_params)

    # ===== 第5组：运算符 =====
    elif isinstance(expression, ast.UnaryOp):
        # -x, not flag, ~n → 一元运算
        return evaluate_unaryop(expression, *common_params)
    elif isinstance(expression, ast.BinOp):
        # x + y, a * b → 二元运算
        return evaluate_binop(expression, *common_params)
    elif isinstance(expression, ast.BoolOp):
        # a and b, x or y → 布尔运算（支持短路求值）
        return evaluate_boolop(expression, *common_params)
    elif isinstance(expression, ast.Compare):
        # x > 5, a == b, 1 < x < 10 → 比较运算
        return evaluate_condition(expression, *common_params)
    elif isinstance(expression, ast.Starred):
        # *args → 解包运算符
        return evaluate_ast(expression.value, *common_params)

    # ===== 第6组：控制流 =====
    elif isinstance(expression, ast.If):
        # if/elif/else
        return evaluate_if(expression, *common_params)
    elif isinstance(expression, ast.IfExp):
        # x if condition else y → 三元表达式
        test_val = evaluate_ast(expression.test, *common_params)
        if test_val:
            return evaluate_ast(expression.body, *common_params)
        else:
            return evaluate_ast(expression.orelse, *common_params)
    elif isinstance(expression, ast.For):
        # for 循环
        return evaluate_for(expression, *common_params)
    elif isinstance(expression, ast.While):
        # while 循环
        return evaluate_while(expression, *common_params)
    elif isinstance(expression, ast.Break):
        # break → 用异常模拟，由 evaluate_for/evaluate_while 的 except 捕获
        raise BreakException()
    elif isinstance(expression, ast.Continue):
        # continue → 用异常模拟，跳到下一次迭代
        raise ContinueException()
    elif isinstance(expression, ast.Return):
        # return → 用异常模拟，由 create_function 中的 except 捕获
        raise ReturnException(evaluate_ast(expression.value, *common_params) if expression.value else None)
    elif isinstance(expression, ast.Pass):
        # pass → 什么都不做
        return None

    # ===== 第7组：函数和类定义 =====
    elif isinstance(expression, ast.FunctionDef):
        # def my_func(): ... → 在沙箱中创建函数
        return evaluate_function_def(expression, *common_params)
    elif isinstance(expression, ast.Lambda):
        # lambda x: x + 1
        return evaluate_lambda(expression, *common_params)
    elif isinstance(expression, ast.ClassDef):
        # class MyClass: ... → 在沙箱中创建类
        return evaluate_class_def(expression, *common_params)

    # ===== 第8组：属性访问和索引 =====
    elif isinstance(expression, ast.Name):
        # 变量名 result, x, data → 从 state 中查找值
        return evaluate_name(expression, *common_params)
    elif isinstance(expression, ast.Attribute):
        # obj.name, data.items() → 属性访问（dunder 属性会被拦截）
        return evaluate_attribute(expression, *common_params)
    elif isinstance(expression, ast.Subscript):
        # data[0], dict["key"], arr[1:3] → 索引/切片
        return evaluate_subscript(expression, *common_params)
    elif isinstance(expression, ast.Slice):
        # 切片对象 1:3, ::2 → 构造 slice 对象
        return slice(
            evaluate_ast(expression.lower, *common_params) if expression.lower is not None else None,
            evaluate_ast(expression.upper, *common_params) if expression.upper is not None else None,
            evaluate_ast(expression.step, *common_params) if expression.step is not None else None,
        )
    elif hasattr(ast, "Index") and isinstance(expression, ast.Index):
        # Python 3.8 兼容：旧版本的索引节点
        return evaluate_ast(expression.value, *common_params)

    # ===== 第9组：字符串格式化 =====
    elif isinstance(expression, ast.JoinedStr):
        # f"hello {name}" → f-string，拼接所有部分
        return "".join([str(evaluate_ast(v, *common_params)) for v in expression.values])
    elif isinstance(expression, ast.FormattedValue):
        # f-string 中的 {name:.2f} → 计算值并应用格式
        value = evaluate_ast(expression.value, *common_params)
        if not expression.format_spec:
            return value
        format_spec = evaluate_ast(expression.format_spec, *common_params)
        return format(value, format_spec)

    # ===== 第10组：表达式语句 =====
    elif isinstance(expression, ast.Expr):
        # 纯表达式语句（如单独一行 print(x)）→ 剥掉 Expr 壳，递归处理内容
        return evaluate_ast(expression.value, *common_params)

    # ===== 第11组：import（安全检查的关键入口） =====
    elif isinstance(expression, (ast.Import, ast.ImportFrom)):
        # import json, from math import sqrt → 白名单检查
        return evaluate_import(expression, state, authorized_imports)

    # ===== 第12组：异常处理 =====
    elif isinstance(expression, ast.Try):
        # try/except/finally
        return evaluate_try(expression, *common_params)
    elif isinstance(expression, ast.Raise):
        # raise ValueError("xxx")
        return evaluate_raise(expression, *common_params)
    elif isinstance(expression, ast.Assert):
        # assert x > 0
        return evaluate_assert(expression, *common_params)

    # ===== 第13组：上下文管理器和删除 =====
    elif isinstance(expression, ast.With):
        # with open(...) as f: → 上下文管理器
        return evaluate_with(expression, *common_params)
    elif isinstance(expression, ast.Delete):
        # del x → 从 state 中删除变量
        return evaluate_delete(expression, *common_params)

    # ===== 兜底：不支持的语法一律拒绝 =====
    else:
        raise InterpreterError(f"{expression.__class__.__name__} is not supported.")


class FinalAnswerException(BaseException):
    """final_answer() 被调用时抛出的特殊异常，用于立即终止代码执行并返回答案。

    继承自 BaseException 而非 Exception，是为了防止 LLM 生成的代码中的
    `except Exception` 子句意外捕获它，导致 final_answer 调用被吞掉。
    BaseException 只被 `except BaseException` 或裸 `except:` 捕获，更安全。

    工作流程：
        1. final_answer(value) → 触发 FinalAnswerException(value)
        2. evaluate_python_code 的 except FinalAnswerException 捕获
        3. 返回 (e.value, is_final_answer=True)
        4. CodeAgent 看到 is_final_answer=True，终止 ReAct 循环
    """

    def __init__(self, value):
        self.value = value


def evaluate_python_code(
    code: str,
    static_tools: dict[str, Callable] | None = None,
    custom_tools: dict[str, Callable] | None = None,
    state: dict[str, Any] | None = None,
    authorized_imports: list[str] = BASE_BUILTIN_MODULES,
    max_print_outputs_length: int = DEFAULT_MAX_LEN_OUTPUT,
    timeout_seconds: int | None = MAX_EXECUTION_TIME_SECONDS,
):
    """沙箱执行 Python 代码的核心函数。

    整体流程：
        1. ast.parse(code) 将代码字符串解析为 AST 语法树
        2. 初始化 state（变量存储）、print 容器、操作计数器
        3. 包装 final_answer 为异常触发版本
        4. 遍历 AST 顶层节点，逐个调用 evaluate_ast() 模拟执行
        5. 返回 (最终结果, 是否为 final_answer)

    Args:
        code: LLM 生成的 Python 代码字符串
        static_tools: 不可被代码覆盖的工具函数（Agent 的工具 + 内置函数）
            赋值给同名变量会报错，保证工具不被篡改
        custom_tools: 可被代码覆盖的工具函数
            赋值给同名变量会替换掉原来的工具
        state: 变量存储字典，跨步骤持久化（上一步定义的变量下一步还能用）
        authorized_imports: 允许导入的模块白名单
        max_print_outputs_length: print 输出的最大字符数，超出则截断
        timeout_seconds: 最大执行时间（秒），None 表示不限时

    Returns:
        tuple: (result, is_final_answer)
            - result: 代码最后一个表达式的值，或 final_answer() 的参数
            - is_final_answer: 是否调用了 final_answer()
    """
    # ===== 第1步：解析代码为 AST =====
    try:
        expression = ast.parse(code)
    except SyntaxError as e:
        raise InterpreterError(
            f"Code parsing failed on line {e.lineno} due to: {type(e).__name__}: {str(e)}\n"
            f"{e.text}"
            f"{' ' * (e.offset or 0)}^"
        )

    # ===== 第2步：初始化执行环境 =====
    if state is None:
        state = {}
    static_tools = static_tools.copy() if static_tools is not None else {}  # 复制一份，避免修改原始字典
    custom_tools = custom_tools if custom_tools is not None else {}
    state["_print_outputs"] = PrintContainer()  # 每次执行重置 print 容器，收集print输出，沙箱控制输出
    state["_operations_count"] = {"counter": 0}  # 操作计数器（用 dict 包装以便在嵌套函数中修改），闭包中修改外层函数的变量时，整数是不可修改变量，要用可修改的引用类型

    # ===== 第3步：包装 final_answer =====
    # 将 final_answer 工具包装为异常触发版本：
    # 当 LLM 代码调用 final_answer(result) 时，先执行原始工具，再抛出异常终止执行
    if "final_answer" in static_tools:
        previous_final_answer = static_tools["final_answer"]

        def final_answer(*args, **kwargs):
            # 先调用原始 final_answer 工具（可能做格式化等处理），再抛异常
            raise FinalAnswerException(previous_final_answer(*args, **kwargs))

        static_tools["final_answer"] = final_answer

    # ===== 第4步：定义实际执行逻辑 =====
    def _execute_code():
        result = None
        try:
            # 遍历 AST 顶层节点，逐个执行（赋值、函数定义、表达式等）
            for node in expression.body:
                result = evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
            # 正常执行完毕：截断过长的 print 输出
            state["_print_outputs"].value = truncate_content(
                str(state["_print_outputs"]), max_length=max_print_outputs_length
            )
            is_final_answer = False
            return result, is_final_answer
        except FinalAnswerException as e:
            # 捕获 final_answer() 触发的异常 → 标记为最终答案
            state["_print_outputs"].value = truncate_content(
                str(state["_print_outputs"]), max_length=max_print_outputs_length
            )
            is_final_answer = True
            return e.value, is_final_answer
        except Exception as e:
            # 其他异常 → 包装为 InterpreterError，附带出错行的源码
            state["_print_outputs"].value = truncate_content(
                str(state["_print_outputs"]), max_length=max_print_outputs_length
            )
            raise InterpreterError(
                f"Code execution failed at line '{ast.get_source_segment(code, node)}' due to: {type(e).__name__}: {e}"
            )

    # ===== 第5步：应用超时装饰器并执行 =====
    if timeout_seconds is not None:
        _execute_code = timeout(timeout_seconds)(_execute_code)

    return _execute_code()


@dataclass
class CodeOutput:
    """代码执行的完整结果。
    output: 代码的最终返回值（最后一个表达式的值，或 final_answer 的参数）
    logs: 代码中 print() 输出的内容（被重定向收集）
    is_final_answer: 是否调用了 final_answer()（True 则终止 ReAct 循环）
    """
    output: Any
    logs: str
    is_final_answer: bool


class PythonExecutor(ABC):
    """Python 代码执行器的抽象基类。
    LocalPythonExecutor：在当前进程内通过 AST 解释器执行（本文件实现）
    远程执行器（BlaxelExecutor/E2BExecutor/DockerExecutor/ModalExecutor/WasmExecutor）：
        在隔离的远程环境中执行（见 remote_executors.py）
    """
    @abstractmethod
    def send_tools(self, tools: dict[str, Tool]) -> None: ...

    @abstractmethod
    def send_variables(self, variables: dict[str, Any]) -> None: ...

    @abstractmethod
    def __call__(self, code_action: str) -> CodeOutput: ...


class LocalPythonExecutor(PythonExecutor):
    """本地 Python 沙箱执行器 —— CodeAgent 的默认执行器。

    在当前进程内通过 AST 解释器执行 LLM 生成的代码，不启动子进程。
    每次 Agent 的一个 step 调用 __call__() 执行一段代码。

    关键特性：
        - state 跨步骤持久化：step1 定义的变量 step2 还能用
        - static_tools 不可覆盖：LLM 代码不能给工具名赋值（防止篡改）
        - print 输出收集：不打印到终端，收集到 logs 中返回
        - 导入白名单：只允许安全模块

    与远程执行器的区别：
        - LocalPythonExecutor：快，无网络开销，但安全性依赖 AST 解释器
        - 远程执行器（E2B/Docker/Modal 等）：慢，有网络开销，但物理隔离更安全

    Args:
        additional_authorized_imports: 额外允许导入的模块列表
        max_print_outputs_length: print 输出最大字符数
        additional_functions: 额外注入的 Python 函数（会合并到 static_tools）
        timeout_seconds: 单次执行最大时间（秒）
    """

    def __init__(
        self,
        additional_authorized_imports: list[str],
        max_print_outputs_length: int | None = None,
        additional_functions: dict[str, Callable] | None = None,
        timeout_seconds: int | None = MAX_EXECUTION_TIME_SECONDS,
    ):
        self.custom_tools = {}
        self.state = {"__name__": "__main__"}  # 变量存储，跨步骤持久化
        self.max_print_outputs_length = max_print_outputs_length
        if max_print_outputs_length is None:
            self.max_print_outputs_length = DEFAULT_MAX_LEN_OUTPUT
        # 合并默认白名单 + 用户额外白名单，去重
        self.additional_authorized_imports = additional_authorized_imports
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        self._check_authorized_imports_are_installed()
        self.static_tools = None  # 在 send_tools() 中初始化
        self.additional_functions = additional_functions or {}
        self.timeout_seconds = timeout_seconds

    def _check_authorized_imports_are_installed(self):
        """检查白名单中的模块是否已安装。
        在初始化时就检查，避免运行时才发现模块缺失。
        跳过通配符 "*"，只检查顶层模块名（如 "numpy.linalg" 只检查 "numpy"）。
        """
        missing_modules = [
            base_module
            for imp in self.authorized_imports
            if imp != "*" and find_spec(base_module := imp.split(".")[0]) is None
        ]
        if missing_modules:
            raise InterpreterError(
                f"Non-installed authorized modules: {', '.join(missing_modules)}. "
                f"Please install these modules or remove them from the authorized imports list."
            )

    def __call__(self, code_action: str) -> CodeOutput:
        """执行一段 LLM 生成的代码（Agent 每个 step 调用一次）。

        调用 evaluate_python_code() 执行代码，然后把结果包装成 CodeOutput 返回。
        注意 self.state 是持久化的，所以上一步的变量这一步还能用。
        """
        output, is_final_answer = evaluate_python_code(
            code_action,                                        # code: LLM 生成的代码字符串
            static_tools=self.static_tools,                     # 不可覆盖的工具（Agent工具 + 内置函数）
            custom_tools=self.custom_tools,                     # 可覆盖的工具
            state=self.state,                                   # 跨步骤持久化的变量存储
            authorized_imports=self.authorized_imports,          # 导入白名单
            max_print_outputs_length=self.max_print_outputs_length,  # print 输出最大长度
            timeout_seconds=self.timeout_seconds,               # 执行超时秒数
        )
        logs = str(self.state["_print_outputs"])
        return CodeOutput(output=output, logs=logs, is_final_answer=is_final_answer)

    def send_variables(self, variables: dict[str, Any]):
        """注入变量到 state 中（Agent 的 additional_args 通过这个方法传入）。"""
        self.state.update(variables)

    def send_tools(self, tools: dict[str, Tool]):
        """初始化 static_tools：合并 Agent 工具 + 内置安全函数 + 额外函数。
        这些工具在沙箱中不可被 LLM 代码覆盖。
        """
        self.static_tools = {**tools, **BASE_PYTHON_TOOLS.copy(), **self.additional_functions}


__all__ = ["evaluate_python_code", "LocalPythonExecutor"]
