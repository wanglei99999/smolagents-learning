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


def evaluate_attribute(
    expression: ast.Attribute,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    if expression.attr.startswith("__") and expression.attr.endswith("__"):
        raise InterpreterError(f"Forbidden access to dunder attribute: {expression.attr}")
    value = evaluate_ast(expression.value, state, static_tools, custom_tools, authorized_imports)
    return getattr(value, expression.attr)


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
    source_code = ast.unparse(func_def)

    def new_func(*args: Any, **kwargs: Any) -> Any:
        func_state = state.copy()
        arg_names = [arg.arg for arg in func_def.args.args]
        default_values = [
            evaluate_ast(d, state, static_tools, custom_tools, authorized_imports) for d in func_def.args.defaults
        ]

        # Apply default values
        defaults = dict(zip(arg_names[-len(default_values) :], default_values))

        # Set positional arguments
        for name, value in zip(arg_names, args):
            func_state[name] = value

        # Set keyword arguments
        for name, value in kwargs.items():
            func_state[name] = value

        # Handle variable arguments
        if func_def.args.vararg:
            vararg_name = func_def.args.vararg.arg
            func_state[vararg_name] = args

        if func_def.args.kwarg:
            kwarg_name = func_def.args.kwarg.arg
            func_state[kwarg_name] = kwargs

        # Set default values for arguments that were not provided
        for name, value in defaults.items():
            if name not in func_state:
                func_state[name] = value

        # Update function state with self and __class__
        if func_def.args.args and func_def.args.args[0].arg == "self":
            if args:
                func_state["self"] = args[0]
                func_state["__class__"] = args[0].__class__

        result = None
        try:
            for stmt in func_def.body:
                result = evaluate_ast(stmt, func_state, static_tools, custom_tools, authorized_imports)
        except ReturnException as e:
            result = e.value

        if func_def.name == "__init__":
            return None

        return result

    # Store original AST, source code, and name
    new_func.__ast__ = func_def
    new_func.__source__ = source_code
    new_func.__name__ = func_def.name

    return new_func


def evaluate_function_def(
    func_def: ast.FunctionDef,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Callable:
    custom_tools[func_def.name] = create_function(func_def, state, static_tools, custom_tools, authorized_imports)
    return custom_tools[func_def.name]


def evaluate_class_def(
    class_def: ast.ClassDef,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> type:
    class_name = class_def.name
    bases = [evaluate_ast(base, state, static_tools, custom_tools, authorized_imports) for base in class_def.bases]

    # Determine the metaclass to use
    # If any base class has a custom metaclass, use it
    metaclass = type
    for base in bases:
        base_metaclass = type(base)
        if base_metaclass is not type:
            metaclass = base_metaclass
            break

    # Use __prepare__ if the metaclass provides it (e.g., Enum uses _EnumDict)
    if hasattr(metaclass, "__prepare__"):
        class_dict = metaclass.__prepare__(class_name, bases)
    else:
        class_dict = {}

    for stmt in class_def.body:
        if isinstance(stmt, ast.FunctionDef):
            class_dict[stmt.name] = evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
        elif isinstance(stmt, ast.AnnAssign):
            if stmt.value:
                value = evaluate_ast(stmt.value, state, static_tools, custom_tools, authorized_imports)
            target = stmt.target
            # Handle target types for annotation
            if isinstance(target, ast.Name):
                # Simple variable annotation like "x: int"
                annotation = evaluate_ast(stmt.annotation, state, static_tools, custom_tools, authorized_imports)
                class_dict.setdefault("__annotations__", {})[target.id] = annotation
                # Assign value if provided
                if stmt.value:
                    class_dict[target.id] = value
            elif isinstance(target, ast.Attribute):
                # Attribute annotation like "obj.attr: int"
                obj = evaluate_ast(target.value, class_dict, static_tools, custom_tools, authorized_imports)
                # If there's a value assignment, set the attribute
                if stmt.value:
                    setattr(obj, target.attr, value)
            elif isinstance(target, ast.Subscript):
                # Subscript annotation like "dict[key]: int"
                container = evaluate_ast(target.value, class_dict, static_tools, custom_tools, authorized_imports)
                index = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
                # If there's a value assignment, set the item
                if stmt.value:
                    container[index] = value
            else:
                raise InterpreterError(f"Unsupported AnnAssign target in class body: {type(target).__name__}")
        elif isinstance(stmt, ast.Assign):
            value = evaluate_ast(stmt.value, state, static_tools, custom_tools, authorized_imports)
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    class_dict[target.id] = value
                elif isinstance(target, ast.Attribute):
                    obj = evaluate_ast(target.value, class_dict, static_tools, custom_tools, authorized_imports)
                    setattr(obj, target.attr, value)
        elif isinstance(stmt, ast.Pass):
            pass
        elif (
            isinstance(stmt, ast.Expr)
            and stmt == class_def.body[0]
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            # Check if it is a docstring: first statement in class body which is a string literal expression
            class_dict["__doc__"] = stmt.value.value
        else:
            raise InterpreterError(f"Unsupported statement in class body: {stmt.__class__.__name__}")

    new_class = metaclass(class_name, tuple(bases), class_dict)
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
    def get_current_value(target: ast.AST) -> Any:
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

    current_value = get_current_value(expression.target)
    value_to_add = evaluate_ast(expression.value, state, static_tools, custom_tools, authorized_imports)

    if isinstance(expression.op, ast.Add):
        if isinstance(current_value, list):
            if not isinstance(value_to_add, list):
                raise InterpreterError(f"Cannot add non-list value {value_to_add} to a list.")
            current_value += value_to_add
        else:
            current_value += value_to_add
    elif isinstance(expression.op, ast.Sub):
        current_value -= value_to_add
    elif isinstance(expression.op, ast.Mult):
        current_value *= value_to_add
    elif isinstance(expression.op, ast.Div):
        current_value /= value_to_add
    elif isinstance(expression.op, ast.Mod):
        current_value %= value_to_add
    elif isinstance(expression.op, ast.Pow):
        current_value **= value_to_add
    elif isinstance(expression.op, ast.FloorDiv):
        current_value //= value_to_add
    elif isinstance(expression.op, ast.BitAnd):
        current_value &= value_to_add
    elif isinstance(expression.op, ast.BitOr):
        current_value |= value_to_add
    elif isinstance(expression.op, ast.BitXor):
        current_value ^= value_to_add
    elif isinstance(expression.op, ast.LShift):
        current_value <<= value_to_add
    elif isinstance(expression.op, ast.RShift):
        current_value >>= value_to_add
    else:
        raise InterpreterError(f"Operation {type(expression.op).__name__} is not supported.")

    # Update the state: current_value has been updated in-place
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
    # Determine which value should trigger short-circuit based on operation type:
    # - 'and' returns the first falsy value encountered (or the last value if all are truthy)
    # - 'or' returns the first truthy value encountered (or the last value if all are falsy)
    is_short_circuit_value = (lambda x: not x) if isinstance(node.op, ast.And) else (lambda x: bool(x))
    for value in node.values:
        result = evaluate_ast(value, state, static_tools, custom_tools, authorized_imports)
        # Short-circuit: return immediately if the condition is met
        if is_short_circuit_value(result):
            return result
    # If no short-circuit occurred, return the last evaluated value
    return result


def evaluate_binop(
    binop: ast.BinOp,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    # Recursively evaluate the left and right operands
    left_val = evaluate_ast(binop.left, state, static_tools, custom_tools, authorized_imports)
    right_val = evaluate_ast(binop.right, state, static_tools, custom_tools, authorized_imports)

    # Determine the operation based on the type of the operator in the BinOp
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
        raise NotImplementedError(f"Binary operation {type(binop.op).__name__} is not implemented.")


def evaluate_assign(
    assign: ast.Assign,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    result = evaluate_ast(assign.value, state, static_tools, custom_tools, authorized_imports)
    if len(assign.targets) == 1:
        target = assign.targets[0]
        set_value(target, result, state, static_tools, custom_tools, authorized_imports)
    else:
        expanded_values = []
        for tgt in assign.targets:
            if isinstance(tgt, ast.Starred):
                expanded_values.extend(result)
            else:
                expanded_values.append(result)

        for tgt, val in zip(assign.targets, expanded_values):
            set_value(tgt, val, state, static_tools, custom_tools, authorized_imports)
    return result


def set_value(
    target: ast.AST,
    value: Any,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> None:
    if isinstance(target, ast.Name):
        if target.id in static_tools:
            raise InterpreterError(f"Cannot assign to name '{target.id}': doing this would erase the existing tool!")
        state[target.id] = value
    elif isinstance(target, ast.Tuple):
        if not isinstance(value, tuple):
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                value = tuple(value)
            else:
                raise InterpreterError("Cannot unpack non-tuple value")
        if len(target.elts) != len(value):
            raise InterpreterError("Cannot unpack tuple of wrong size")
        for i, elem in enumerate(target.elts):
            set_value(elem, value[i], state, static_tools, custom_tools, authorized_imports)
    elif isinstance(target, ast.Subscript):
        obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
        key = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
        obj[key] = value
    elif isinstance(target, ast.Attribute):
        obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
        setattr(obj, target.attr, value)


def evaluate_call(
    call: ast.Call,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    if not isinstance(call.func, (ast.Call, ast.Lambda, ast.Attribute, ast.Name, ast.Subscript)):
        raise InterpreterError(f"This is not a correct function: {call.func}).")

    func, func_name = None, None

    if isinstance(call.func, ast.Call):
        func = evaluate_ast(call.func, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(call.func, ast.Lambda):
        func = evaluate_ast(call.func, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(call.func, ast.Attribute):
        obj = evaluate_ast(call.func.value, state, static_tools, custom_tools, authorized_imports)
        func_name = call.func.attr
        if not hasattr(obj, func_name):
            raise InterpreterError(f"Object {obj} has no attribute {func_name}")
        func = getattr(obj, func_name)
    elif isinstance(call.func, ast.Name):
        func_name = call.func.id
        if func_name in state:
            func = state[func_name]
        elif func_name in static_tools:
            func = static_tools[func_name]
        elif func_name in custom_tools:
            func = custom_tools[func_name]
        elif func_name in ERRORS:
            func = ERRORS[func_name]
        else:
            raise InterpreterError(
                f"Forbidden function evaluation: '{call.func.id}' is not among the explicitly allowed tools or defined/imported in the preceding code"
            )
    elif isinstance(call.func, ast.Subscript):
        func = evaluate_ast(call.func, state, static_tools, custom_tools, authorized_imports)
        if not callable(func):
            raise InterpreterError(f"This is not a correct function: {call.func}).")
        func_name = None

    args = []
    for arg in call.args:
        if isinstance(arg, ast.Starred):
            args.extend(evaluate_ast(arg.value, state, static_tools, custom_tools, authorized_imports))
        else:
            args.append(evaluate_ast(arg, state, static_tools, custom_tools, authorized_imports))

    kwargs = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            # **kwargs unpacking
            starred_dict = evaluate_ast(keyword.value, state, static_tools, custom_tools, authorized_imports)
            if not isinstance(starred_dict, dict):
                raise InterpreterError(f"Cannot unpack non-dict value in **kwargs: {type(starred_dict).__name__}")
            kwargs.update(starred_dict)
        else:
            # Normal keyword argument
            kwargs[keyword.arg] = evaluate_ast(keyword.value, state, static_tools, custom_tools, authorized_imports)

    if func_name == "super":
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
        state["_print_outputs"] += " ".join(map(str, args)) + "\n"
        return None
    else:  # Assume it's a callable object
        if (inspect.getmodule(func) == builtins) and inspect.isbuiltin(func) and (func not in static_tools.values()):
            raise InterpreterError(
                f"Invoking a builtin function that has not been explicitly added as a tool is not allowed ({func_name})."
            )
        if (
            hasattr(func, "__name__")
            and func.__name__.startswith("__")
            and func.__name__.endswith("__")
            and (func.__name__ not in static_tools)
            and (func.__name__ not in ALLOWED_DUNDER_METHODS)
        ):
            raise InterpreterError(f"Forbidden call to dunder function: {func.__name__}")
        return func(*args, **kwargs)


def evaluate_subscript(
    subscript: ast.Subscript,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    index = evaluate_ast(subscript.slice, state, static_tools, custom_tools, authorized_imports)
    value = evaluate_ast(subscript.value, state, static_tools, custom_tools, authorized_imports)
    try:
        return value[index]
    except (KeyError, IndexError, TypeError) as e:
        error_message = f"Could not index {value} with '{index}': {type(e).__name__}: {e}"
        if isinstance(index, str) and isinstance(value, Mapping):
            close_matches = difflib.get_close_matches(index, list(value.keys()))
            if len(close_matches) > 0:
                error_message += f". Maybe you meant one of these indexes instead: {str(close_matches)}"
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
    result = True
    left = evaluate_ast(condition.left, state, static_tools, custom_tools, authorized_imports)
    for i, (op, comparator) in enumerate(zip(condition.ops, condition.comparators)):
        op = type(op)
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

        if current_result is False:
            return False
        result = current_result if i == 0 else (result and current_result)
        left = right
    return result


def evaluate_if(
    if_statement: ast.If,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    result = None
    test_result = evaluate_ast(if_statement.test, state, static_tools, custom_tools, authorized_imports)
    if test_result:
        for line in if_statement.body:
            line_result = evaluate_ast(line, state, static_tools, custom_tools, authorized_imports)
            if line_result is not None:
                result = line_result
    else:
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
    result = None
    iterator = evaluate_ast(for_loop.iter, state, static_tools, custom_tools, authorized_imports)
    for counter in iterator:
        set_value(
            for_loop.target,
            counter,
            state,
            static_tools,
            custom_tools,
            authorized_imports,
        )
        for node in for_loop.body:
            try:
                line_result = evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
                if line_result is not None:
                    result = line_result
            except BreakException:
                return result
            except ContinueException:
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
    # Base case: no more comprehensions
    if not comprehensions:
        yield evaluate_element(state)
        return
    # Evaluate first comprehension
    comprehension = comprehensions[0]
    iter_value = evaluate_ast(comprehension.iter, state, static_tools, custom_tools, authorized_imports)
    for value in iter_value:
        new_state = state.copy()
        set_value(comprehension.target, value, new_state, static_tools, custom_tools, authorized_imports)
        # Check all filter conditions
        if all(
            evaluate_ast(if_clause, new_state, static_tools, custom_tools, authorized_imports)
            for if_clause in comprehension.ifs
        ):
            # Recurse with remaining comprehensions
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
    """Creates a safe copy of a module or returns the original if it's a function"""
    # If it's a function or non-module object, return it directly
    if not isinstance(raw_module, ModuleType):
        return raw_module

    # Handle circular references: Initialize visited set for the first call
    if visited is None:
        visited = set()

    module_id = id(raw_module)
    if module_id in visited:
        return raw_module  # Return original for circular refs

    visited.add(module_id)

    # Create new module for actual modules
    safe_module = ModuleType(raw_module.__name__)

    # Copy all attributes by reference, recursively checking modules
    for attr_name in dir(raw_module):
        try:
            attr_value = getattr(raw_module, attr_name)
        except (ImportError, AttributeError) as e:
            # lazy / dynamic loading module -> INFO log and skip
            logger.info(
                f"Skipping import error while copying {raw_module.__name__}.{attr_name}: {type(e).__name__} - {e}"
            )
            continue
        # Recursively process nested modules, passing visited set
        if isinstance(attr_value, ModuleType):
            attr_value = get_safe_module(attr_value, authorized_imports, visited=visited)

        setattr(safe_module, attr_name, attr_value)

    return safe_module


def evaluate_import(expression, state, authorized_imports):
    if isinstance(expression, ast.Import):
        for alias in expression.names:
            if check_import_authorized(alias.name, authorized_imports):
                raw_module = import_module(alias.name)
                state[alias.asname or alias.name] = get_safe_module(raw_module, authorized_imports)
            else:
                raise InterpreterError(
                    f"Import of {alias.name} is not allowed. Authorized imports are: {str(authorized_imports)}"
                )
        return None
    elif isinstance(expression, ast.ImportFrom):
        if check_import_authorized(expression.module, authorized_imports):
            raw_module = __import__(expression.module, fromlist=[alias.name for alias in expression.names])
            module = get_safe_module(raw_module, authorized_imports)
            if expression.names[0].name == "*":  # Handle "from module import *"
                if hasattr(module, "__all__"):  # If module has __all__, import only those names
                    for name in module.__all__:
                        state[name] = getattr(module, name)
                else:  # If no __all__, import all public names (those not starting with '_')
                    for name in dir(module):
                        if not name.startswith("_"):
                            state[name] = getattr(module, name)
            else:  # regular from imports
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
    """
    Evaluate an abstract syntax tree using the content of the variables stored in a state and only evaluating a given
    set of functions.

    This function will recurse through the nodes of the tree provided.

    Args:
        expression (`ast.AST`):
            The code to evaluate, as an abstract syntax tree.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to values. The `state` is updated if need be when the evaluation
            encounters assignments.
        static_tools (`Dict[str, Callable]`):
            Functions that may be called during the evaluation. Trying to change one of these static_tools will raise an error.
        custom_tools (`Dict[str, Callable]`):
            Functions that may be called during the evaluation. These custom_tools can be overwritten.
        authorized_imports (`List[str]`):
            The list of modules that can be imported by the code. By default, only a few safe modules are allowed.
            If it contains "*", it will authorize any import. Use this at your own risk!
    """
    if state.setdefault("_operations_count", {"counter": 0})["counter"] >= MAX_OPERATIONS:
        raise InterpreterError(
            f"Reached the max number of operations of {MAX_OPERATIONS}. Maybe there is an infinite loop somewhere in the code, or you're just asking too many calculations."
        )
    state["_operations_count"]["counter"] += 1
    common_params = (state, static_tools, custom_tools, authorized_imports)
    if isinstance(expression, ast.Assign):
        # Assignment -> we evaluate the assignment which should update the state
        # We return the variable assigned as it may be used to determine the final result.
        return evaluate_assign(expression, *common_params)
    elif isinstance(expression, ast.AnnAssign):
        return evaluate_annassign(expression, *common_params)
    elif isinstance(expression, ast.AugAssign):
        return evaluate_augassign(expression, *common_params)
    elif isinstance(expression, ast.Call):
        # Function call -> we return the value of the function call
        return evaluate_call(expression, *common_params)
    elif isinstance(expression, ast.Constant):
        # Constant -> just return the value
        return expression.value
    elif isinstance(expression, ast.Tuple):
        return tuple((evaluate_ast(elt, *common_params) for elt in expression.elts))
    elif isinstance(expression, ast.GeneratorExp):
        return evaluate_generatorexp(expression, *common_params)
    elif isinstance(expression, ast.ListComp):
        return evaluate_listcomp(expression, *common_params)
    elif isinstance(expression, ast.DictComp):
        return evaluate_dictcomp(expression, *common_params)
    elif isinstance(expression, ast.SetComp):
        return evaluate_setcomp(expression, *common_params)
    elif isinstance(expression, ast.UnaryOp):
        return evaluate_unaryop(expression, *common_params)
    elif isinstance(expression, ast.Starred):
        return evaluate_ast(expression.value, *common_params)
    elif isinstance(expression, ast.BoolOp):
        # Boolean operation -> evaluate the operation
        return evaluate_boolop(expression, *common_params)
    elif isinstance(expression, ast.Break):
        raise BreakException()
    elif isinstance(expression, ast.Continue):
        raise ContinueException()
    elif isinstance(expression, ast.BinOp):
        # Binary operation -> execute operation
        return evaluate_binop(expression, *common_params)
    elif isinstance(expression, ast.Compare):
        # Comparison -> evaluate the comparison
        return evaluate_condition(expression, *common_params)
    elif isinstance(expression, ast.Lambda):
        return evaluate_lambda(expression, *common_params)
    elif isinstance(expression, ast.FunctionDef):
        return evaluate_function_def(expression, *common_params)
    elif isinstance(expression, ast.Dict):
        # Dict -> evaluate all keys and values
        keys = (evaluate_ast(k, *common_params) for k in expression.keys)
        values = (evaluate_ast(v, *common_params) for v in expression.values)
        return dict(zip(keys, values))
    elif isinstance(expression, ast.Expr):
        # Expression -> evaluate the content
        return evaluate_ast(expression.value, *common_params)
    elif isinstance(expression, ast.For):
        # For loop -> execute the loop
        return evaluate_for(expression, *common_params)
    elif isinstance(expression, ast.FormattedValue):
        # Formatted value (part of f-string) -> evaluate the content and format it
        value = evaluate_ast(expression.value, *common_params)
        # Early return if no format spec
        if not expression.format_spec:
            return value
        # Apply format specification
        format_spec = evaluate_ast(expression.format_spec, *common_params)
        return format(value, format_spec)
    elif isinstance(expression, ast.If):
        # If -> execute the right branch
        return evaluate_if(expression, *common_params)
    elif hasattr(ast, "Index") and isinstance(expression, ast.Index):
        return evaluate_ast(expression.value, *common_params)
    elif isinstance(expression, ast.JoinedStr):
        return "".join([str(evaluate_ast(v, *common_params)) for v in expression.values])
    elif isinstance(expression, ast.List):
        # List -> evaluate all elements
        return [evaluate_ast(elt, *common_params) for elt in expression.elts]
    elif isinstance(expression, ast.Name):
        # Name -> pick up the value in the state
        return evaluate_name(expression, *common_params)
    elif isinstance(expression, ast.Subscript):
        # Subscript -> return the value of the indexing
        return evaluate_subscript(expression, *common_params)
    elif isinstance(expression, ast.IfExp):
        test_val = evaluate_ast(expression.test, *common_params)
        if test_val:
            return evaluate_ast(expression.body, *common_params)
        else:
            return evaluate_ast(expression.orelse, *common_params)
    elif isinstance(expression, ast.Attribute):
        return evaluate_attribute(expression, *common_params)
    elif isinstance(expression, ast.Slice):
        return slice(
            evaluate_ast(expression.lower, *common_params) if expression.lower is not None else None,
            evaluate_ast(expression.upper, *common_params) if expression.upper is not None else None,
            evaluate_ast(expression.step, *common_params) if expression.step is not None else None,
        )
    elif isinstance(expression, ast.While):
        return evaluate_while(expression, *common_params)
    elif isinstance(expression, (ast.Import, ast.ImportFrom)):
        return evaluate_import(expression, state, authorized_imports)
    elif isinstance(expression, ast.ClassDef):
        return evaluate_class_def(expression, *common_params)
    elif isinstance(expression, ast.Try):
        return evaluate_try(expression, *common_params)
    elif isinstance(expression, ast.Raise):
        return evaluate_raise(expression, *common_params)
    elif isinstance(expression, ast.Assert):
        return evaluate_assert(expression, *common_params)
    elif isinstance(expression, ast.With):
        return evaluate_with(expression, *common_params)
    elif isinstance(expression, ast.Set):
        return set((evaluate_ast(elt, *common_params) for elt in expression.elts))
    elif isinstance(expression, ast.Return):
        raise ReturnException(evaluate_ast(expression.value, *common_params) if expression.value else None)
    elif isinstance(expression, ast.Pass):
        return None
    elif isinstance(expression, ast.Delete):
        return evaluate_delete(expression, *common_params)
    else:
        # For now we refuse anything else. Let's add things as we need them.
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
    state["_print_outputs"] = PrintContainer()  # 每次执行重置 print 容器
    state["_operations_count"] = {"counter": 0}  # 操作计数器（用 dict 包装以便在嵌套函数中修改）

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
