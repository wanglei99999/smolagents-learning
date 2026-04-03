import ast
import builtins
from itertools import zip_longest

from .utils import BASE_BUILTIN_MODULES, get_source, is_valid_name


_BUILTIN_NAMES = set(vars(builtins))


class MethodChecker(ast.NodeVisitor):
    """
    Checks that a method
    - only uses defined names
    - contains no local imports (e.g. numpy is ok but local_script is not)
    """

    def __init__(self, class_attributes: set[str], check_imports: bool = True):
        # MethodChecker 的角色可以理解成“单个方法体的 AST 静态检查器”。
        # 它不运行方法，而是遍历方法的 AST，回答两个问题：
        # 1. 这个方法里用到的名字，来源是否都解释得清楚？
        # 2. 这个方法是否足够自包含，适合被抽取源码、重建、远程执行？
        self.undefined_names = set()
        self.imports = {} #import XXX 得到的本地名字
        self.from_imports = {} # 也是本地名字
        self.assigned_names = set() #方法体里赋值出来的名字
        self.arg_names = set() #参数名
        self.class_attributes = class_attributes #类属性名
        self.errors = [] #问题列表
        self.check_imports = check_imports
        self.typing_names = {"Any"}
        self.defined_classes = set()

    def visit_arguments(self, node):
        """Collect function arguments"""
        # 参数名天然属于“已定义名字”，后面 visit_Name / visit_Call 会用到。
        self.arg_names = {arg.arg for arg in node.args}
        if node.kwarg:
            self.arg_names.add(node.kwarg.arg)
        if node.vararg:
            self.arg_names.add(node.vararg.arg)

    def visit_Import(self, node):
        # 记录 import 得到的本地名字。
        # 例如：import numpy as np -> imports["np"] = "numpy"
        for name in node.names:
            actual_name = name.asname or name.name
            self.imports[actual_name] = name.name

    def visit_ImportFrom(self, node):
        # 记录 from ... import ... 得到的本地名字。
        # 例如：from math import sqrt as s -> from_imports["s"] = ("math", "sqrt")
        module = node.module or ""
        for name in node.names:
            actual_name = name.asname or name.name
            self.from_imports[actual_name] = (module, name.name)

    def visit_Assign(self, node):
        # 追踪普通赋值产生的局部名字。
        # 例如：x = ...
        #      a, b = ...
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assigned_names.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.assigned_names.add(elt.id)
        self.visit(node.value)

    def visit_With(self, node):
        """Track aliases in 'with' statements (the 'y' in 'with X as y')"""
        for item in node.items:
            if item.optional_vars:  # This is the 'y' in 'with X as y'
                if isinstance(item.optional_vars, ast.Name):
                    self.assigned_names.add(item.optional_vars.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        """Track exception aliases (the 'e' in 'except Exception as e')"""
        if node.name:  # This is the 'e' in 'except Exception as e'
            self.assigned_names.add(node.name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        """Track annotated assignments."""
        # 追踪注解赋值，例如：x: int = 1
        if isinstance(node.target, ast.Name):
            self.assigned_names.add(node.target.id)
        if node.value:
            self.visit(node.value)

    def visit_For(self, node):
        # 追踪 for 循环目标变量，例如：
        #   for x in ...
        #   for a, b in ...
        target = node.target
        if isinstance(target, ast.Name):
            self.assigned_names.add(target.id)
        elif isinstance(target, ast.Tuple):
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    self.assigned_names.add(elt.id)
        self.generic_visit(node)

    def _handle_comprehension_generators(self, generators):
        """Helper method to handle generators in all types of comprehensions"""
        # 推导式也会引入局部变量，例如：
        #   [x for x in items]
        #   [(a, b) for a, b in pairs]
        for generator in generators:
            if isinstance(generator.target, ast.Name):
                self.assigned_names.add(generator.target.id)
            elif isinstance(generator.target, ast.Tuple):
                for elt in generator.target.elts:
                    if isinstance(elt, ast.Name):
                        self.assigned_names.add(elt.id)

    def visit_ListComp(self, node):
        """Track variables in list comprehensions"""
        self._handle_comprehension_generators(node.generators)
        self.generic_visit(node)

    def visit_DictComp(self, node):
        """Track variables in dictionary comprehensions"""
        self._handle_comprehension_generators(node.generators)
        self.generic_visit(node)

    def visit_SetComp(self, node):
        """Track variables in set comprehensions"""
        self._handle_comprehension_generators(node.generators)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # 对 self.xxx 这种访问不继续向里追踪，是因为类属性/实例属性由外层规则处理；
        # 其他属性访问则继续 generic_visit，防止漏掉内部名字引用。
        if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
            self.generic_visit(node)

    def visit_ClassDef(self, node):
        """Track class definitions"""
        # 方法体内部如果定义了类，这个类名也应视为“已定义名字”。
        self.defined_classes.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node):
        # 这里是最核心的“名字是否已定义”检查。
        # 只在 Load 场景下检查：也就是读取某个名字时，才关心它是否有来源。
        if isinstance(node.ctx, ast.Load):
            if not (
                node.id in _BUILTIN_NAMES
                or node.id in BASE_BUILTIN_MODULES
                or node.id in self.arg_names
                or node.id == "self"
                or node.id in self.class_attributes
                or node.id in self.imports
                or node.id in self.from_imports
                or node.id in self.assigned_names
                or node.id in self.typing_names
                or node.id in self.defined_classes
            ):
                self.errors.append(f"Name '{node.id}' is undefined.")

    def visit_Call(self, node):
        # 对调用表达式额外检查“被调用的名字”是否已定义。
        # 例如 foo(...) 里的 foo，如果来源不明，也要报错。
        if isinstance(node.func, ast.Name):
            if not (
                node.func.id in _BUILTIN_NAMES
                or node.func.id in BASE_BUILTIN_MODULES
                or node.func.id in self.arg_names
                or node.func.id == "self"
                or node.func.id in self.class_attributes
                or node.func.id in self.imports
                or node.func.id in self.from_imports
                or node.func.id in self.assigned_names
                or node.func.id in self.defined_classes
            ):
                self.errors.append(f"Name '{node.func.id}' is undefined.")
        self.generic_visit(node)


def validate_tool_attributes(cls, check_imports: bool = True) -> None:
    """
    Validates that a Tool class follows the proper patterns:
    0. Any argument of __init__ should have a default.
    Args chosen at init are not traceable, so we cannot rebuild the source code for them, thus any important arg should be defined as a class attribute.
    1. About the class:
        - Class attributes should only be strings or dicts
        - Class attributes cannot be complex attributes
    2. About all class methods:
        - Imports must be from packages, not local files
        - All methods must be self-contained

    Raises all errors encountered, if no error returns None.
    """
    # 这是工具校验的总入口。
    # 可以把它理解成：对一个 Tool 类做两层 AST 静态检查。
    # 1. 类级别：类属性、__init__ 参数、name 合法性
    # 2. 方法级别：方法体是否自包含、是否引用了未定义名字

    class ClassLevelChecker(ast.NodeVisitor):
        def __init__(self):
            # 这一层专门看“类定义本身”是否适合被工具系统重建和传输。
            self.imported_names = set()
            self.complex_attributes = set()
            self.class_attributes = set()
            self.non_defaults = set()
            self.non_literal_defaults = set()
            self.in_method = False
            self.invalid_attributes = []

        def visit_FunctionDef(self, node):
            # 只要碰到 __init__，就额外检查参数默认值规则。
            if node.name == "__init__":
                self._check_init_function_parameters(node)
            old_context = self.in_method
            self.in_method = True
            self.generic_visit(node)
            self.in_method = old_context

        def visit_Assign(self, node):
            if self.in_method:
                return
            # Track class attributes
            # 这里只看“类体顶层赋值”，因为那才是真正的 class attributes。
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.class_attributes.add(target.id)

            # Check if the assignment is more complex than simple literals
            # 类属性如果太复杂（例如运行时表达式、函数调用结果），
            # 对工具源码提取和重建都不友好，因此会被标记为 complex_attributes。
            if not all(isinstance(val, (ast.Constant, ast.Dict, ast.List, ast.Set)) for val in ast.walk(node.value)):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.complex_attributes.add(target.id)

            # Check specific class attributes
            # Tool 的 name 字段要求最严格：必须是常量字符串，而且要是合法名字。
            if getattr(node.targets[0], "id", "") == "name":
                if not isinstance(node.value, ast.Constant):
                    self.invalid_attributes.append(f"Class attribute 'name' must be a constant, found '{node.value}'")
                elif not isinstance(node.value.value, str):
                    self.invalid_attributes.append(
                        f"Class attribute 'name' must be a string, found '{node.value.value}'"
                    )
                elif not is_valid_name(node.value.value):
                    self.invalid_attributes.append(
                        f"Class attribute 'name' must be a valid Python identifier and not a reserved keyword, found '{node.value.value}'"
                    )

        def _check_init_function_parameters(self, node):
            # Check defaults in parameters
            # __init__ 的参数必须都有默认值，而且默认值最好是字面量。
            # 原因是 Tool 类常常需要被“只靠源码重建”，
            # 如果初始化依赖外部传入的必填参数或复杂默认值，会让这件事变得不稳定。
            for arg, default in reversed(list(zip_longest(reversed(node.args.args), reversed(node.args.defaults)))):
                if default is None:
                    if arg.arg != "self":
                        self.non_defaults.add(arg.arg)
                elif not isinstance(default, (ast.Constant, ast.Dict, ast.List, ast.Set)):
                    self.non_literal_defaults.add(arg.arg)

    class_level_checker = ClassLevelChecker()
    source = get_source(cls)
    tree = ast.parse(source)
    class_node = tree.body[0]
    if not isinstance(class_node, ast.ClassDef):
        raise ValueError("Source code must define a class")
    class_level_checker.visit(class_node)

    errors = []
    # Check invalid class attributes
    if class_level_checker.invalid_attributes:
        errors += class_level_checker.invalid_attributes
    if class_level_checker.complex_attributes:
        errors.append(
            f"Complex attributes should be defined in __init__, not as class attributes: "
            f"{', '.join(class_level_checker.complex_attributes)}"
        )
    if class_level_checker.non_defaults:
        errors.append(
            f"Parameters in __init__ must have default values, found required parameters: "
            f"{', '.join(class_level_checker.non_defaults)}"
        )
    if class_level_checker.non_literal_defaults:
        errors.append(
            f"Parameters in __init__ must have literal default values, found non-literal defaults: "
            f"{', '.join(class_level_checker.non_literal_defaults)}"
        )

    # Run checks on all methods
    # 类级别通过后，再逐个方法做 MethodChecker 静态检查。
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef):
            method_checker = MethodChecker(class_level_checker.class_attributes, check_imports=check_imports)
            method_checker.visit(node)
            errors += [f"- {node.name}: {error}" for error in method_checker.errors]

    if errors:
        # 统一汇总后一次性抛出，方便调用方直接看到完整问题列表。
        raise ValueError(f"Tool validation failed for {cls.__name__}:\n" + "\n".join(errors))
    return
