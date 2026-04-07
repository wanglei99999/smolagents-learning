# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

"""
Agent 数据类型系统

这个模块定义了 Agent 系统中使用的特殊数据类型包装器。
这些类型解决了多模态 AI 系统中的核心问题：如何统一处理文本、图片、音频等不同类型的数据。

核心设计理念：
1. **行为一致性**：包装后的对象行为与原始类型完全一致
2. **字符串化**：所有对象都能转换为字符串（通常是文件路径）
3. **Jupyter 友好**：在笔记本环境中能正确显示
4. **类型安全**：提供类型检查和转换功能

使用场景：
- Agent 工具的输入输出标准化
- 多模态数据在不同组件间的传递
- Jupyter/Colab 环境中的数据展示
- 文件系统和内存对象的统一处理
"""

import logging
import os
import pathlib
import tempfile
import uuid
from io import BytesIO
from typing import Any

import PIL.Image
import requests

from .utils import _is_package_available


logger = logging.getLogger(__name__)


class AgentType:
    """
    Agent 数据类型的抽象基类
    
    这是所有 Agent 数据类型的基础类，定义了统一的接口规范。
    所有继承此类的对象都具备三个核心能力：
    
    1. **原生行为**：表现得像它们所代表的原始类型
       例如：AgentText 表现得像字符串，AgentImage 表现得像 PIL.Image
    
    2. **字符串化**：通过 str(object) 可以获得对象的字符串表示
       通常是文件路径，便于在不同系统间传递和存储
    
    3. **Jupyter 显示**：在 Jupyter/Colab 等笔记本环境中正确显示
       例如：图片会直接显示，音频会有播放控件
    
    设计模式：
    - 使用组合模式包装原始数据类型
    - 提供统一的 to_raw() 和 to_string() 接口
    - 支持延迟加载和多种数据源
    
    Args:
        value: 要包装的原始值，可以是各种类型的数据
    """

    def __init__(self, value):
        """
        初始化 Agent 类型包装器
        
        Args:
            value: 要包装的原始数据
        """
        self._value = value

    def __str__(self):
        """
        字符串表示方法
        
        当调用 str(agent_object) 时会调用此方法，
        内部委托给 to_string() 方法实现具体逻辑。
        
        Returns:
            str: 对象的字符串表示
        """
        return self.to_string()

    def to_raw(self):
        """
        获取原始数据对象
        
        返回被包装的原始数据，例如：
        - AgentText → str
        - AgentImage → PIL.Image.Image  
        - AgentAudio → torch.Tensor
        
        注意：基类实现会记录错误日志，子类应该重写此方法
        
        Returns:
            Any: 原始数据对象
        """
        logger.error(
            "This is a raw AgentType of unknown type. Display in notebooks and string conversion will be unreliable"
        )
        return self._value

    def to_string(self) -> str:
        """
        获取字符串表示
        
        返回对象的字符串形式，通常是：
        - 文件路径（对于图片、音频）
        - 文本内容（对于文本）
        - 序列化后的表示（对于复杂对象）
        
        注意：基类实现会记录错误日志，子类应该重写此方法
        
        Returns:
            str: 对象的字符串表示
        """
        logger.error(
            "This is a raw AgentType of unknown type. Display in notebooks and string conversion will be unreliable"
        )
        return str(self._value)


class AgentText(AgentType, str):
    """
    Agent 文本类型
    
    这是对字符串类型的包装，继承了 str 的所有行为。
    主要用于标识这是 Agent 系统产生的文本输出，
    与普通字符串在功能上完全相同，但提供了统一的类型标识。
    
    使用场景：
    - Agent 工具返回的文本结果
    - LLM 生成的文本内容
    - 需要类型标识的字符串数据
    
    Example:
        ```python
        text = AgentText("这是 Agent 生成的文本")
        print(text)  # 输出：这是 Agent 生成的文本
        print(len(text))  # 输出：11（字符串长度）
        print(text.upper())  # 输出：这是 AGENT 生成的文本
        ```
    """

    def to_raw(self):
        """
        获取原始字符串
        
        Returns:
            str: 包装的字符串值
        """
        return self._value

    def to_string(self):
        """
        获取字符串表示
        
        对于文本类型，字符串表示就是其内容本身
        
        Returns:
            str: 文本内容
        """
        return str(self._value)


class AgentImage(AgentType, PIL.Image.Image):
    """
    Agent 图像类型
    
    这是对 PIL.Image.Image 的智能包装，支持多种图像数据源和格式。
    继承了 PIL.Image 的所有方法，同时提供了灵活的数据加载和转换能力。
    
    支持的数据源：
    - PIL.Image.Image 对象：直接包装
    - 文件路径（str/pathlib.Path）：延迟加载
    - 字节数据（bytes）：从内存加载
    - Tensor 数据（torch.Tensor/numpy.ndarray）：从数组转换
    - 另一个 AgentImage：复制构造
    
    核心特性：
    - **延迟加载**：文件路径只在需要时才加载图像
    - **多格式支持**：自动处理不同的输入格式
    - **Jupyter 显示**：在笔记本中直接显示图像
    - **临时文件管理**：自动生成临时文件用于序列化
    
    Example:
        ```python
        # 从文件创建
        img1 = AgentImage("path/to/image.jpg")
        
        # 从 PIL 对象创建
        pil_img = PIL.Image.open("image.jpg")
        img2 = AgentImage(pil_img)
        
        # 从字节数据创建
        with open("image.jpg", "rb") as f:
            img3 = AgentImage(f.read())
        
        # 使用 PIL 方法
        resized = img1.resize((224, 224))
        img1.save("output.png")
        ```
    """

    def __init__(self, value):
        """
        初始化 Agent 图像对象
        
        Args:
            value: 图像数据，支持多种格式：
                - AgentImage: 复制另一个 AgentImage
                - PIL.Image.Image: PIL 图像对象
                - bytes: 图像的字节数据
                - str/pathlib.Path: 图像文件路径
                - torch.Tensor: PyTorch 张量
                - numpy.ndarray: NumPy 数组
        
        Raises:
            TypeError: 当输入类型不受支持时抛出
        """
        # 调用父类初始化
        AgentType.__init__(self, value)
        PIL.Image.Image.__init__(self)

        # 初始化内部状态：支持三种数据存储方式
        self._path = None      # 文件路径（延迟加载）
        self._raw = None       # PIL.Image 对象（已加载）
        self._tensor = None    # Tensor 数据（需要转换）

        # 根据输入类型进行相应处理
        if isinstance(value, AgentImage):
            # 复制构造：继承另一个 AgentImage 的所有状态
            self._raw, self._path, self._tensor = value._raw, value._path, value._tensor
        elif isinstance(value, PIL.Image.Image):
            # PIL 图像：直接存储
            self._raw = value
        elif isinstance(value, bytes):
            # 字节数据：立即从内存加载
            self._raw = PIL.Image.open(BytesIO(value))
        elif isinstance(value, (str, pathlib.Path)):
            # 文件路径：延迟加载（只在需要时才读取文件）
            self._path = value
        else:
            # 尝试处理 Tensor 类型（需要可选依赖）
            try:
                import torch
                if isinstance(value, torch.Tensor):
                    self._tensor = value
                
                import numpy as np
                if isinstance(value, np.ndarray):
                    # NumPy 数组转换为 PyTorch Tensor
                    self._tensor = torch.from_numpy(value)
            except ModuleNotFoundError:
                # 如果没有安装 torch/numpy，跳过 Tensor 处理
                pass

        # 验证：至少要有一种有效的数据源
        if self._path is None and self._raw is None and self._tensor is None:
            raise TypeError(f"Unsupported type for {self.__class__.__name__}: {type(value)}")

    def _ipython_display_(self, include=None, exclude=None):
        """
        Jupyter/IPython 显示支持
        
        这是 IPython 的特殊方法，当在 Jupyter 笔记本中显示对象时会自动调用。
        实现了图像的直接显示功能，无需额外的显示命令。
        
        Args:
            include: IPython 显示包含的格式（未使用）
            exclude: IPython 显示排除的格式（未使用）
        """
        from IPython.display import Image, display
        # 使用图像的文件路径进行显示
        display(Image(self.to_string()))

    def to_raw(self):
        """
        获取 PIL.Image.Image 对象
        
        这个方法实现了智能的数据加载策略：
        1. 如果已有 PIL 对象，直接返回
        2. 如果是文件路径，延迟加载
        3. 如果是 Tensor，转换为 PIL 对象
        
        Returns:
            PIL.Image.Image: 可以直接使用的 PIL 图像对象
        """
        # 策略1：已有 PIL 对象，直接返回
        if self._raw is not None:
            return self._raw

        # 策略2：从文件路径加载
        if self._path is not None:
            self._raw = PIL.Image.open(self._path)
            return self._raw

        # 策略3：从 Tensor 转换
        if self._tensor is not None:
            import numpy as np
            # 将 Tensor 转换为 NumPy 数组，然后转换为 PIL 图像
            array = self._tensor.cpu().detach().numpy()
            return PIL.Image.fromarray((255 - array * 255).astype(np.uint8))

    def to_string(self):
        """
        获取图像的文件路径表示
        
        这个方法确保图像总是有一个文件路径表示，用于：
        - 在不同系统间传递图像引用
        - 序列化和反序列化
        - 与其他工具的集成
        
        实现策略：
        1. 如果已有路径，直接返回
        2. 如果是 PIL 对象，保存为临时文件
        3. 如果是 Tensor，转换后保存为临时文件
        
        Returns:
            str: 图像文件的路径
        """
        # 策略1：已有文件路径
        if self._path is not None:
            return self._path

        # 策略2：从 PIL 对象保存
        if self._raw is not None:
            # 创建临时目录和唯一文件名
            directory = tempfile.mkdtemp()
            self._path = os.path.join(directory, str(uuid.uuid4()) + ".png")
            # 保存为 PNG 格式
            self._raw.save(self._path, format="png")
            return self._path

        # 策略3：从 Tensor 转换并保存
        if self._tensor is not None:
            import numpy as np
            # Tensor → NumPy → PIL → 文件
            array = self._tensor.cpu().detach().numpy()
            img = PIL.Image.fromarray((255 - array * 255).astype(np.uint8))
            
            directory = tempfile.mkdtemp()
            self._path = os.path.join(directory, str(uuid.uuid4()) + ".png")
            img.save(self._path, format="png")
            return self._path

    def save(self, output_bytes, format: str = None, **params):
        """
        保存图像到指定输出
        
        这是对 PIL.Image.save 方法的包装，提供了统一的保存接口。
        
        Args:
            output_bytes: 输出目标（文件路径或字节流）
            format (str, optional): 图像格式（PNG、JPEG 等）
            **params: 传递给 PIL.Image.save 的额外参数
        """
        # 获取 PIL 对象并调用其 save 方法
        img = self.to_raw()
        img.save(output_bytes, format=format, **params)


class AgentAudio(AgentType, str):
    """
    Agent 音频类型
    
    这是对音频数据的智能包装，支持多种音频格式和数据源。
    继承了 str 类型（返回文件路径），同时提供了音频处理的专业功能。
    
    支持的数据源：
    - 文件路径（str/pathlib.Path）：本地文件或 URL
    - PyTorch Tensor：音频波形数据
    - 元组 (samplerate, data)：采样率和音频数据的组合
    
    核心特性：
    - **多格式支持**：WAV、MP3 等常见音频格式
    - **Tensor 处理**：与深度学习框架无缝集成
    - **Jupyter 播放**：在笔记本中提供音频播放控件
    - **网络支持**：支持从 URL 加载音频
    
    依赖要求：
    - soundfile：音频文件读写
    - torch：张量处理
    
    Example:
        ```python
        # 从文件创建
        audio1 = AgentAudio("path/to/audio.wav")
        
        # 从 Tensor 创建
        import torch
        waveform = torch.randn(16000)  # 1秒的随机音频
        audio2 = AgentAudio(waveform, samplerate=16000)
        
        # 从采样率和数据创建
        audio3 = AgentAudio((44100, numpy_array))
        
        # 获取原始数据
        tensor = audio1.to_raw()  # torch.Tensor
        path = str(audio1)        # 文件路径
        ```
    """

    def __init__(self, value, samplerate=16_000):
        """
        初始化 Agent 音频对象
        
        Args:
            value: 音频数据，支持多种格式：
                - str/pathlib.Path: 音频文件路径或 URL
                - torch.Tensor: 音频波形张量
                - tuple: (采样率, 音频数据) 的组合
            samplerate (int): 音频采样率，默认 16kHz
        
        Raises:
            ModuleNotFoundError: 当缺少必需的音频处理库时
            ValueError: 当音频类型不受支持时
        """
        # 检查必需的依赖库
        if not _is_package_available("soundfile") or not _is_package_available("torch"):
            raise ModuleNotFoundError(
                "Please install 'audio' extra to use AgentAudio: `pip install 'smolagents[audio]'`"
            )
        
        import numpy as np
        import torch

        super().__init__(value)

        # 初始化内部状态
        self._path = None      # 文件路径（延迟加载）
        self._tensor = None    # Tensor 数据（已加载）

        self.samplerate = samplerate  # 音频采样率
        
        # 根据输入类型进行处理
        if isinstance(value, (str, pathlib.Path)):
            # 文件路径或 URL：延迟加载
            self._path = value
        elif isinstance(value, torch.Tensor):
            # PyTorch Tensor：直接存储
            self._tensor = value
        elif isinstance(value, tuple):
            # 元组格式：(采样率, 音频数据)
            self.samplerate = value[0]
            if isinstance(value[1], np.ndarray):
                # NumPy 数组转换为 Tensor
                self._tensor = torch.from_numpy(value[1])
            else:
                # 其他数据转换为 Tensor
                self._tensor = torch.tensor(value[1])
        else:
            raise ValueError(f"Unsupported audio type: {type(value)}")

    def _ipython_display_(self, include=None, exclude=None):
        """
        Jupyter/IPython 音频播放支持
        
        在 Jupyter 笔记本中显示音频播放控件，
        用户可以直接播放音频而无需下载文件。
        
        Args:
            include: IPython 显示包含的格式（未使用）
            exclude: IPython 显示排除的格式（未使用）
        """
        from IPython.display import Audio, display
        # 使用音频文件路径和采样率创建播放控件
        display(Audio(self.to_string(), rate=self.samplerate))

    def to_raw(self):
        """
        获取音频的 Tensor 表示
        
        返回音频的原始波形数据作为 PyTorch Tensor，
        这是音频处理和机器学习任务的标准格式。
        
        加载策略：
        1. 如果已有 Tensor，直接返回
        2. 如果是文件路径，使用 soundfile 加载
        3. 支持本地文件和网络 URL
        
        Returns:
            torch.Tensor: 音频波形数据张量
        """
        import soundfile as sf
        import torch

        # 策略1：已有 Tensor 数据
        if self._tensor is not None:
            return self._tensor

        # 策略2：从文件或 URL 加载
        if self._path is not None:
            if "://" in str(self._path):
                # 网络 URL：下载后加载
                response = requests.get(self._path)
                response.raise_for_status()  # 检查 HTTP 错误
                tensor, self.samplerate = sf.read(BytesIO(response.content))
            else:
                # 本地文件：直接加载
                tensor, self.samplerate = sf.read(self._path)
            
            # 转换为 PyTorch Tensor 并缓存
            self._tensor = torch.tensor(tensor)
            return self._tensor

    def to_string(self):
        """
        获取音频的文件路径表示
        
        返回音频文件的路径，用于：
        - 在系统间传递音频引用
        - 与其他音频工具集成
        - 序列化和存储
        
        实现策略：
        1. 如果已有路径，直接返回
        2. 如果是 Tensor，保存为临时 WAV 文件
        
        Returns:
            str: 音频文件路径
        """
        import soundfile as sf

        # 策略1：已有文件路径
        if self._path is not None:
            return self._path

        # 策略2：从 Tensor 保存为文件
        if self._tensor is not None:
            # 创建临时文件
            directory = tempfile.mkdtemp()
            self._path = os.path.join(directory, str(uuid.uuid4()) + ".wav")
            # 使用 soundfile 保存为 WAV 格式
            sf.write(self._path, self._tensor, samplerate=self.samplerate)
            return self._path


# Agent 类型映射表：字符串标识符到具体类型的映射
# 用于根据字符串名称动态创建对应的 Agent 类型对象
_AGENT_TYPE_MAPPING = {"string": AgentText, "image": AgentImage, "audio": AgentAudio}


def handle_agent_input_types(*args, **kwargs):
    """
    处理 Agent 输入类型转换
    
    这个函数将 Agent 类型对象转换为其原始形式，用于工具调用时的参数处理。
    确保工具接收到的是标准的 Python 对象而不是包装后的 Agent 类型。
    
    转换规则：
    - AgentText → str
    - AgentImage → PIL.Image.Image
    - AgentAudio → torch.Tensor
    - 其他类型保持不变
    
    Args:                           
        *args: 位置参数列表
        **kwargs: 关键字参数字典
    
    Returns:
        tuple: (转换后的位置参数, 转换后的关键字参数)
    
    Example:
        ```python
        # 输入包含 Agent 类型
        text = AgentText("hello")
        image = AgentImage("image.jpg")
        
        args, kwargs = handle_agent_input_types(text, image, param=text)
        # args = ("hello", <PIL.Image.Image>)
        # kwargs = {"param": "hello"}
        ```
    """
    # 处理位置参数：如果是 AgentType，转换为原始类型
    args = [(arg.to_raw() if isinstance(arg, AgentType) else arg) for arg in args]
    
    # 处理关键字参数：如果值是 AgentType，转换为原始类型
    kwargs = {k: (v.to_raw() if isinstance(v, AgentType) else v) for k, v in kwargs.items()}
    
    return args, kwargs


def handle_agent_output_types(output: Any, output_type: str | None = None) -> Any:
    """
    处理 Agent 输出类型转换
    
    这个函数将工具的输出转换为适当的 Agent 类型，提供统一的类型包装。
    支持两种转换策略：显式类型指定和自动类型推断。
    
    转换策略：
    1. **显式转换**：根据 output_type 参数强制转换
    2. **自动推断**：根据输出对象的实际类型自动选择包装器
    
    支持的自动推断：
    - str → AgentText
    - PIL.Image.Image → AgentImage  
    - torch.Tensor → AgentAudio
    
    Args:
        output (Any): 工具的原始输出
        output_type (str | None): 期望的输出类型标识符
            可选值："string", "image", "audio"
    
    Returns:
        Any: 包装后的 Agent 类型对象，或原始对象（如果不需要包装）
    
    Example:
        ```python
        # 显式类型转换
        result = handle_agent_output_types("hello", "string")
        # 返回：AgentText("hello")
        
        # 自动类型推断
        pil_img = PIL.Image.open("image.jpg")
        result = handle_agent_output_types(pil_img)
        # 返回：AgentImage(pil_img)
        
        # 不需要包装的类型
        result = handle_agent_output_types(42)
        # 返回：42（保持原样）
        ```
    """
    # 策略1：显式类型转换
    if output_type in _AGENT_TYPE_MAPPING:
        # 根据类型映射表创建对应的 Agent 类型
        decoded_outputs = _AGENT_TYPE_MAPPING[output_type](output)
        return decoded_outputs

    # 策略2：自动类型推断
    # 字符串 → AgentText
    if isinstance(output, str):
        return AgentText(output)
    
    # PIL 图像 → AgentImage
    if isinstance(output, PIL.Image.Image):
        return AgentImage(output)
    
    # PyTorch Tensor → AgentAudio（需要可选依赖）
    try:
        import torch
        if isinstance(output, torch.Tensor):
            return AgentAudio(output)
    except ModuleNotFoundError:
        # 如果没有安装 torch，跳过 Tensor 处理
        pass
    
    # 其他类型：保持原样，不进行包装
    return output


# 导出的公共接口
__all__ = ["AgentType", "AgentImage", "AgentText", "AgentAudio"]
