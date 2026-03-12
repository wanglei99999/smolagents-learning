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
# default_tools.py —— smolagents 的内置工具集
#
# 本文件定义了 Agent 可以使用的所有默认工具，包括：
#   1. PythonInterpreterTool - Python 代码解释器（用于 ToolCallingAgent）
#   2. FinalAnswerTool - 返回最终答案（所有 Agent 必备）
#   3. UserInputTool - 请求用户输入
#   4. WebSearchTool 系列 - 网页搜索工具（DuckDuckGo/Google/Bing/API）
#   5. VisitWebpageTool - 访问网页并提取内容
#   6. WikipediaSearchTool - 维基百科搜索
#   7. SpeechToTextTool - 语音转文字
#
# 工具设计原则：
#   - 每个工具都继承自 Tool 基类
#   - 必须定义 name, description, inputs, output_type
#   - 核心逻辑在 forward() 方法中实现
# =============================================================================
from dataclasses import dataclass
from typing import Any

from .local_python_executor import (
    BASE_BUILTIN_MODULES,
    BASE_PYTHON_TOOLS,
    MAX_EXECUTION_TIME_SECONDS,
    evaluate_python_code,
)
from .tools import PipelineTool, Tool


@dataclass
class PreTool:
    name: str
    inputs: dict[str, str]
    output_type: type
    task: str
    description: str
    repo_id: str


class PythonInterpreterTool(Tool):
    """
    Python 代码解释器工具
    
    这个工具主要用于 ToolCallingAgent，让它可以执行 Python 代码。
    注意：CodeAgent 不需要这个工具，因为它直接执行代码。
    
    功能：
    - 在沙箱环境中执行 Python 代码
    - 限制可导入的模块（安全控制）
    - 捕获 print 输出和返回值
    - 设置执行超时（防止死循环）
    
    使用场景：
    - ToolCallingAgent 需要执行计算
    - 需要处理数据但不想用 CodeAgent
    """
    name = "python_interpreter"
    description = "This is a tool that evaluates python code. It can be used to perform calculations."
    inputs = {
        "code": {
            "type": "string",
            "description": "The python code to run in interpreter",
        }
    }
    output_type = "string"

    def __init__(self, *args, authorized_imports=None, timeout_seconds=MAX_EXECUTION_TIME_SECONDS, **kwargs):
        """
        初始化 Python 解释器工具
        
        Args:
            authorized_imports: 额外授权的 Python 模块列表
            timeout_seconds: 代码执行超时时间（秒）
        """
        # 设置 import 白名单：基础模块 + 用户授权的模块
        if authorized_imports is None:
            self.authorized_imports = list(set(BASE_BUILTIN_MODULES))
        else:
            self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(authorized_imports))
        
        # 更新输入描述，告诉 LLM 可以导入哪些模块
        self.inputs = {
            "code": {
                "type": "string",
                "description": (
                    "The code snippet to evaluate. All variables used in this snippet must be defined in this same snippet, "
                    f"else you will get an error. This code can only import the following python libraries: {self.authorized_imports}."
                ),
            }
        }
        self.base_python_tools = BASE_PYTHON_TOOLS  # 基础工具（如 final_answer）
        self.python_evaluator = evaluate_python_code  # 代码执行器
        self.timeout_seconds = timeout_seconds
        super().__init__(*args, **kwargs)

    def forward(self, code: str) -> str:
        """
        执行 Python 代码并返回结果
        
        Args:
            code: 要执行的 Python 代码字符串
            
        Returns:
            格式化的执行结果，包含 stdout 和返回值
            
        Example:
            >>> tool = PythonInterpreterTool()
            >>> result = tool.forward("print('Hello'); 2 + 3")
            >>> print(result)
            Stdout:
            Hello
            Output: 5
        """
        state = {}  # 用于存储执行状态（如 print 输出）
        output = str(
            self.python_evaluator(
                code,
                state=state,
                static_tools=self.base_python_tools,
                authorized_imports=self.authorized_imports,
                timeout_seconds=self.timeout_seconds,
            )[0]  # 第一个元素是返回值，第二个元素是 is_final_answer 标志
        )
        # 返回格式化的结果：stdout + 返回值
        return f"Stdout:\n{str(state['_print_outputs'])}\nOutput: {output}"


class FinalAnswerTool(Tool):
    """
    最终答案工具（所有 Agent 的必备工具）
    
    这是一个特殊的工具，用于标记任务完成并返回最终答案。
    当 Agent 调用这个工具时，ReAct 循环会终止。
    
    工作原理：
    - ToolCallingAgent: 调用 {"tool": "final_answer", "arguments": {"answer": "..."}}
    - CodeAgent: 调用 final_answer("...")
    
    注意：这个工具不做任何处理，只是原样返回答案
    """
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {"answer": {"type": "any", "description": "The final answer to the problem"}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        """直接返回答案，不做任何处理"""
        return answer


class UserInputTool(Tool):
    """
    用户输入工具
    
    允许 Agent 在执行过程中向用户提问并获取输入。
    这对于需要用户确认或提供额外信息的场景很有用。
    
    使用场景：
    - Agent 需要用户确认某个操作
    - Agent 需要用户提供额外信息（如密码、选择等）
    - 交互式对话流程
    
    Example:
        Agent: "我找到了3个选项，你想选择哪一个？"
        User: "选择第2个"
    """
    name = "user_input"
    description = "Asks for user's input on a specific question"
    inputs = {"question": {"type": "string", "description": "The question to ask the user"}}
    output_type = "string"

    def forward(self, question):
        """
        向用户提问并等待输入
        
        Args:
            question: 要问用户的问题
            
        Returns:
            用户的输入（字符串）
        """
        user_input = input(f"{question} => Type your answer here:")
        return user_input


class DuckDuckGoSearchTool(Tool):
    """
    DuckDuckGo 网页搜索工具
    
    使用 DuckDuckGo 搜索引擎进行网页搜索（类似 Google 搜索）。
    优点：免费、无需 API key、隐私友好
    
    功能：
    - 搜索网页并返回前 N 个结果
    - 支持速率限制（防止被封禁）
    - 返回格式化的 Markdown 结果（标题、链接、摘要）
    
    Web search tool that performs searches using the DuckDuckGo search engine.

    Args:
        max_results (`int`, default `10`): Maximum number of search results to return.
            最多返回多少个搜索结果
        rate_limit (`float`, default `1.0`): Maximum queries per second. Set to `None` to disable rate limiting.
            每秒最多查询次数（None 表示不限制）
        **kwargs: Additional keyword arguments for the `DDGS` client.

    Examples:
        ```python
        >>> from smolagents import DuckDuckGoSearchTool
        >>> web_search_tool = DuckDuckGoSearchTool(max_results=5, rate_limit=2.0)
        >>> results = web_search_tool("Hugging Face")
        >>> print(results)
        ```
    """

    name = "web_search"
    description = """Performs a duckduckgo web search based on your query (think a Google search) then returns the top search results."""
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(self, max_results: int = 10, rate_limit: float | None = 1.0, **kwargs):
        super().__init__()
        self.max_results = max_results
        self.rate_limit = rate_limit
        # 计算最小请求间隔（用于速率限制）
        self._min_interval = 1.0 / rate_limit if rate_limit else 0.0
        self._last_request_time = 0.0
        
        # 导入 DuckDuckGo 搜索库
        try:
            from ddgs import DDGS
        except ImportError as e:
            raise ImportError(
                "You must install package `ddgs` to run this tool: for instance run `pip install ddgs`."
            ) from e
        self.ddgs = DDGS(**kwargs)

    def forward(self, query: str) -> str:
        """
        执行搜索并返回格式化的结果
        
        Args:
            query: 搜索查询字符串
            
        Returns:
            Markdown 格式的搜索结果
            
        Example:
            ## Search Results
            
            [Python Tutorial](https://example.com)
            Learn Python programming...
            
            [Python Documentation](https://docs.python.org)
            Official Python docs...
        """
        # 执行速率限制（避免请求过快被封禁）
        self._enforce_rate_limit()
        
        # 执行搜索
        results = self.ddgs.text(query, max_results=self.max_results)
        if len(results) == 0:
            raise Exception("No results found! Try a less restrictive/shorter query.")
        
        # 格式化结果为 Markdown
        postprocessed_results = [f"[{result['title']}]({result['href']})\n{result['body']}" for result in results]
        return "## Search Results\n\n" + "\n\n".join(postprocessed_results)

    def _enforce_rate_limit(self) -> None:
        """
        强制执行速率限制
        
        如果距离上次请求的时间太短，会 sleep 等待。
        这样可以避免请求过快被 DuckDuckGo 封禁。
        """
        import time

        # 如果没有设置速率限制，直接返回
        if not self.rate_limit:
            return

        now = time.time()
        elapsed = now - self._last_request_time
        # 如果距离上次请求时间太短，等待
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()


class GoogleSearchTool(Tool):
    name = "web_search"
    description = """Performs a google web search for your query then returns a string of the top search results."""
    inputs = {
        "query": {"type": "string", "description": "The search query to perform."},
        "filter_year": {
            "type": "integer",
            "description": "Optionally restrict results to a certain year",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, provider: str = "serpapi"):
        super().__init__()
        import os

        self.provider = provider
        if provider == "serpapi":
            self.organic_key = "organic_results"
            api_key_env_name = "SERPAPI_API_KEY"
        else:
            self.organic_key = "organic"
            api_key_env_name = "SERPER_API_KEY"
        self.api_key = os.getenv(api_key_env_name)
        if self.api_key is None:
            raise ValueError(f"Missing API key. Make sure you have '{api_key_env_name}' in your env variables.")

    def forward(self, query: str, filter_year: int | None = None) -> str:
        import requests

        if self.provider == "serpapi":
            params = {
                "q": query,
                "api_key": self.api_key,
                "engine": "google",
                "google_domain": "google.com",
            }
            base_url = "https://serpapi.com/search.json"
        else:
            params = {
                "q": query,
                "api_key": self.api_key,
            }
            base_url = "https://google.serper.dev/search"
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            results = response.json()
        else:
            raise ValueError(response.json())

        if self.organic_key not in results.keys():
            if filter_year is not None:
                raise Exception(
                    f"No results found for query: '{query}' with filtering on year={filter_year}. Use a less restrictive query or do not filter on year."
                )
            else:
                raise Exception(f"No results found for query: '{query}'. Use a less restrictive query.")
        if len(results[self.organic_key]) == 0:
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            return f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter."

        web_snippets = []
        if self.organic_key in results:
            for idx, page in enumerate(results[self.organic_key]):
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                web_snippets.append(redacted_version)

        return "## Search Results\n" + "\n\n".join(web_snippets)


class ApiWebSearchTool(Tool):
    """Web search tool that performs API-based searches.
    By default, it uses the Brave Search API.

    This tool implements a rate limiting mechanism to ensure compliance with API usage policies.
    By default, it limits requests to 1 query per second.

    Args:
        endpoint (`str`): API endpoint URL. Defaults to Brave Search API.
        api_key (`str`): API key for authentication.
        api_key_name (`str`): Environment variable name containing the API key. Defaults to "BRAVE_API_KEY".
        headers (`dict`, *optional*): Headers for API requests.
        params (`dict`, *optional*): Parameters for API requests.
        rate_limit (`float`, default `1.0`): Maximum queries per second. Set to `None` to disable rate limiting.

    Examples:
        ```python
        >>> from smolagents import ApiWebSearchTool
        >>> web_search_tool = ApiWebSearchTool(rate_limit=50.0)
        >>> results = web_search_tool("Hugging Face")
        >>> print(results)
        ```
    """

    name = "web_search"
    description = "Performs a web search for a query and returns a string of the top search results formatted as markdown with titles, URLs, and descriptions."
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(
        self,
        endpoint: str = "",
        api_key: str = "",
        api_key_name: str = "",
        headers: dict = None,
        params: dict = None,
        rate_limit: float | None = 1.0,
    ):
        import os

        super().__init__()
        self.endpoint = endpoint or "https://api.search.brave.com/res/v1/web/search"
        self.api_key_name = api_key_name or "BRAVE_API_KEY"
        self.api_key = api_key or os.getenv(self.api_key_name)
        self.headers = headers or {"X-Subscription-Token": self.api_key}
        self.params = params or {"count": 10}
        self.rate_limit = rate_limit
        self._min_interval = 1.0 / rate_limit if rate_limit else 0.0
        self._last_request_time = 0.0

    def _enforce_rate_limit(self) -> None:
        import time

        # No rate limit enforced
        if not self.rate_limit:
            return

        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def forward(self, query: str) -> str:
        import requests

        self._enforce_rate_limit()
        params = {**self.params, "q": query}
        response = requests.get(self.endpoint, headers=self.headers, params=params)
        response.raise_for_status()
        data = response.json()
        results = self.extract_results(data)
        return self.format_markdown(results)

    def extract_results(self, data: dict) -> list:
        results = []
        for result in data.get("web", {}).get("results", []):
            results.append(
                {"title": result["title"], "url": result["url"], "description": result.get("description", "")}
            )
        return results

    def format_markdown(self, results: list) -> str:
        if not results:
            return "No results found."
        return "## Search Results\n\n" + "\n\n".join(
            [
                f"{idx}. [{result['title']}]({result['url']})\n{result['description']}"
                for idx, result in enumerate(results, start=1)
            ]
        )


class WebSearchTool(Tool):
    name = "web_search"
    description = "Performs a web search for a query and returns a string of the top search results formatted as markdown with titles, links, and descriptions."
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(self, max_results: int = 10, engine: str = "duckduckgo"):
        super().__init__()
        self.max_results = max_results
        self.engine = engine

    def forward(self, query: str) -> str:
        results = self.search(query)
        if len(results) == 0:
            raise Exception("No results found! Try a less restrictive/shorter query.")
        return self.parse_results(results)

    def search(self, query: str) -> list:
        if self.engine == "duckduckgo":
            return self.search_duckduckgo(query)
        elif self.engine == "bing":
            return self.search_bing(query)
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")

    def parse_results(self, results: list) -> str:
        return "## Search Results\n\n" + "\n\n".join(
            [f"[{result['title']}]({result['link']})\n{result['description']}" for result in results]
        )

    def search_duckduckgo(self, query: str) -> list:
        import requests

        response = requests.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        parser = self._create_duckduckgo_parser()
        parser.feed(response.text)
        return parser.results

    def _create_duckduckgo_parser(self):
        from html.parser import HTMLParser

        class SimpleResultParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.current = {}
                self.capture_title = False
                self.capture_description = False
                self.capture_link = False

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "a" and attrs.get("class") == "result-link":
                    self.capture_title = True
                elif tag == "td" and attrs.get("class") == "result-snippet":
                    self.capture_description = True
                elif tag == "span" and attrs.get("class") == "link-text":
                    self.capture_link = True

            def handle_endtag(self, tag):
                if tag == "a" and self.capture_title:
                    self.capture_title = False
                elif tag == "td" and self.capture_description:
                    self.capture_description = False
                elif tag == "span" and self.capture_link:
                    self.capture_link = False
                elif tag == "tr":
                    # Store current result if all parts are present
                    if {"title", "description", "link"} <= self.current.keys():
                        self.current["description"] = " ".join(self.current["description"])
                        self.results.append(self.current)
                        self.current = {}

            def handle_data(self, data):
                if self.capture_title:
                    self.current["title"] = data.strip()
                elif self.capture_description:
                    self.current.setdefault("description", [])
                    self.current["description"].append(data.strip())
                elif self.capture_link:
                    self.current["link"] = "https://" + data.strip()

        return SimpleResultParser()

    def search_bing(self, query: str) -> list:
        import xml.etree.ElementTree as ET

        import requests

        response = requests.get(
            "https://www.bing.com/search",
            params={"q": query, "format": "rss"},
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        items = root.findall(".//item")
        results = [
            {
                "title": item.findtext("title"),
                "link": item.findtext("link"),
                "description": item.findtext("description"),
            }
            for item in items[: self.max_results]
        ]
        return results


class VisitWebpageTool(Tool):
    name = "visit_webpage"
    description = (
        "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "The url of the webpage to visit.",
        }
    }
    output_type = "string"

    def __init__(self, max_output_length: int = 40000):
        super().__init__()
        self.max_output_length = max_output_length

    def _truncate_content(self, content: str, max_length: int) -> str:
        if len(content) <= max_length:
            return content
        return (
            content[:max_length] + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
        )

    def forward(self, url: str) -> str:
        try:
            import re

            import requests
            from markdownify import markdownify
            from requests.exceptions import RequestException
        except ImportError as e:
            raise ImportError(
                "You must install packages `markdownify` and `requests` to run this tool: for instance run `pip install markdownify requests`."
            ) from e
        try:
            # Send a GET request to the URL with a 20-second timeout
            response = requests.get(url, timeout=20)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Convert the HTML content to Markdown
            markdown_content = markdownify(response.text).strip()

            # Remove multiple line breaks
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

            return self._truncate_content(markdown_content, self.max_output_length)

        except requests.exceptions.Timeout:
            return "The request timed out. Please try again later or check the URL."
        except RequestException as e:
            return f"Error fetching the webpage: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"


class WikipediaSearchTool(Tool):
    """
    Search Wikipedia and return the summary or full text of the requested article, along with the page URL.

    Attributes:
        user_agent (`str`): Custom user-agent string to identify the project. This is required as per Wikipedia API policies.
            See: https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy
        language (`str`, default `"en"`): Language in which to retrieve Wikipedia article.
            See: http://meta.wikimedia.org/wiki/List_of_Wikipedias
        content_type (`Literal["summary", "text"]`, default `"text"`): Type of content to fetch. Can be "summary" for a short summary or "text" for the full article.
        extract_format (`Literal["HTML", "WIKI"]`, default `"WIKI"`): Extraction format of the output. Can be `"WIKI"` or `"HTML"`.

    Example:
        ```python
        >>> from smolagents import CodeAgent, InferenceClientModel, WikipediaSearchTool
        >>> agent = CodeAgent(
        >>>     tools=[
        >>>            WikipediaSearchTool(
        >>>                user_agent="MyResearchBot (myemail@example.com)",
        >>>                language="en",
        >>>                content_type="summary",  # or "text"
        >>>                extract_format="WIKI",
        >>>            )
        >>>        ],
        >>>     model=InferenceClientModel(),
        >>> )
        >>> agent.run("Python_(programming_language)")
        ```
    """

    name = "wikipedia_search"
    description = "Searches Wikipedia and returns a summary or full text of the given topic, along with the page URL."
    inputs = {
        "query": {
            "type": "string",
            "description": "The topic to search on Wikipedia.",
        }
    }
    output_type = "string"

    def __init__(
        self,
        user_agent: str = "Smolagents (myemail@example.com)",
        language: str = "en",
        content_type: str = "text",
        extract_format: str = "WIKI",
    ):
        super().__init__()
        try:
            import wikipediaapi
        except ImportError as e:
            raise ImportError(
                "You must install `wikipedia-api` to run this tool: for instance run `pip install wikipedia-api`"
            ) from e
        if not user_agent:
            raise ValueError("User-agent is required. Provide a meaningful identifier for your project.")

        self.user_agent = user_agent
        self.language = language
        self.content_type = content_type

        # Map string format to wikipediaapi.ExtractFormat
        extract_format_map = {
            "WIKI": wikipediaapi.ExtractFormat.WIKI,
            "HTML": wikipediaapi.ExtractFormat.HTML,
        }

        if extract_format not in extract_format_map:
            raise ValueError("Invalid extract_format. Choose between 'WIKI' or 'HTML'.")

        self.extract_format = extract_format_map[extract_format]

        self.wiki = wikipediaapi.Wikipedia(
            user_agent=self.user_agent, language=self.language, extract_format=self.extract_format
        )

    def forward(self, query: str) -> str:
        try:
            page = self.wiki.page(query)

            if not page.exists():
                return f"No Wikipedia page found for '{query}'. Try a different query."

            title = page.title
            url = page.fullurl

            if self.content_type == "summary":
                text = page.summary
            elif self.content_type == "text":
                text = page.text
            else:
                return "⚠️ Invalid `content_type`. Use either 'summary' or 'text'."

            return f"✅ **Wikipedia Page:** {title}\n\n**Content:** {text}\n\n🔗 **Read more:** {url}"

        except Exception as e:
            return f"Error fetching Wikipedia summary: {str(e)}"


class SpeechToTextTool(PipelineTool):
    default_checkpoint = "openai/whisper-large-v3-turbo"
    description = "This is a tool that transcribes an audio into text. It returns the transcribed text."
    name = "transcriber"
    inputs = {
        "audio": {
            "type": "audio",
            "description": "The audio to transcribe. Can be a local path, an url, or a tensor.",
        }
    }
    output_type = "string"

    def __new__(cls, *args, **kwargs):
        from transformers.models.whisper import WhisperForConditionalGeneration, WhisperProcessor

        cls.pre_processor_class = WhisperProcessor
        cls.model_class = WhisperForConditionalGeneration
        return super().__new__(cls)

    def encode(self, audio):
        from .agent_types import AgentAudio

        audio = AgentAudio(audio).to_raw()
        return self.pre_processor(audio, return_tensors="pt")

    def forward(self, inputs):
        return self.model.generate(inputs["input_features"])

    def decode(self, outputs):
        return self.pre_processor.batch_decode(outputs, skip_special_tokens=True)[0]


# =============================================================================
# 工具映射表：用于 add_base_tools=True 时自动添加工具
# =============================================================================
TOOL_MAPPING = {
    tool_class.name: tool_class
    for tool_class in [
        PythonInterpreterTool,      # Python 代码解释器
        DuckDuckGoSearchTool,        # DuckDuckGo 搜索
        VisitWebpageTool,            # 访问网页
    ]
}

# =============================================================================
# 导出的工具类列表
# =============================================================================
__all__ = [
    "ApiWebSearchTool",          # API 网页搜索（Brave Search）
    "PythonInterpreterTool",     # Python 解释器
    "FinalAnswerTool",           # 最终答案工具
    "UserInputTool",             # 用户输入工具
    "WebSearchTool",             # 通用网页搜索（支持多引擎）
    "DuckDuckGoSearchTool",      # DuckDuckGo 搜索
    "GoogleSearchTool",          # Google 搜索（需要 API key）
    "VisitWebpageTool",          # 访问网页
    "WikipediaSearchTool",       # 维基百科搜索
    "SpeechToTextTool",          # 语音转文字
]
