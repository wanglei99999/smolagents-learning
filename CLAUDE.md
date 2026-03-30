# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install for development
pip install -e ".[dev]"
# Or with uv (faster)
uv pip install -e "smolagents[dev] @ ."

# Lint and format checks
make quality          # ruff check + ruff format --check
make style            # auto-fix with ruff

# Run tests
make test             # pytest ./tests/
pytest ./tests/test_agents.py        # single test file
pytest ./tests/test_agents.py::TestClass::test_method  # single test
```

## Architecture

The library implements a **ReAct loop** where agents alternate between planning, acting (generating Python code or tool calls), and observing results:

```
Input → Initialize Memory → Loop: Plan → Action → Execute → Observe → Final Answer
```

### Core files

- **`src/smolagents/agents.py`** — The heart of the library. Contains:
  - `MultiStepAgent`: Abstract base implementing the ReAct loop (`_run_stream`, `_step_stream`)
  - `CodeAgent`: LLM generates Python code snippets, executed in a sandbox (~30% fewer steps)
  - `ToolCallingAgent`: LLM generates JSON tool calls, supports parallel execution

- **`src/smolagents/models.py`** — LLM provider adapters. All implement `Model.__call__()`. Includes `InferenceClientModel` (HF), `OpenAIModel`, `LiteLLMModel`, `TransformersModel`, `OllamaModel`, etc.

- **`src/smolagents/tools.py`** — `Tool` base class with name/description/inputs schema. `ToolCollection` loads tools from MCP servers, Hub, or LangChain.

- **`src/smolagents/memory.py`** — `AgentMemory` stores `ActionStep`, `PlanningStep`, `FinalAnswerStep` objects. Converts history to LLM chat messages.

- **`src/smolagents/local_python_executor.py`** — Secure Python sandbox via RestrictedPython. Maintains an import whitelist. Used by `CodeAgent` for local execution.

- **`src/smolagents/remote_executors.py`** — `E2BExecutor`, `DockerExecutor`, `ModalExecutor`, `WasmExecutor` for sandboxed remote execution.

### Data flow

1. `agent.run(task)` → `_run_stream()` enters the ReAct loop
2. Each iteration calls `_step_stream()` (overridden per agent type)
3. `CodeAgent` parses LLM output for ```` ```python ``` ```` blocks → executes via `LocalPythonExecutor` or remote executor
4. `ToolCallingAgent` parses tool call JSON → executes tools (possibly in parallel via `ThreadPoolExecutor`)
5. Results are stored as `ActionStep` in `AgentMemory`
6. Loop exits when `FinalAnswerTool` is called or `max_steps` reached

### Prompt templates

YAML prompt templates live in `src/smolagents/prompts/`. The system prompt is built dynamically in `initialize_system_prompt()` — injecting available tools, authorized imports, and formatting instructions.

## Code style

- Line length: 119 characters (Ruff)
- Ruff rules: `E`, `F`, `I`, `W` (ignores `F403`, `E501`)
- Python ≥ 3.10 required

## Learning notes

This repo contains personal study notes in Chinese under `notes/` and `READING_GUIDE.md`, `LEARNING_CHECKLIST.md`, `SMOLAGENTS_ARCHITECTURE.md`. These are learning materials, not production docs.
