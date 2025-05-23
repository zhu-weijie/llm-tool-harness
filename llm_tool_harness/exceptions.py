# llm_tool_harness/exceptions.py


class LLMToolHarnessError(Exception):
    """Base exception for the library."""

    pass


class LLMProviderError(LLMToolHarnessError):
    """Exception related to LLM provider interactions."""

    pass


class ToolExecutionError(LLMToolHarnessError):
    """Exception during tool execution."""

    pass
