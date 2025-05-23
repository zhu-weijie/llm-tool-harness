# llm_tool_harness/providers/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from llm_tool_harness.tool import ToolDefinition

# Define common message formats (can be refined)
Message = Dict[str, Any]  # e.g., {"role": "user", "content": "..."}
ToolCall = Dict[str, Any]  # e.g., {"id": "...", "name": "...", "input": {...}}
ToolResult = Dict[
    str, Any
]  # e.g., {"type": "tool_result", "tool_use_id": "...", "content": "..."}


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def __init__(self, model: str, **kwargs: Any):
        self.model = model

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[Any] = None,  # Provider-specific tool choice
        system_prompt: Optional[str] = None,
        **kwargs: Any,  # For other provider-specific params like max_tokens, temperature
    ) -> Tuple[Optional[str], Optional[List[ToolCall]], Optional[Message]]:
        """
        Sends a request to the LLM and gets a response.

        Returns:
            A tuple containing:
            - text_response (Optional[str]): The text part of the LLM's response.
            - tool_calls (Optional[List[ToolCall]]): A list of tool calls requested by the LLM.
            - raw_assistant_message (Optional[Message]): The complete raw message from the assistant for history.
        """
        pass
