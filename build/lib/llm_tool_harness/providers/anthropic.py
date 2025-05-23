# llm_tool_harness/providers/anthropic.py
import os
from typing import List, Dict, Any, Tuple, Optional
import anthropic
from .base import LLMProvider, Message, ToolCall, ToolDefinition
from ..exceptions import LLMProviderError


# Helper to convert our generic Message format to Anthropic's content block
def _format_content_for_anthropic(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    elif isinstance(
        content, list
    ):  # Assumes it's already in Anthropic's content block format
        # Basic validation, can be improved
        for item in content:
            if not isinstance(item, dict) or "type" not in item:
                raise LLMProviderError(f"Invalid content item for Anthropic: {item}")
        return content
    raise LLMProviderError(f"Unsupported content type for Anthropic: {type(content)}")


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any):
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMProviderError(
                "ANTHROPIC_API_KEY not found in environment or provided."
            )
        try:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Anthropic client: {e}")
        self.extra_params = kwargs

    def chat_completion(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[Any] = None,  # Anthropic uses specific tool_choice dict
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[str], Optional[List[ToolCall]], Optional[Message]]:
        # Anthropic needs messages with "role" and "content".
        # Content needs to be a list of blocks.
        anthropic_messages = []
        for msg in messages:
            # Ensure content is in Anthropic's list-of-blocks format
            if "content" in msg:
                if isinstance(msg["content"], list) and all(
                    isinstance(item, dict) and "type" in item for item in msg["content"]
                ):
                    # Content is already in the correct list-of-dicts format
                    anthropic_messages.append(msg)
                elif (
                    isinstance(msg["content"], str) and msg.get("type") == "tool_result"
                ):
                    # This is a tool result, reformat
                    anthropic_messages.append(
                        {
                            "role": "user",  # Tool results are from user perspective for Anthropic
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": msg["tool_use_id"],
                                    "content": [
                                        {"type": "text", "text": str(msg["content"])}
                                    ],  # Ensure content is string
                                }
                            ],
                        }
                    )
                elif isinstance(msg["content"], str):
                    anthropic_messages.append(
                        {
                            "role": msg["role"],
                            "content": [{"type": "text", "text": msg["content"]}],
                        }
                    )
                else:  # Assume it's a list of user inputs already formatted
                    anthropic_messages.append(
                        {
                            "role": msg["role"],
                            "content": msg[
                                "content"
                            ],  # Expects [{'type': 'text', 'text': '...'}, ...]
                        }
                    )
            else:
                anthropic_messages.append(
                    msg
                )  # If no content, pass as is (e.g. assistant message start)

        api_params = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.extra_params.get(
                "max_tokens", 4096
            ),  # Default to a higher value
            **self.extra_params,
            **kwargs,
        }
        if system_prompt:
            api_params["system"] = system_prompt
        if tools:
            api_params["tools"] = tools
        if tool_choice:
            api_params["tool_choice"] = tool_choice

        try:
            response = self.client.messages.create(**api_params)
        except Exception as e:
            raise LLMProviderError(f"Anthropic API error: {e}")

        text_response = ""
        tool_calls: List[ToolCall] = []
        assistant_response_content: List[Dict] = []

        for content_block in response.content:
            assistant_response_content.append(
                content_block.model_dump()
            )  # Save raw content
            if content_block.type == "text":
                text_response += content_block.text
            elif content_block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": content_block.id,
                        "name": content_block.name,
                        "input": content_block.input,
                        "type": "tool_use",  # For consistency with our ToolCall type
                    }
                )

        raw_assistant_message: Message = {
            "role": "assistant",
            "content": assistant_response_content,  # Store the structured content
        }
        # If only tool calls, text_response might be empty. Return None if truly empty.
        return text_response or None, tool_calls or None, raw_assistant_message
