# llm_tool_harness/utils.py
from typing import List, Any, Dict


def format_user_text_message(text: str) -> List[Dict[str, Any]]:
    """Formats a simple text string into the content block for user messages."""
    return [{"type": "text", "text": text}]


def format_tool_result_message(
    tool_use_id: str, output: Any, is_error: bool = False
) -> Dict[str, Any]:
    """
    Formats the tool execution output into a message for the LLM.
    Anthropic expects tool_result under a "user" role message.
    """
    content_blocks = []
    if isinstance(output, str):
        content_blocks.append({"type": "text", "text": output})
    elif (
        isinstance(output, dict) and "type" in output
    ):  #  Assume it's already a content block
        content_blocks.append(output)
    elif isinstance(output, list):  # Assume it's a list of content blocks
        content_blocks.extend(output)
    else:  # Default to stringifying
        content_blocks.append({"type": "text", "text": str(output)})

    return {
        "type": "tool_result",  # This key is for internal tracking, actual structure depends on LLM
        "tool_use_id": tool_use_id,
        "content": content_blocks,  # For Anthropic, this will be wrapped in a user message later
        "is_error": is_error,
    }
