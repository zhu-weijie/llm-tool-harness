# llm_tool_harness/agent.py
from typing import List, Any
import anthropic
from .providers.base import LLMProvider, Message
from .tool import Tool, ToolRegistry
from .utils import format_user_text_message, format_tool_result_message

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant. You have access to tools. If you need to use a tool, explain what you're doing."


class LLMAgent:
    def __init__(
        self,
        llm_provider: LLMProvider,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tool_iterations: int = 5,  # Prevent infinite tool call loops
    ):
        self.llm = llm_provider
        self.system_prompt = system_prompt
        self.tool_registry = ToolRegistry()
        self.messages: List[Message] = []
        self.max_tool_iterations = max_tool_iterations

    def register_tool(self, tool: Tool):
        self.tool_registry.register_tool(tool)

    def _add_message(self, role: str, content: Any, **kwargs):
        # For Anthropic, content needs to be a list of dicts for user/assistant roles
        # Tool results are handled differently.
        if role in ["user", "assistant"] and isinstance(content, str):
            self.messages.append(
                {"role": role, "content": [{"type": "text", "text": content}], **kwargs}
            )
        elif role == "user" and isinstance(
            content, list
        ):  #  Already formatted content blocks
            self.messages.append({"role": role, "content": content, **kwargs})
        elif role == "assistant" and isinstance(
            content, list
        ):  # Raw assistant message content
            self.messages.append({"role": role, "content": content, **kwargs})
        elif (
            isinstance(content, dict) and content.get("type") == "tool_result"
        ):  # Our internal tool result format
            # Anthropic expects tool results to be passed within a "user" role message.
            # The AnthropicProvider's chat_completion handles this transformation.
            self.messages.append(
                {
                    "role": "user",  # This role is important for Anthropic
                    "content": [
                        content
                    ],  # Wrap tool_result in a list for Anthropic's content model
                }
            )
        else:
            # Fallback for simple string content or other structures if needed
            self.messages.append({"role": role, "content": content, **kwargs})

    def process_message(self, user_input: str) -> str:
        """
        Processes a single user message, potentially calling tools, and returns the LLM's final response.
        """
        self._add_message("user", format_user_text_message(user_input))

        current_iteration = 0
        while current_iteration < self.max_tool_iterations:
            current_iteration += 1

            tool_definitions = (
                self.tool_registry.get_all_tool_definitions()
                if self.tool_registry
                else None
            )

            # Determine tool_choice: if tools are available, let the LLM decide.
            # Anthropic uses 'auto' or a specific tool request.
            tool_choice = {"type": "auto"} if tool_definitions else None
            if hasattr(self.llm, "client") and isinstance(
                self.llm.client, anthropic.Anthropic
            ):  # Anthropic specific
                pass  # 'auto' is fine
            # else: add other provider specific logic for tool_choice if needed

            text_response, tool_calls, raw_assistant_message = self.llm.chat_completion(
                messages=self.messages,
                tools=tool_definitions,
                tool_choice=tool_choice,
                system_prompt=self.system_prompt,
            )

            if raw_assistant_message:
                # Add the assistant's full message (including potential partial text before tool calls)
                # We'll filter out existing tool_calls to avoid duplication when adding to history
                # The raw_assistant_message already contains the correct structure.
                self.messages.append(raw_assistant_message)

            if not tool_calls:
                # No tools called, or LLM decided to respond directly after tool use
                return text_response or "No response from assistant."

            # If we are here, tools were called.
            tool_results_for_next_iteration: List[Message] = []

            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_id = tool_call.get("id")
                tool_input = tool_call.get("input", {})

                if not tool_name or not tool_id:
                    # Malformed tool call from LLM
                    tool_output_str = (
                        f"Error: Malformed tool call from LLM: {tool_call}"
                    )
                    print(f"Agent: {tool_output_str}")  # Log to console
                    # Create a tool result message indicating the error to send back to LLM
                    tool_results_for_next_iteration.append(
                        format_tool_result_message(
                            tool_id or "unknown_id", tool_output_str, is_error=True
                        )
                    )
                    continue

                print(
                    f"Agent: Requesting to use tool '{tool_name}' with ID '{tool_id}' and input: {tool_input}"
                )
                tool_to_execute = self.tool_registry.get_tool(tool_name)

                if tool_to_execute:
                    try:
                        tool_output = tool_to_execute.execute(**tool_input)
                        tool_output_str = str(
                            tool_output
                        )  # Ensure it's a string for now
                        print(
                            f"Agent: Tool '{tool_name}' executed. Output:\n{tool_output_str}"
                        )
                        tool_results_for_next_iteration.append(
                            format_tool_result_message(tool_id, tool_output_str)
                        )
                    except Exception as e:
                        error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                        print(f"Agent: {error_msg}")
                        tool_results_for_next_iteration.append(
                            format_tool_result_message(
                                tool_id, error_msg, is_error=True
                            )
                        )
                else:
                    error_msg = f"Error: Tool '{tool_name}' not found."
                    print(f"Agent: {error_msg}")
                    tool_results_for_next_iteration.append(
                        format_tool_result_message(tool_id, error_msg, is_error=True)
                    )

            # Add all tool results to history for the next LLM call
            # Anthropic expects tool results to be inside a "user" role message.
            # The format_tool_result_message and _add_message handle the specific structure.
            for res in tool_results_for_next_iteration:
                self._add_message(
                    role="user", content=res
                )  # Role 'user' is for Anthropic tool results

            # Loop back to call the LLM again with the tool results
            # The text_response from this intermediate step (if any) is usually just an ack of tool use,
            # we care about the final response after tools are processed.
            if text_response:  # Print any intermediate text from assistant
                print(f"Agent: {text_response}")

        # Max iterations reached
        return "Agent reached maximum tool iterations. Returning last text response or error."

    def chat_loop(self):
        """Starts an interactive chat loop in the console."""
        print("\n=== LLM Agent Loop ===\n")
        print(f"System: {self.system_prompt}")
        if self.tool_registry.get_all_tool_definitions():
            print(
                "System: Tools available:",
                ", ".join([t.name for t in self.tool_registry._tools.values()]),
            )
        print("Type 'exit' or 'quit' to end.\n")

        try:
            while True:
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("\nExiting agent loop. Goodbye!")
                    break
                if not user_input.strip():
                    continue

                assistant_response = self.process_message(user_input)
                print(f"Agent: {assistant_response}")
        except KeyboardInterrupt:
            print("\n\nExiting. Goodbye!")
        except Exception as e:
            print(f"\n\nAn error occurred: {str(e)}")
            import traceback

            traceback.print_exc()
