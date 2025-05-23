# LLM Tool Harness

[![PyPI version](https://badge.fury.io/py/llm-tool-harness.svg)](https://badge.fury.io/py/llm-tool-harness)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Versions](https://img.shields.io/pypi/pyversions/llm-tool-harness)
<!-- Add build status badge once CI is set up, e.g., GitHub Actions -->
<!-- [![CI](https://github.com/yourusername/llm-tool-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/llm-tool-harness/actions/workflows/ci.yml) -->

A Python library to easily build LLM agents with tool-using capabilities, allowing them to interact with external systems and perform actions.

## Key Features

*   **Simple Agent Creation:** Quickly set up an LLM-powered agent.
*   **Easy Tool Definition:** Define custom tools with clear schemas and connect them to your Python functions.
*   **Pluggable LLM Providers:** Start with Anthropic (Claude) and easily extend to other LLM providers.
*   **Conversation Management:** Automatically handles conversation history and the flow of tool calls and results.
*   **Extensible:** Designed to be a foundation for more complex agentic systems.

## Installation

```bash
pip install llm-tool-harness
```

You will also need to set up API keys for the LLM providers you intend to use. For Anthropic:

```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

## Quick Start

Here's how to create a simple agent that can use a bash tool:

```python
import os
import subprocess
from llm_tool_harness import LLMAgent, Tool, AnthropicProvider, ToolInputSchema

# 1. Define your tool implementation function
def execute_bash_command(command: str) -> str:
    """Execute a bash command and return a formatted string with the results."""
    print(f"Executing bash command: {command}")
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=30
        )
        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nEXIT CODE: {result.returncode}"
        print(f"Bash output:\n{output}")
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Command '{command}' timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command '{command}': {str(e)}"

def main():
    # Ensure ANTHROPIC_API_KEY is set (or handle error)
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        return

    # 2. Create an LLM provider instance
    # Replace with your desired Claude model, e.g., "claude-3-opus-20240229", "claude-3-haiku-20240307"
    anthropic_llm = AnthropicProvider(model="claude-3-sonnet-20240229", max_tokens=2000)

    # 3. Create the LLM Agent
    agent = LLMAgent(
        llm_provider=anthropic_llm,
        system_prompt="""You are a helpful AI assistant with access to a bash terminal.
        You can help the user by executing commands and interpreting the results.
        Be careful with destructive commands and always ask for confirmation if unsure.
        You have access to the 'bash' tool which allows you to run shell commands."""
    )

    # 4. Define the tool schema and create a Tool object
    bash_input_schema: ToolInputSchema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute (e.g., 'ls -l', 'echo hello')"
            }
        },
        "required": ["command"]
    }
    bash_tool = Tool(
        name="bash",
        description="Execute bash shell commands on the local system and return the output.",
        input_schema=bash_input_schema,
        implementation=execute_bash_command
    )

    # 5. Register the tool with the agent
    agent.register_tool(bash_tool)

    # 6. Start the interactive chat loop
    # Or, for programmatic use:
    # response = agent.process_message("What files are in the current directory?")
    # print(f"Agent: {response}")
    agent.chat_loop()

if __name__ == "__main__":
    main()

```

## Core Concepts

LLMAgent: The main orchestrator. It manages the conversation with an LLM, handles tool calls, and invokes tool implementations.

Tool: Represents an external capability the LLM can use. It consists of:

name: A unique identifier.

description: A natural language description for the LLM to understand when to use the tool.

input_schema: A JSON schema defining the parameters the tool expects.

implementation: The Python function that executes the tool's logic.

LLMProvider: An abstraction for different LLM backends (e.g., AnthropicProvider). This allows the agent to be LLM-agnostic.

## Developing Custom Tools

Write a Python function that performs the desired action. This function will be the tool's implementation. It should accept arguments as defined in your tool's input_schema and return a string or a serializable dictionary.

Define an input_schema (JSON schema) that describes the parameters your function takes.

Create a Tool object providing the name, description, schema, and implementation function.

Register the Tool with your LLMAgent instance using agent.register_tool(your_tool).

## Contributing

Contributions are welcome! Please feel free to open an issue to discuss a new feature or bug, or submit a pull request.

(Consider adding a CONTRIBUTING.md file with more detailed guidelines if the project grows).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

**Before you commit these files:**

1.  **Replace placeholders in `LICENSE`**:
    *   `[Year]` with the current year.
    *   `[Your Full Name or Organization Name]` with your name or org name.
2.  **Review placeholders in `README.md`**:
    *   The build status badge URL (`yourusername/llm-tool-harness`) should be updated if you set up CI on GitHub Actions or another service.
    *   Check the `pip install` name if you decide to call your package something other than `llm-tool-harness` on PyPI.
    *   Update model names in the example if newer ones become standard (e.g., "claude-3-sonnet-20240229" is current but models evolve).
