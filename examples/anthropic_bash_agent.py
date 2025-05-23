# examples/anthropic_bash_agent.py
import os
import subprocess

# Ensure the llm_tool_harness is in PYTHONPATH or installed
# For local development: export PYTHONPATH=$PYTHONPATH:$(pwd)/.. (if in examples dir)
from llm_tool_harness import LLMAgent, Tool, AnthropicProvider, ToolInputSchema
from dotenv import load_dotenv

load_dotenv()


# 1. Define your tool implementation
def execute_bash_command(command: str) -> str:
    """Execute a bash command and return a formatted string with the results."""
    print(f"Executing bash command: {command}")
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=30,  # Increased timeout
        )
        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nEXIT CODE: {result.returncode}"
        print(f"Bash output:\n{output}")
        return output
    except subprocess.TimeoutExpired:
        error_msg = f"Error: Command '{command}' timed out after 30 seconds."
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error executing command '{command}': {str(e)}"
        print(error_msg)
        return error_msg


def main():
    # Ensure ANTHROPIC_API_KEY is set in your environment
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        return

    # 2. Create an LLM provider instance
    # model="claude-3-opus-20240229" or "claude-3-sonnet-20240229" or "claude-3-haiku-20240307" etc.
    # The original script used "claude-3-7-sonnet-latest", use the closest official mapping
    anthropic_llm = AnthropicProvider(
        model="claude-3-sonnet-20240229", max_tokens=2000
    )  # or opus/haiku

    # 3. Create the LLM Agent
    agent = LLMAgent(
        llm_provider=anthropic_llm,
        system_prompt="""You are a helpful AI assistant with access to bash commands.
        You can help the user by executing commands and interpreting the results.
        Be careful with destructive commands and always explain what you're doing.
        You have access to the 'bash' tool which allows you to run shell commands.""",
    )

    # 4. Define the tool schema and create a Tool object
    bash_input_schema: ToolInputSchema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to execute"}
        },
        "required": ["command"],
    }
    bash_tool_obj = Tool(
        name="bash",
        description="Execute bash commands and return the output (stdout, stderr, exit code).",
        input_schema=bash_input_schema,
        implementation=execute_bash_command,
    )

    # 5. Register the tool with the agent
    agent.register_tool(bash_tool_obj)

    # 6. Start the chat loop
    agent.chat_loop()


if __name__ == "__main__":
    main()
