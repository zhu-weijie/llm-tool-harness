# llm_tool_harness/tool.py
from typing import Callable, Dict, Any, List, TypedDict, Optional


class ToolInputSchema(TypedDict):
    type: str
    properties: Dict[str, Any]
    required: Optional[List[str]]


class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: ToolInputSchema


class Tool:
    """Represents a tool that the LLM can use."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: ToolInputSchema,
        implementation: Callable,
    ):
        """
        Args:
            name: The name of the tool.
            description: A description of what the tool does.
            input_schema: A JSON schema describing the tool's input.
            implementation: A callable function that executes the tool.
                            It should accept arguments as defined in input_schema's properties
                            and return a string or a structured dictionary as output.
        """
        if not callable(implementation):
            raise ValueError("Tool implementation must be a callable function.")
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.implementation = implementation

    def get_definition(self) -> ToolDefinition:
        """Returns the tool definition in the format expected by LLM APIs."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def execute(self, **kwargs: Any) -> Any:
        """Executes the tool's implementation with the given arguments."""
        # Future: Add validation against input_schema here
        return self.implementation(**kwargs)


class ToolRegistry:
    """Manages a collection of tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        if tool.name in self._tools:
            raise ValueError(f"Tool with name '{tool.name}' already registered.")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_all_tool_definitions(self) -> List[ToolDefinition]:
        return [tool.get_definition() for tool in self._tools.values()]

    def __bool__(self):
        return bool(self._tools)
