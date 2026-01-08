"""
Tool Trace Format
Format: Tool calling sequences with MCP/function calling

For InfTool closed-loop synthesis and tool-use training
"""

from typing import Dict, Any, List, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolTraceFormatter:
    """Formatter for tool calling traces."""
    
    def __init__(self):
        """Initialize tool trace formatter."""
        pass
    
    def format_trace(self, trace: Dict[str, Any]) -> str:
        """
        Format tool trace into training text.
        
        Args:
            trace: Tool trace with tool calls and results
            
        Returns:
            Formatted training text
        """
        user_query = trace.get("user_query", trace.get("query", ""))
        tool_calls = trace.get("tool_calls", trace.get("calls", []))
        final_response = trace.get("response", trace.get("final_response", ""))
        
        formatted = f"""<user_query>
{user_query}
</user_query>

<tool_calls>
{self._format_tool_calls(tool_calls)}
</tool_calls>

<response>
{final_response}
</response>"""
        
        return formatted
    
    def _format_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> str:
        """Format tool calls list."""
        if not tool_calls:
            return "No tool calls required."
        
        if isinstance(tool_calls, str):
            return tool_calls
        
        formatted_calls = []
        for i, call in enumerate(tool_calls, 1):
            tool_name = call.get("tool", call.get("name", "unknown"))
            tool_id = call.get("id", call.get("call_id", f"call_{i}"))
            arguments = call.get("arguments", call.get("args", {}))
            
            if isinstance(arguments, dict):
                args_str = json.dumps(arguments, indent=2)
            else:
                args_str = str(arguments)
            
            result = call.get("result", call.get("output", call.get("response", "")))
            
            formatted = f"""Tool Call {i} (ID: {tool_id}):
  Tool: {tool_name}
  Arguments:
{args_str}
  Result:
{result}"""
            formatted_calls.append(formatted)
        
        return "\n\n".join(formatted_calls)
    
    def create_mcp_tool_schema(self, tools: List[Dict[str, Any]]) -> str:
        """
        Create MCP tool schema from tool definitions.
        
        Args:
            tools: List of tool definitions
            
        Returns:
            Formatted tool schema
        """
        schema_parts = ["<tool_schema>"]
        
        for tool in tools:
            name = tool.get("name", "unknown")
            description = tool.get("description", "")
            parameters = tool.get("parameters", tool.get("inputSchema", {}))
            
            schema_parts.append(f"""
Tool: {name}
Description: {description}
Parameters:
{json.dumps(parameters, indent=2)}
""")
        
        schema_parts.append("</tool_schema>")
        return "\n".join(schema_parts)


def create_tool_call_prompt(user_query: str, available_tools: List[Dict[str, Any]]) -> str:
    """
    Create prompt for tool calling.
    
    Args:
        user_query: User's query/request
        available_tools: List of available tools with schemas
        
    Returns:
        Prompt for model to generate tool calls
    """
    formatter = ToolTraceFormatter()
    tool_schema = formatter.create_mcp_tool_schema(available_tools)
    
    return f"""<user_query>
{user_query}
</user_query>

{tool_schema}

<tool_calls>
"""

