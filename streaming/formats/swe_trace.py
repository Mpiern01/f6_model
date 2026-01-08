"""
SWE Trace Format
Format: (spec → plan → tool calls → patch → tests → final summary)

High-quality SWE trajectories for SFT
"""

from typing import Dict, Any, List, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SWETraceFormatter:
    """Formatter for SWE (Software Engineering) trajectories."""
    
    REQUIRED_FIELDS = ["spec", "plan", "tool_calls", "patch", "tests", "summary"]
    
    def __init__(self):
        """Initialize SWE trace formatter."""
        pass
    
    def format_trace(self, trace: Dict[str, Any]) -> str:
        """
        Format SWE trace into training text.
        
        Args:
            trace: SWE trace with required fields
            
        Returns:
            Formatted training text
        """
        # Validate required fields
        missing = [field for field in self.REQUIRED_FIELDS if field not in trace]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        
        formatted = f"""<spec>
{trace['spec']}
</spec>

<plan>
{trace['plan']}
</plan>

<tool_calls>
{self._format_tool_calls(trace['tool_calls'])}
</tool_calls>

<patch>
{trace['patch']}
</patch>

<tests>
{trace['tests']}
</tests>

<summary>
{trace['summary']}
</summary>"""
        
        return formatted
    
    def _format_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> str:
        """Format tool calls list."""
        if isinstance(tool_calls, str):
            return tool_calls
        
        formatted_calls = []
        for i, call in enumerate(tool_calls, 1):
            tool_name = call.get("tool", call.get("name", "unknown"))
            arguments = call.get("arguments", call.get("args", {}))
            
            if isinstance(arguments, dict):
                args_str = json.dumps(arguments, indent=2)
            else:
                args_str = str(arguments)
            
            result = call.get("result", call.get("output", ""))
            
            formatted = f"""Tool Call {i}:
  Tool: {tool_name}
  Arguments:
{args_str}
  Result:
{result}"""
            formatted_calls.append(formatted)
        
        return "\n\n".join(formatted_calls)
    
    def create_spec_prompt(self, spec: str) -> str:
        """
        Create prompt from specification.
        
        Args:
            spec: Task specification
            
        Returns:
            Prompt for model to generate plan
        """
        return f"""<spec>
{spec}
</spec>

<plan>
"""
    
    def create_plan_prompt(self, spec: str, plan: str) -> str:
        """
        Create prompt for tool calling given plan.
        
        Args:
            spec: Task specification
            plan: Execution plan
            
        Returns:
            Prompt for model to generate tool calls
        """
        return f"""<spec>
{spec}
</spec>

<plan>
{plan}
</plan>

<tool_calls>
"""
    
    def create_patch_prompt(self, spec: str, tool_results: List[Dict[str, Any]]) -> str:
        """
        Create prompt for patch generation given tool results.
        
        Args:
            spec: Task specification
            tool_results: Results from tool calls
            
        Returns:
            Prompt for model to generate patch
        """
        results_str = "\n".join([
            f"Tool: {r.get('tool', 'unknown')}\nResult: {r.get('result', '')}"
            for r in tool_results
        ])
        
        return f"""<spec>
{spec}
</spec>

<tool_results>
{results_str}
</tool_results>

<patch>
"""
    
    def create_test_prompt(self, spec: str, patch: str) -> str:
        """
        Create prompt for test generation given patch.
        
        Args:
            spec: Task specification
            patch: Generated patch
            
        Returns:
            Prompt for model to generate tests
        """
        return f"""<spec>
{spec}
</spec>

<patch>
{patch}
</patch>

<tests>
"""
    
    def create_summary_prompt(self, spec: str, patch: str, tests: str, test_results: Dict[str, Any]) -> str:
        """
        Create prompt for final summary.
        
        Args:
            spec: Task specification
            patch: Generated patch
            tests: Generated tests
            test_results: Test execution results
            
        Returns:
            Prompt for model to generate summary
        """
        test_status = "PASS" if test_results.get("all_passed", False) else "FAIL"
        
        return f"""<spec>
{spec}
</spec>

<patch>
{patch}
</patch>

<tests>
{tests}
</tests>

<test_results>
Status: {test_status}
{json.dumps(test_results, indent=2)}
</test_results>

<summary>
"""


def validate_swe_trace(trace: Dict[str, Any]) -> bool:
    """
    Validate SWE trace has all required fields.
    
    Args:
        trace: SWE trace to validate
        
    Returns:
        True if valid, False otherwise
    """
    formatter = SWETraceFormatter()
    missing = [field for field in formatter.REQUIRED_FIELDS if field not in trace]
    
    if missing:
        logger.warning(f"Invalid SWE trace: missing fields {missing}")
        return False
    
    return True

