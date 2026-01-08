"""
CommitPack Code-Flow Format
Format: <commit_before>code_before<commit_message>commit_message<commit_after>code_after

IQuest-style code-flow behaviors:
- commit-conditioned patching
- "why this change?" commit-message reasoning
- edit → test → fix loops
- long-context repo reasoning
"""

from typing import Dict, Any, Iterator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_commitpack_sample(sample: Dict[str, Any]) -> str:
    """
    Format CommitPack sample into training text.
    
    Args:
        sample: Raw CommitPack sample
        
    Returns:
        Formatted training text
    """
    commit_before = sample.get("commit_before", "")
    commit_message = sample.get("commit_message", "")
    commit_after = sample.get("commit_after", "")
    
    # Format: IQuest-style code-flow
    formatted = f"""<commit_before>
{commit_before}
</commit_before>

<commit_message>
{commit_message}
</commit_message>

<commit_after>
{commit_after}
</commit_after>

<reasoning>
Analyze the code change:
1. What was changed and why?
2. What problem does this solve?
3. Are there edge cases to consider?
4. How does this fit into the broader codebase?
</reasoning>"""
    
    return formatted


def extract_codeflow_features(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract code-flow features for training.
    
    Args:
        sample: Raw CommitPack sample
        
    Returns:
        Features dictionary
    """
    features = {
        "commit_before": sample.get("commit_before", ""),
        "commit_after": sample.get("commit_after", ""),
        "commit_message": sample.get("commit_message", ""),
        "language": sample.get("language", "unknown"),
        "repo_name": sample.get("repo_name", ""),
    }
    
    # Compute diffs
    before_lines = features["commit_before"].split("\n")
    after_lines = features["commit_after"].split("\n")
    
    features["lines_added"] = len(after_lines) - len(before_lines)
    features["has_test_changes"] = "test" in features["commit_message"].lower()
    features["has_fix"] = "fix" in features["commit_message"].lower()
    
    return features


def create_codeflow_prompt(sample: Dict[str, Any], task_type: str = "patch") -> str:
    """
    Create code-flow prompt for different task types.
    
    Args:
        sample: CommitPack sample
        task_type: One of "patch", "reason", "test", "review"
        
    Returns:
        Formatted prompt
    """
    commit_before = sample.get("commit_before", "")
    commit_message = sample.get("commit_message", "")
    
    if task_type == "patch":
        # Task: Generate the patch
        prompt = f"""Given the code before and the commit message, generate the code after.

<code_before>
{commit_before}
</code_before>

<commit_message>
{commit_message}
</commit_message>

<code_after>
"""
        
    elif task_type == "reason":
        # Task: Explain why this change was made
        prompt = f"""Explain why this code change was made.

<code_before>
{commit_before}
</code_before>

<code_after>
{sample.get("commit_after", "")}
</code_after>

<explanation>
"""
        
    elif task_type == "test":
        # Task: Generate tests for the change
        prompt = f"""Generate tests for this code change.

<code_before>
{commit_before}
</code_before>

<code_after>
{sample.get("commit_after", "")}
</code_after>

<tests>
"""
        
    elif task_type == "review":
        # Task: Review the change
        prompt = f"""Review this code change for correctness, style, and potential issues.

<code_before>
{commit_before}
</code_before>

<code_after>
{sample.get("commit_after", "")}
</code_after>

<review>
"""
    
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    return prompt

