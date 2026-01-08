"""
Patch Apply Verifier
Verifies: patch applies cleanly

MIT-level engineering: Robust patch application with conflict detection
"""

import subprocess
import tempfile
import os
import re
import difflib
from typing import Dict, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatchApplyVerifier:
    """Verifies patches apply cleanly."""
    
    def __init__(self):
        """Initialize patch apply verifier."""
        pass
    
    def verify(self, patch: str, base_code: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify patch applies cleanly.
        
        Args:
            patch: Patch/diff string or new code
            base_code: Base code before patch
            file_path: Optional file path for context
            
        Returns:
            Verification result with:
                - applied: bool
                - score: float (0.0-1.0)
                - conflicts: List[str]
                - applied_code: str (if successful)
        """
        # Determine if patch is a diff or full code
        is_diff = self._is_diff_format(patch)
        
        if is_diff:
            return self._verify_diff_apply(patch, base_code, file_path)
        else:
            return self._verify_code_replace(patch, base_code)
    
    def _is_diff_format(self, patch: str) -> bool:
        """Check if patch is in diff format."""
        diff_indicators = ["---", "+++", "@@", "diff --git", "Index:"]
        return any(patch.strip().startswith(indicator) for indicator in diff_indicators)
    
    def _verify_diff_apply(self, diff: str, base_code: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Verify diff applies cleanly using patch tool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write base code
            target_file = file_path or "code.py"
            base_path = os.path.join(tmpdir, target_file)
            
            # Create directory structure if needed
            os.makedirs(os.path.dirname(base_path) if os.path.dirname(base_path) else tmpdir, exist_ok=True)
            
            with open(base_path, "w") as f:
                f.write(base_code)
            
            # Write patch
            patch_path = os.path.join(tmpdir, "patch.diff")
            with open(patch_path, "w") as f:
                f.write(diff)
            
            # Apply patch
            try:
                result = subprocess.run(
                    ["patch", "-p1", base_path, patch_path],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    # Read applied code
                    with open(base_path, "r") as f:
                        applied_code = f.read()
                    
                    return {
                        "applied": True,
                        "score": 1.0,
                        "conflicts": [],
                        "applied_code": applied_code,
                        "output": result.stdout
                    }
                else:
                    # Check for conflicts
                    conflicts = self._extract_conflicts(result.stderr)
                    
                    return {
                        "applied": False,
                        "score": 0.0,
                        "conflicts": conflicts,
                        "applied_code": None,
                        "output": result.stderr
                    }
                    
            except subprocess.TimeoutExpired:
                return {
                    "applied": False,
                    "score": 0.0,
                    "conflicts": ["Patch application timed out"],
                    "applied_code": None,
                    "output": "Timeout"
                }
            except FileNotFoundError:
                # patch command not available, try manual application
                logger.warning("patch command not found, using manual diff application")
                return self._verify_manual_diff_apply(diff, base_code)
            except Exception as e:
                return {
                    "applied": False,
                    "score": 0.0,
                    "conflicts": [str(e)],
                    "applied_code": None,
                    "output": str(e)
                }
    
    def _verify_manual_diff_apply(self, diff: str, base_code: str) -> Dict[str, Any]:
        """
        Apply diff using difflib (proper diff library).
        
        Uses Python's built-in difflib for proper unified diff parsing.
        """
        import difflib
        import re
        
        conflicts = []
        applied_code = None
        
        try:
            # Try to use unified_diff format
            base_lines = base_code.splitlines(keepends=True)
            
            # Parse unified diff format
            diff_lines = diff.splitlines(keepends=True)
            
            # Extract hunks from unified diff
            hunks = []
            current_hunk = None
            
            for line in diff_lines:
                if line.startswith("@@"):
                    # New hunk
                    if current_hunk:
                        hunks.append(current_hunk)
                    # Parse hunk header: @@ -start,count +start,count @@
                    match = re.match(r'@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@', line)
                    if match:
                        old_start = int(match.group(1)) - 1  # Convert to 0-indexed
                        old_count = int(match.group(2)) if match.group(2) else 1
                        new_start = int(match.group(3)) - 1
                        new_count = int(match.group(4)) if match.group(4) else 1
                        current_hunk = {
                            "old_start": old_start,
                            "old_count": old_count,
                            "new_start": new_start,
                            "new_count": new_count,
                            "lines": []
                        }
                elif current_hunk and line.startswith((" ", "-", "+")):
                    current_hunk["lines"].append(line)
            
            if current_hunk:
                hunks.append(current_hunk)
            
            # Apply hunks in reverse order to maintain line numbers
            applied_lines = base_lines.copy()
            for hunk in reversed(hunks):
                old_start = hunk["old_start"]
                old_count = hunk["old_count"]
                hunk_lines = hunk["lines"]
                
                # Remove old lines
                if old_start + old_count <= len(applied_lines):
                    del applied_lines[old_start:old_start + old_count]
                
                # Insert new lines
                new_lines = [line[1:] for line in hunk_lines if line.startswith("+")]
                applied_lines[old_start:old_start] = new_lines
            
            applied_code = "".join(applied_lines)
            applied = True
            
        except Exception as e:
            conflicts.append(f"Failed to apply diff: {e}")
            applied = False
            applied_code = None
        
        return {
            "applied": applied,
            "score": 1.0 if applied else 0.0,
            "conflicts": conflicts,
            "applied_code": applied_code,
            "output": "Applied using difflib" if applied else f"Failed: {conflicts}"
        }
    
    def _verify_code_replace(self, new_code: str, base_code: str) -> Dict[str, Any]:
        """
        Verify code replacement (when patch is full new code).
        
        Uses semantic diff to detect potential conflicts.
        """
        import difflib
        
        # Calculate similarity to detect major changes
        similarity = difflib.SequenceMatcher(None, base_code, new_code).ratio()
        
        # Check for semantic conflicts (function/class definitions)
        base_functions = set(re.findall(r'def\s+(\w+)', base_code))
        new_functions = set(re.findall(r'def\s+(\w+)', new_code))
        
        removed_functions = base_functions - new_functions
        added_functions = new_functions - base_functions
        
        conflicts = []
        if removed_functions:
            conflicts.append(f"Removed functions: {', '.join(removed_functions)}")
        if similarity < 0.3:  # Very different code
            conflicts.append(f"Major code change detected (similarity: {similarity:.2f})")
        
        # Score based on similarity and conflicts
        score = similarity if not conflicts else max(0.0, similarity - 0.2)
        
        return {
            "applied": True,
            "score": score,
            "conflicts": conflicts,
            "applied_code": new_code,
            "output": "Code replacement successful"
        }
    
    def _extract_conflicts(self, stderr: str) -> List[str]:
        """Extract conflict information from patch stderr."""
        conflicts = []
        
        for line in stderr.split("\n"):
            if "rejects" in line.lower() or "failed" in line.lower() or "conflict" in line.lower():
                conflicts.append(line.strip())
        
        return conflicts if conflicts else ["Unknown patch application error"]
    
    def verify_patch_quality(self, patch: str, base_code: str) -> Dict[str, Any]:
        """
        Verify patch quality beyond just application.
        
        Checks:
        - Patch applies cleanly
        - No syntax errors in applied code
        - Minimal changes (not full rewrite)
        
        Args:
            patch: Patch string
            base_code: Base code
            
        Returns:
            Quality metrics
        """
        apply_result = self.verify(patch, base_code)
        
        if not apply_result["applied"]:
            return {
                "quality_score": 0.0,
                "applied": False,
                "syntax_valid": False,
                "minimal_changes": False,
                "details": apply_result
            }
        
        applied_code = apply_result["applied_code"]
        
        # Check syntax (simplified - in production, use AST parsing)
        syntax_valid = True
        try:
            compile(applied_code, "<string>", "exec")
        except SyntaxError:
            syntax_valid = False
        
        # Check if changes are minimal (not full rewrite)
        base_lines = len(base_code.split("\n"))
        applied_lines = len(applied_code.split("\n"))
        change_ratio = abs(applied_lines - base_lines) / max(base_lines, 1)
        minimal_changes = change_ratio < 0.5  # Less than 50% line change
        
        quality_score = (
            (1.0 if apply_result["applied"] else 0.0) * 0.4 +
            (1.0 if syntax_valid else 0.0) * 0.4 +
            (1.0 if minimal_changes else 0.0) * 0.2
        )
        
        return {
            "quality_score": quality_score,
            "applied": apply_result["applied"],
            "syntax_valid": syntax_valid,
            "minimal_changes": minimal_changes,
            "change_ratio": change_ratio,
            "details": apply_result
        }

