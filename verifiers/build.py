"""
Build Verifier
Verifies: build/lint pass/fail

MIT-level engineering: Comprehensive build verification
"""

import subprocess
import tempfile
import os
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BuildVerifier:
    """Verifies build and lint results."""
    
    def __init__(self, timeout: int = 600):
        """
        Initialize build verifier.
        
        Args:
            timeout: Build execution timeout in seconds
        """
        self.timeout = timeout
    
    def verify_build(self, code_dir: str, build_command: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify build passes.
        
        Args:
            code_dir: Directory containing code to build
            build_command: Custom build command (auto-detected if None)
            
        Returns:
            Verification result with:
                - passed: bool
                - score: float (0.0-1.0)
                - output: str
                - errors: List[str]
        """
        # Auto-detect build system
        if build_command is None:
            build_command = self._detect_build_command(code_dir)
        
        if build_command is None:
            return {
                "passed": False,
                "score": 0.0,
                "output": "Could not detect build system",
                "errors": ["No build command found"]
            }
        
        try:
            result = subprocess.run(
                build_command,
                shell=True,
                cwd=code_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            passed = result.returncode == 0
            output = result.stdout + result.stderr
            
            return {
                "passed": passed,
                "score": 1.0 if passed else 0.0,
                "output": output,
                "errors": [] if passed else [output],
                "build_command": build_command
            }
            
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "score": 0.0,
                "output": "Build execution timed out",
                "errors": ["Build exceeded timeout"]
            }
        except Exception as e:
            return {
                "passed": False,
                "score": 0.0,
                "output": str(e),
                "errors": [str(e)]
            }
    
    def verify_lint(self, code_dir: str, lint_command: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify lint passes.
        
        Args:
            code_dir: Directory containing code to lint
            lint_command: Custom lint command (auto-detected if None)
            
        Returns:
            Verification result
        """
        # Auto-detect linter
        if lint_command is None:
            lint_command = self._detect_lint_command(code_dir)
        
        if lint_command is None:
            return {
                "passed": True,  # No linter = pass (optional check)
                "score": 1.0,
                "output": "No linter configured",
                "errors": []
            }
        
        try:
            result = subprocess.run(
                lint_command,
                shell=True,
                cwd=code_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Linters often return non-zero for warnings, so check output
            output = result.stdout + result.stderr
            has_errors = "error" in output.lower() or "failed" in output.lower()
            passed = result.returncode == 0 and not has_errors
            
            return {
                "passed": passed,
                "score": 1.0 if passed else 0.0,
                "output": output,
                "errors": [] if passed else [output]
            }
            
        except Exception as e:
            return {
                "passed": False,
                "score": 0.0,
                "output": str(e),
                "errors": [str(e)]
            }
    
    def _detect_build_command(self, code_dir: str) -> Optional[str]:
        """Detect build command from project structure."""
        # Check for common build files
        if os.path.exists(os.path.join(code_dir, "setup.py")):
            return "python setup.py build"
        elif os.path.exists(os.path.join(code_dir, "pyproject.toml")):
            return "pip install -e ."
        elif os.path.exists(os.path.join(code_dir, "package.json")):
            return "npm run build"
        elif os.path.exists(os.path.join(code_dir, "Makefile")):
            return "make"
        elif os.path.exists(os.path.join(code_dir, "CMakeLists.txt")):
            return "cmake --build ."
        elif os.path.exists(os.path.join(code_dir, "Cargo.toml")):
            return "cargo build"
        else:
            return None
    
    def _detect_lint_command(self, code_dir: str) -> Optional[str]:
        """Detect lint command from project structure."""
        # Check for common linters
        if os.path.exists(os.path.join(code_dir, "pyproject.toml")):
            # Check for ruff, black, etc.
            if os.path.exists(os.path.join(code_dir, ".ruff.toml")):
                return "ruff check ."
            elif os.path.exists(os.path.join(code_dir, ".flake8")):
                return "flake8 ."
            else:
                return "ruff check ."  # Default to ruff
        elif os.path.exists(os.path.join(code_dir, "package.json")):
            return "npm run lint"
        else:
            return None
    
    def verify_build_and_lint(self, code_dir: str) -> Dict[str, Any]:
        """
        Verify both build and lint.
        
        Args:
            code_dir: Directory containing code
            
        Returns:
            Combined verification result
        """
        build_result = self.verify_build(code_dir)
        lint_result = self.verify_lint(code_dir)
        
        passed = build_result["passed"] and lint_result["passed"]
        score = (build_result["score"] + lint_result["score"]) / 2.0
        
        return {
            "passed": passed,
            "score": score,
            "build": build_result,
            "lint": lint_result,
            "output": f"Build: {build_result['output']}\nLint: {lint_result['output']}",
            "errors": build_result["errors"] + lint_result["errors"]
        }

