"""
Test Verifier
Verifies: tests pass/fail

MIT-level engineering: Robust test execution with isolation
"""

import subprocess
import tempfile
import os
from typing import Dict, Any, List, Optional
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestVerifier:
    """Verifies test execution results."""
    
    def __init__(self, timeout: int = 300):
        """
        Initialize test verifier.
        
        Args:
            timeout: Test execution timeout in seconds
        """
        self.timeout = timeout
    
    def verify(self, code: str, tests: str, language: str = "python") -> Dict[str, Any]:
        """
        Verify tests pass for given code.
        
        Args:
            code: Code to test
            tests: Test code
            language: Programming language (default: "python")
            
        Returns:
            Verification result with:
                - passed: bool
                - score: float (0.0-1.0)
                - output: str
                - errors: List[str]
        """
        if language == "python":
            return self._verify_python(code, tests)
        elif language == "javascript":
            return self._verify_javascript(code, tests)
        else:
            logger.warning(f"Unsupported language: {language}, defaulting to Python")
            return self._verify_python(code, tests)
    
    def _verify_python(self, code: str, tests: str) -> Dict[str, Any]:
        """Verify Python tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code and tests
            code_path = os.path.join(tmpdir, "code.py")
            test_path = os.path.join(tmpdir, "test_code.py")
            
            with open(code_path, "w") as f:
                f.write(code)
            
            with open(test_path, "w") as f:
                f.write(tests)
            
            # Run tests with pytest
            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", test_path, "-v", "--tb=short"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                passed = result.returncode == 0
                output = result.stdout + result.stderr
                
                # Parse test results
                test_count = 0
                passed_count = 0
                
                for line in output.split("\n"):
                    if "passed" in line.lower():
                        test_count += 1
                        if "PASSED" in line or "passed" in line:
                            passed_count += 1
                
                score = passed_count / test_count if test_count > 0 else (1.0 if passed else 0.0)
                
                return {
                    "passed": passed,
                    "score": score,
                    "output": output,
                    "errors": [] if passed else [output],
                    "test_count": test_count,
                    "passed_count": passed_count
                }
                
            except subprocess.TimeoutExpired:
                return {
                    "passed": False,
                    "score": 0.0,
                    "output": "Test execution timed out",
                    "errors": ["Test execution exceeded timeout"],
                    "test_count": 0,
                    "passed_count": 0
                }
            except Exception as e:
                return {
                    "passed": False,
                    "score": 0.0,
                    "output": str(e),
                    "errors": [str(e)],
                    "test_count": 0,
                    "passed_count": 0
                }
    
    def _verify_javascript(self, code: str, tests: str) -> Dict[str, Any]:
        """Verify JavaScript tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code and tests
            code_path = os.path.join(tmpdir, "code.js")
            test_path = os.path.join(tmpdir, "test_code.js")
            
            with open(code_path, "w") as f:
                f.write(code)
            
            with open(test_path, "w") as f:
                f.write(tests)
            
            # Run tests with Node.js (assuming Jest or similar)
            try:
                result = subprocess.run(
                    ["node", test_path],
                    cwd=tmpdir,
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
                    "errors": [] if passed else [output]
                }
                
            except Exception as e:
                return {
                    "passed": False,
                    "score": 0.0,
                    "output": str(e),
                    "errors": [str(e)]
                }
    
    def verify_patch(self, patch: str, base_code: str, tests: str, language: str = "python") -> Dict[str, Any]:
        """
        Verify patch by applying it and running tests.
        
        Args:
            patch: Patch/diff to apply
            base_code: Base code before patch
            tests: Test code
            language: Programming language
            
        Returns:
            Verification result
        """
        # Apply patch (simplified - in production, use proper diff/patch tools)
        try:
            # For now, assume patch is the full new code
            if patch.startswith("---") or patch.startswith("diff"):
                # It's a diff, need to apply it
                # Simplified: just use patch as new code for now
                new_code = patch
            else:
                new_code = patch
            
            return self.verify(new_code, tests, language)
            
        except Exception as e:
            return {
                "passed": False,
                "score": 0.0,
                "output": f"Failed to apply patch: {e}",
                "errors": [str(e)]
            }

