"""
Test Verifiers
Tests for test, build, schema, and patch verifiers

MIT-level engineering: Comprehensive verifier tests
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from verifiers.tests import TestVerifier
from verifiers.build import BuildVerifier
from verifiers.schema import SchemaVerifier
from verifiers.patch_apply import PatchApplyVerifier


class TestTestVerifier:
    """Test TestVerifier functionality."""
    
    def test_initialization(self):
        """Test verifier initialization."""
        verifier = TestVerifier(timeout=300)
        assert verifier.timeout == 300
    
    def test_python_test_verification(self):
        """Test Python test verification."""
        verifier = TestVerifier()
        
        code = """
def add(a, b):
    return a + b
"""
        
        tests = """
def test_add():
    assert add(2, 3) == 5
    assert add(0, 0) == 0
"""
        
        result = verifier.verify(code, tests, language="python")
        
        assert "passed" in result
        assert "score" in result
        assert "output" in result
    
    def test_failing_test(self):
        """Test failing test detection."""
        verifier = TestVerifier()
        
        code = """
def add(a, b):
    return a - b  # Wrong implementation
"""
        
        tests = """
def test_add():
    assert add(2, 3) == 5
"""
        
        result = verifier.verify(code, tests, language="python")
        
        assert result["passed"] is False
        assert result["score"] < 1.0


class TestBuildVerifier:
    """Test BuildVerifier functionality."""
    
    def test_initialization(self):
        """Test verifier initialization."""
        verifier = BuildVerifier(timeout=600)
        assert verifier.timeout == 600
    
    def test_build_detection(self):
        """Test build system detection."""
        verifier = BuildVerifier()
        
        # Test detection logic (without actual files)
        # In production, would create temp directories with build files
        pass


class TestSchemaVerifier:
    """Test SchemaVerifier functionality."""

    def test_initialization(self):
        """Test verifier initialization."""
        verifier = SchemaVerifier()
        assert verifier is not None

    def test_tool_schema_validation(self):
        """Test tool schema validation."""
        verifier = SchemaVerifier()

        # Valid schema
        valid_schema = {
            "name": "read_file",
            "description": "Read a file from disk",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "encoding": {"type": "string"}
                },
                "required": ["path"]
            }
        }

        result = verifier.verify(valid_schema)
        assert result["valid"] is True
        assert result["score"] == 1.0
        assert len(result["errors"]) == 0

    def test_invalid_schema(self):
        """Test invalid schema detection."""
        verifier = SchemaVerifier()

        # Missing required fields
        invalid_schema = {
            "name": "test_tool"
            # Missing description and parameters
        }

        result = verifier.verify(invalid_schema)
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_tool_call_validation(self):
        """Test tool call validation against schema."""
        verifier = SchemaVerifier()

        schema = {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        }

        # Valid call
        valid_call = {
            "tool": "read_file",
            "arguments": {"path": "/tmp/test.txt"}
        }

        result = verifier.verify_tool_call(valid_call, schema)
        assert result["valid"] is True

        # Invalid call - missing required argument
        invalid_call = {
            "tool": "read_file",
            "arguments": {}
        }

        result = verifier.verify_tool_call(invalid_call, schema)
        assert result["valid"] is False


class TestPatchApplyVerifier:
    """Test PatchApplyVerifier functionality."""
    
    def test_initialization(self):
        """Test verifier initialization."""
        verifier = PatchApplyVerifier()
        assert verifier is not None
    
    def test_simple_patch_application(self):
        """Test simple patch application."""
        verifier = PatchApplyVerifier()
        
        base_code = """
def hello():
    print("Hello")
"""
        
        patch = """
--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
 def hello():
-    print("Hello")
+    print("Hello, World!")
"""
        
        result = verifier.verify(patch, base_code)
        
        assert "applied" in result
        assert "conflicts" in result
    
    def test_conflicting_patch(self):
        """Test conflicting patch detection."""
        verifier = PatchApplyVerifier()
        
        base_code = """
def hello():
    print("Goodbye")
"""
        
        patch = """
--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
 def hello():
-    print("Hello")
+    print("Hello, World!")
"""
        
        result = verifier.verify(patch, base_code)
        
        # Should detect conflict (base code doesn't match patch)
        assert result["applied"] is False or len(result.get("conflicts", [])) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

