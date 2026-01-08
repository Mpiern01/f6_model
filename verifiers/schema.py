"""
Schema Verifier
Verifies: tool schema validity

MIT-level engineering: Comprehensive schema validation
"""

import json
from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SchemaVerifier:
    """Verifies tool schema validity."""
    
    REQUIRED_SCHEMA_FIELDS = ["name", "description", "parameters"]
    REQUIRED_PARAMETER_FIELDS = ["type"]
    
    def __init__(self):
        """Initialize schema verifier."""
        pass
    
    def verify(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify tool schema is valid.
        
        Args:
            schema: Tool schema dictionary or list of schemas
            
        Returns:
            Verification result with:
                - valid: bool
                - score: float (0.0-1.0)
                - errors: List[str]
        """
        if isinstance(schema, list):
            return self._verify_multiple(schema)
        else:
            return self._verify_single(schema)
    
    def _verify_single(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Verify single tool schema."""
        errors = []
        
        # Check required fields
        for field in self.REQUIRED_SCHEMA_FIELDS:
            if field not in schema:
                errors.append(f"Missing required field: {field}")
        
        # Validate name
        if "name" in schema:
            name = schema["name"]
            if not isinstance(name, str) or not name.strip():
                errors.append("Schema name must be a non-empty string")
            if not name.replace("_", "").replace("-", "").isalnum():
                errors.append("Schema name must be alphanumeric (with _ or -)")
        
        # Validate description
        if "description" in schema:
            if not isinstance(schema["description"], str):
                errors.append("Description must be a string")
        
        # Validate parameters
        if "parameters" in schema:
            param_errors = self._validate_parameters(schema["parameters"])
            errors.extend(param_errors)
        
        # Check for JSON Schema compliance (if using JSON Schema format)
        if "parameters" in schema and isinstance(schema["parameters"], dict):
            if "type" in schema["parameters"]:
                # It's a JSON Schema
                json_schema_errors = self._validate_json_schema(schema["parameters"])
                errors.extend(json_schema_errors)
        
        valid = len(errors) == 0
        score = 1.0 if valid else max(0.0, 1.0 - len(errors) * 0.2)
        
        return {
            "valid": valid,
            "score": score,
            "errors": errors,
            "schema": schema
        }
    
    def _verify_multiple(self, schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify multiple tool schemas."""
        all_errors = []
        valid_count = 0
        
        for i, schema in enumerate(schemas):
            result = self._verify_single(schema)
            if not result["valid"]:
                all_errors.append(f"Schema {i} ({schema.get('name', 'unknown')}): {result['errors']}")
            else:
                valid_count += 1
        
        valid = len(all_errors) == 0
        score = valid_count / len(schemas) if schemas else 0.0
        
        return {
            "valid": valid,
            "score": score,
            "errors": all_errors,
            "valid_count": valid_count,
            "total_count": len(schemas)
        }
    
    def _validate_parameters(self, parameters: Any) -> List[str]:
        """Validate parameters field."""
        errors = []
        
        if isinstance(parameters, dict):
            # JSON Schema format
            if "type" not in parameters:
                errors.append("Parameters must have 'type' field")
            elif parameters["type"] not in ["object", "string", "number", "integer", "boolean", "array"]:
                errors.append(f"Invalid parameter type: {parameters['type']}")
            
            # Validate properties if type is object
            if parameters.get("type") == "object" and "properties" in parameters:
                props = parameters["properties"]
                if not isinstance(props, dict):
                    errors.append("Properties must be a dictionary")
                else:
                    for prop_name, prop_schema in props.items():
                        if not isinstance(prop_schema, dict):
                            errors.append(f"Property '{prop_name}' schema must be a dictionary")
                        elif "type" not in prop_schema:
                            errors.append(f"Property '{prop_name}' must have 'type' field")
        
        elif isinstance(parameters, list):
            # List of parameters
            for i, param in enumerate(parameters):
                if not isinstance(param, dict):
                    errors.append(f"Parameter {i} must be a dictionary")
                elif "name" not in param:
                    errors.append(f"Parameter {i} must have 'name' field")
                elif "type" not in param:
                    errors.append(f"Parameter {i} must have 'type' field")
        
        else:
            errors.append("Parameters must be a dictionary or list")
        
        return errors
    
    def _validate_json_schema(self, schema: Dict[str, Any]) -> List[str]:
        """Validate JSON Schema format."""
        errors = []
        
        # Basic JSON Schema validation
        valid_types = ["string", "number", "integer", "boolean", "array", "object", "null"]
        if "type" in schema and schema["type"] not in valid_types:
            errors.append(f"Invalid JSON Schema type: {schema['type']}")
        
        # Validate required fields if present
        if "required" in schema:
            if not isinstance(schema["required"], list):
                errors.append("'required' must be a list")
            elif "properties" in schema:
                props = schema["properties"]
                for req_field in schema["required"]:
                    if req_field not in props:
                        errors.append(f"Required field '{req_field}' not in properties")
        
        return errors
    
    def verify_tool_call(self, tool_call: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify tool call matches schema.
        
        Args:
            tool_call: Tool call with 'tool' and 'arguments' fields
            schema: Tool schema
            
        Returns:
            Verification result
        """
        errors = []
        
        # Check tool name matches
        call_tool = tool_call.get("tool", tool_call.get("name"))
        schema_name = schema.get("name")
        
        if call_tool != schema_name:
            errors.append(f"Tool name mismatch: {call_tool} != {schema_name}")
        
        # Validate arguments
        if "arguments" in tool_call:
            args = tool_call["arguments"]
            if "parameters" in schema:
                param_errors = self._validate_arguments_against_schema(args, schema["parameters"])
                errors.extend(param_errors)
        
        valid = len(errors) == 0
        score = 1.0 if valid else max(0.0, 1.0 - len(errors) * 0.3)
        
        return {
            "valid": valid,
            "score": score,
            "errors": errors
        }
    
    def _validate_arguments_against_schema(self, arguments: Dict[str, Any], parameters: Dict[str, Any]) -> List[str]:
        """Validate arguments against parameter schema."""
        errors = []
        
        if isinstance(parameters, dict) and parameters.get("type") == "object":
            if "properties" in parameters:
                props = parameters["properties"]
                required = parameters.get("required", [])
                
                # Check required fields
                for req_field in required:
                    if req_field not in arguments:
                        errors.append(f"Missing required argument: {req_field}")
                
                # Check argument types
                for arg_name, arg_value in arguments.items():
                    if arg_name in props:
                        prop_schema = props[arg_name]
                        expected_type = prop_schema.get("type")
                        actual_type = self._get_value_type(arg_value)
                        
                        if expected_type and not self._type_matches(actual_type, expected_type):
                            errors.append(f"Argument '{arg_name}' type mismatch: {actual_type} != {expected_type}")
        
        return errors
    
    def _get_value_type(self, value: Any) -> str:
        """Get JSON Schema type for value."""
        if isinstance(value, str):
            return "string"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "null"
    
    def _type_matches(self, actual: str, expected: str) -> bool:
        """Check if actual type matches expected type."""
        # Integer is a subset of number
        if expected == "number" and actual == "integer":
            return True
        return actual == expected

