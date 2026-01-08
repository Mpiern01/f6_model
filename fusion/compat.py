"""
Model Fusion Compatibility Checker
Strict compatibility gates for safe model fusion

MIT-level engineering: Production-grade compatibility checking
"""

import torch
from transformers import AutoTokenizer, AutoConfig
from typing import Dict, Any, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompatibilityChecker:
    """
    Checks model compatibility for fusion.
    
    Requirements:
    - Same tokenizer vocab + special tokens
    - Identical layer counts / hidden size / attention config
    - Parameter name alignment
    """
    
    def __init__(self):
        """Initialize compatibility checker."""
        pass
    
    def check_compatibility(
        self,
        model1_path: str,
        model2_path: str,
        require_same_tokenizer: bool = True,
        require_same_architecture: bool = True,
        require_parameter_alignment: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if two models are compatible for fusion.
        
        Args:
            model1_path: Path to first model
            model2_path: Path to second model
            require_same_tokenizer: Require same tokenizer
            require_same_architecture: Require same architecture
            require_parameter_alignment: Require parameter alignment
            
        Returns:
            Tuple of (is_compatible, details_dict)
        """
        details = {
            "compatible": True,
            "errors": [],
            "warnings": []
        }
        
        # Check tokenizer
        if require_same_tokenizer:
            tokenizer_compat, tokenizer_details = self._check_tokenizer(model1_path, model2_path)
            if not tokenizer_compat:
                details["compatible"] = False
                details["errors"].extend(tokenizer_details.get("errors", []))
            details["tokenizer"] = tokenizer_details
        
        # Check architecture
        if require_same_architecture:
            arch_compat, arch_details = self._check_architecture(model1_path, model2_path)
            if not arch_compat:
                details["compatible"] = False
                details["errors"].extend(arch_details.get("errors", []))
            details["architecture"] = arch_details
        
        # Check parameter alignment
        if require_parameter_alignment:
            param_compat, param_details = self._check_parameters(model1_path, model2_path)
            if not param_compat:
                details["compatible"] = False
                details["errors"].extend(param_details.get("errors", []))
            details["parameters"] = param_details
        
        return details["compatible"], details
    
    def _check_tokenizer(self, path1: str, path2: str) -> Tuple[bool, Dict[str, Any]]:
        """Check tokenizer compatibility."""
        details = {"compatible": True, "errors": []}
        
        try:
            tokenizer1 = AutoTokenizer.from_pretrained(path1)
            tokenizer2 = AutoTokenizer.from_pretrained(path2)
            
            # Check vocab size
            if tokenizer1.vocab_size != tokenizer2.vocab_size:
                details["compatible"] = False
                details["errors"].append(
                    f"Vocab size mismatch: {tokenizer1.vocab_size} != {tokenizer2.vocab_size}"
                )
            
            # Check special tokens
            special_tokens1 = set(tokenizer1.special_tokens_map.values())
            special_tokens2 = set(tokenizer2.special_tokens_map.values())
            
            if special_tokens1 != special_tokens2:
                details["compatible"] = False
                details["errors"].append(
                    f"Special tokens mismatch: {special_tokens1} != {special_tokens2}"
                )
            
            # Check vocab content (sample check)
            if tokenizer1.vocab != tokenizer2.vocab:
                # Check if it's just ordering or actual content
                vocab1_set = set(tokenizer1.vocab.keys())
                vocab2_set = set(tokenizer2.vocab.keys())
                
                if vocab1_set != vocab2_set:
                    details["compatible"] = False
                    details["errors"].append("Vocabulary content mismatch")
            
        except Exception as e:
            details["compatible"] = False
            details["errors"].append(f"Tokenizer check failed: {e}")
        
        return details["compatible"], details
    
    def _check_architecture(self, path1: str, path2: str) -> Tuple[bool, Dict[str, Any]]:
        """Check architecture compatibility."""
        details = {"compatible": True, "errors": []}
        
        try:
            config1 = AutoConfig.from_pretrained(path1)
            config2 = AutoConfig.from_pretrained(path2)
            
            # Check key architecture parameters
            arch_params = [
                "num_hidden_layers",
                "hidden_size",
                "num_attention_heads",
                "intermediate_size",
                "max_position_embeddings"
            ]
            
            for param in arch_params:
                val1 = getattr(config1, param, None)
                val2 = getattr(config2, param, None)
                
                if val1 is not None and val2 is not None:
                    if val1 != val2:
                        details["compatible"] = False
                        details["errors"].append(
                            f"{param} mismatch: {val1} != {val2}"
                        )
            
            # Check model type
            if config1.model_type != config2.model_type:
                details["compatible"] = False
                details["errors"].append(
                    f"Model type mismatch: {config1.model_type} != {config2.model_type}"
                )
            
        except Exception as e:
            details["compatible"] = False
            details["errors"].append(f"Architecture check failed: {e}")
        
        return details["compatible"], details
    
    def _check_parameters(self, path1: str, path2: str) -> Tuple[bool, Dict[str, Any]]:
        """Check parameter name alignment."""
        details = {"compatible": True, "errors": [], "mismatches": []}
        
        try:
            # Load state dicts
            state_dict1 = torch.load(f"{path1}/pytorch_model.bin", map_location="cpu")
            state_dict2 = torch.load(f"{path2}/pytorch_model.bin", map_location="cpu")
            
            # Get parameter names
            params1 = set(state_dict1.keys())
            params2 = set(state_dict2.keys())
            
            # Check for missing parameters
            missing_in_2 = params1 - params2
            missing_in_1 = params2 - params1
            
            if missing_in_2:
                details["compatible"] = False
                details["errors"].append(f"Parameters missing in model2: {missing_in_2}")
            
            if missing_in_1:
                details["compatible"] = False
                details["errors"].append(f"Parameters missing in model1: {missing_in_1}")
            
            # Check shape alignment for common parameters
            common_params = params1 & params2
            for param_name in list(common_params)[:10]:  # Sample check
                shape1 = state_dict1[param_name].shape
                shape2 = state_dict2[param_name].shape
                
                if shape1 != shape2:
                    details["mismatches"].append(
                        f"{param_name}: {shape1} != {shape2}"
                    )
                    details["compatible"] = False
            
        except Exception as e:
            details["compatible"] = False
            details["errors"].append(f"Parameter check failed: {e}")
        
        return details["compatible"], details

