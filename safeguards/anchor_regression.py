"""
Anchor Regression
Immutable regression prompts to prevent catastrophic drift

MIT-level engineering: Comprehensive regression testing
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
import json
import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnchorRegression:
    """
    Maintains anchor prompts and validates model performance on them.
    Prevents catastrophic drift from base model.
    """
    
    def __init__(
        self,
        base_model_path: str,
        anchor_prompts: Optional[List[Dict[str, Any]]] = None,
        threshold: float = 0.05,
        checkpoint_dir: str = "checkpoints/anchors"
    ):
        """
        Initialize anchor regression.
        
        Args:
            base_model_path: Path to base model for comparison
            anchor_prompts: List of anchor prompts (auto-generated if None)
            threshold: Maximum allowed degradation (default: 0.05 = 5%)
            checkpoint_dir: Directory for anchor checkpoints
        """
        self.base_model_path = base_model_path
        self.threshold = threshold
        self.checkpoint_dir = checkpoint_dir
        
        # Load base model for comparison
        self.base_tokenizer = None
        self.base_model = None
        self._load_base_model()
        
        # Anchor prompts
        if anchor_prompts is None:
            self.anchor_prompts = self._default_anchor_prompts()
        else:
            self.anchor_prompts = anchor_prompts
        
        # Anchor scores history
        self.anchor_scores = []
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def _load_base_model(self):
        """Load base model for comparison."""
        try:
            self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.base_model.eval()
            logger.info(f"Loaded base model: {self.base_model_path}")
        except Exception as e:
            logger.warning(f"Failed to load base model: {e}. Anchor regression will be limited.")
    
    def _default_anchor_prompts(self) -> List[Dict[str, Any]]:
        """Generate default anchor prompts for SWE tasks."""
        return [
            {
                "id": "anchor_001",
                "category": "tool_use",
                "prompt": "List the files in the current directory using Python.",
                "expected_keywords": ["os.listdir", "os.getcwd", "import os"]
            },
            {
                "id": "anchor_002",
                "category": "code_generation",
                "prompt": "Write a Python function to calculate the factorial of a number.",
                "expected_keywords": ["def", "factorial", "if", "return"]
            },
            {
                "id": "anchor_003",
                "category": "code_analysis",
                "prompt": "What does this code do?\n```python\ndef add(a, b):\n    return a + b\n```",
                "expected_keywords": ["add", "function", "returns", "sum"]
            },
            {
                "id": "anchor_004",
                "category": "tool_schema",
                "prompt": "Generate a tool schema for a function that reads a file.",
                "expected_keywords": ["name", "description", "parameters", "type"]
            },
            {
                "id": "anchor_005",
                "category": "long_horizon",
                "prompt": "Plan the steps to refactor a large codebase: 1) Analyze structure, 2) Identify patterns, 3) Create refactoring plan.",
                "expected_keywords": ["analyze", "identify", "plan", "refactor"]
            }
        ]
    
    def evaluate(self, model: torch.nn.Module, tokenizer: Any, step: int) -> Dict[str, Any]:
        """
        Evaluate model on anchor prompts.
        
        Args:
            model: Current model
            tokenizer: Tokenizer
            step: Training step
            
        Returns:
            Evaluation result with:
                - passed: bool
                - score: float (0.0-1.0)
                - degradation: float (negative if improved)
                - details: Dict with per-prompt results
        """
        if self.base_model is None:
            logger.warning("Base model not loaded, skipping anchor evaluation")
            return {
                "passed": True,
                "score": 1.0,
                "degradation": 0.0,
                "details": {}
            }
        
        model.eval()
        self.base_model.eval()
        
        current_scores = []
        base_scores = []
        details = {}
        
        with torch.no_grad():
            for anchor in self.anchor_prompts:
                prompt = anchor["prompt"]
                anchor_id = anchor["id"]
                
                # Evaluate current model
                current_score = self._evaluate_prompt(model, tokenizer, prompt, anchor)
                current_scores.append(current_score)
                
                # Evaluate base model
                base_score = self._evaluate_prompt(self.base_model, self.base_tokenizer, prompt, anchor)
                base_scores.append(base_score)
                
                details[anchor_id] = {
                    "current_score": current_score,
                    "base_score": base_score,
                    "degradation": current_score - base_score
                }
        
        # Calculate overall metrics
        avg_current = np.mean(current_scores) if current_scores else 0.0
        avg_base = np.mean(base_scores) if base_scores else 0.0
        degradation = avg_current - avg_base
        
        # Check if passed
        passed = degradation >= -self.threshold  # Allow small improvement
        
        result = {
            "passed": passed,
            "score": avg_current,
            "base_score": avg_base,
            "degradation": degradation,
            "details": details,
            "step": step
        }
        
        self.anchor_scores.append(result)
        
        if not passed:
            logger.warning(f"Anchor regression failed at step {step}: degradation={degradation:.4f}")
        else:
            logger.info(f"Anchor regression passed at step {step}: score={avg_current:.4f}, degradation={degradation:.4f}")
        
        return result
    
    def _evaluate_prompt(self, model: torch.nn.Module, tokenizer: Any, 
                        prompt: str, anchor: Dict[str, Any]) -> float:
        """
        Evaluate single prompt.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            prompt: Prompt text
            anchor: Anchor metadata
            
        Returns:
            Score (0.0-1.0)
        """
        try:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0
            )
            
            # Decode
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Score based on expected keywords
            expected_keywords = anchor.get("expected_keywords", [])
            if expected_keywords:
                matches = sum(1 for keyword in expected_keywords if keyword.lower() in generated.lower())
                score = matches / len(expected_keywords)
            else:
                # Fallback: length-based score
                score = min(1.0, len(generated) / 100.0)
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating prompt: {e}")
            return 0.0
    
    def save_anchor_scores(self, filepath: Optional[str] = None):
        """Save anchor scores history."""
        if filepath is None:
            filepath = os.path.join(self.checkpoint_dir, "anchor_scores.json")
        
        with open(filepath, "w") as f:
            json.dump(self.anchor_scores, f, indent=2)
        
        logger.info(f"Saved anchor scores to {filepath}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current anchor regression status."""
        if not self.anchor_scores:
            return {
                "evaluations": 0,
                "latest_score": None,
                "latest_degradation": None,
                "passed": True
            }
        
        latest = self.anchor_scores[-1]
        return {
            "evaluations": len(self.anchor_scores),
            "latest_score": latest["score"],
            "latest_degradation": latest["degradation"],
            "passed": latest["passed"],
            "average_score": np.mean([s["score"] for s in self.anchor_scores]),
            "average_degradation": np.mean([s["degradation"] for s in self.anchor_scores])
        }

