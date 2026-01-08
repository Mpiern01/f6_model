"""
Stage 3: Rollout-DPO (Preference Optimization)
Generate K rollouts per prompt, label winners/losers by verifiers

MIT-level engineering: Production-grade DPO with verifier-based preferences
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments
)
from trl import DPOTrainer, DPOConfig
from typing import Dict, Any, Optional, List
import logging
import os
from pathlib import Path
import json
import random

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from verifiers import TestVerifier, BuildVerifier, SchemaVerifier, PatchApplyVerifier
from safeguards.catastrophic_loss import CatastrophicLossPrevention

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerifierBasedPreferenceLabeler:
    """Labels rollouts as winners/losers based on verifiers."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preference labeler.
        
        Args:
            config: DPO configuration with verifiers and preference criteria
        """
        self.verifiers = {
            "tests": TestVerifier(),
            "build": BuildVerifier(),
            "schema": SchemaVerifier(),
            "patch_apply": PatchApplyVerifier()
        }
        
        self.preference_criteria = config.get("preference_criteria", [])
    
    def label_rollouts(self, prompt: str, rollouts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Label rollouts based on verifiers.
        
        Args:
            prompt: Input prompt
            rollouts: List of rollouts with "response" field
            
        Returns:
            List of labeled rollouts with "score" field
        """
        labeled = []
        
        for rollout in rollouts:
            response = rollout.get("response", "")
            score = self._compute_score(response, prompt)
            
            labeled.append({
                **rollout,
                "score": score,
                "verifier_results": self._get_verifier_results(response, prompt)
            })
        
        # Sort by score (highest first)
        labeled.sort(key=lambda x: x["score"], reverse=True)
        
        return labeled
    
    def _compute_score(self, response: str, prompt: str) -> float:
        """Compute preference score based on verifiers."""
        total_score = 0.0
        total_weight = 0.0
        
        for criterion in self.preference_criteria:
            weight = criterion.get("weight", 1.0)
            verifier_name = criterion.get("verifier")
            
            if verifier_name in self.verifiers:
                verifier_result = self._run_verifier(verifier_name, response, prompt)
                score = verifier_result.get("score", 0.0) if verifier_result.get("passed", False) else 0.0
                
                total_score += weight * score
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _run_verifier(self, verifier_name: str, response: str, prompt: str) -> Dict[str, Any]:
        """Run specific verifier."""
        verifier = self.verifiers[verifier_name]
        
        if verifier_name == "tests":
            # Extract code and tests from response
            code, tests = self._extract_code_and_tests(response)
            return verifier.verify(code, tests, language="python")
        elif verifier_name == "build":
            # Extract code directory from response
            code_dir = self._extract_code_directory(response)
            if code_dir:
                return verifier.verify(code_dir)
            else:
                return {"passed": False, "score": 0.0, "error": "Could not extract code directory"}
        elif verifier_name == "schema":
            # Extract tool schema from response
            schema = self._extract_schema(response)
            if schema:
                return verifier.verify(schema)
            else:
                return {"valid": False, "score": 0.0, "error": "Could not extract schema"}
        elif verifier_name == "patch_apply":
            # Extract patch and base code from response
            patch, base_code = self._extract_patch_and_base(response)
            if patch and base_code:
                return verifier.verify(patch, base_code)
            else:
                return {"applied": False, "score": 0.0, "error": "Could not extract patch"}
        
        return {"passed": False, "score": 0.0}
    
    def _extract_code_and_tests(self, response: str) -> tuple[str, str]:
        """Extract code and tests from response."""
        import re
        # Look for code blocks
        code_blocks = re.findall(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
        if len(code_blocks) >= 2:
            return code_blocks[0], code_blocks[1]
        elif len(code_blocks) == 1:
            # Try to split by test markers
            code = code_blocks[0]
            if "def test_" in code or "assert" in code:
                # Split at first test function
                parts = re.split(r'(def test_|assert)', code, maxsplit=1)
                if len(parts) >= 3:
                    return parts[0], parts[1] + parts[2]
            return code, ""
        else:
            # No code blocks, return response as code
            return response, ""
    
    def _extract_code_directory(self, response: str) -> Optional[str]:
        """Extract code directory path from response."""
        import re
        import os
        # Look for file paths
        paths = re.findall(r'[\/\w]+(?:\.py|\.js|\.ts|\.java|\.cpp|\.c)', response)
        if paths:
            # Get directory of first file
            first_path = paths[0]
            if os.path.exists(first_path):
                return os.path.dirname(os.path.abspath(first_path))
        # Look for directory mentions
        dirs = re.findall(r'[\/\w]+(?:/src|/lib|/tests|/code)', response)
        if dirs and os.path.exists(dirs[0]):
            return os.path.abspath(dirs[0])
        return None
    
    def _extract_schema(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON schema from response."""
        import json
        import re
        # Try to find JSON block
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        # Try parsing entire response as JSON
        try:
            return json.loads(response)
        except:
            return None
    
    def _extract_patch_and_base(self, response: str) -> tuple[Optional[str], Optional[str]]:
        """Extract patch and base code from response."""
        import re
        # Look for diff format
        diff_match = re.search(r'```diff\n(.*?)```', response, re.DOTALL)
        if diff_match:
            patch = diff_match.group(1)
            # Try to extract base code from context
            base_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
            base_code = base_match.group(1) if base_match else None
            return patch, base_code
        # Look for unified diff
        unified_match = re.search(r'---.*?\n\+\+\+.*?\n(@@.*?@@.*?)(?=\n---|\Z)', response, re.DOTALL)
        if unified_match:
            return unified_match.group(1), None
        return None, None
    
    def _get_verifier_results(self, response: str, prompt: str) -> Dict[str, Any]:
        """Get all verifier results."""
        results = {}
        for name, verifier in self.verifiers.items():
            results[name] = self._run_verifier(name, response, prompt)
        return results


class RolloutGenerator:
    """Generates multiple rollouts for a prompt."""
    
    def __init__(self, model, tokenizer, config: Dict[str, Any]):
        """
        Initialize rollout generator.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            config: Rollout configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def generate_rollouts(self, prompt: str, num_rollouts: int = 4) -> List[Dict[str, Any]]:
        """
        Generate multiple rollouts for a prompt.
        
        Args:
            prompt: Input prompt
            num_rollouts: Number of rollouts to generate
            
        Returns:
            List of rollouts
        """
        rollouts = []
        
        rollout_config = self.config.get("rollout_config", {})
        
        for i in range(num_rollouts):
            # Generate with sampling
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=rollout_config.get("max_new_tokens", 2048),
                    temperature=rollout_config.get("temperature", 1.0),
                    top_p=rollout_config.get("top_p", 0.95),
                    top_k=rollout_config.get("top_k", 20),
                    do_sample=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            rollouts.append({
                "prompt": prompt,
                "response": response,
                "rollout_id": i
            })
        
        return rollouts


def run_stage3(config: Dict[str, Any], resume: Optional[str] = None, dry_run: bool = False):
    """
    Run Stage 3: Rollout-DPO.
    
    Args:
        config: Configuration dictionary
        resume: Optional checkpoint to resume from
        dry_run: If True, validate config without training
    """
    logger.info("=" * 80)
    logger.info("Stage 3: Rollout-DPO (Preference Optimization)")
    logger.info("=" * 80)
    
    # Load model from Stage 2
    stage2_checkpoint = config.get("training", {}).get("reference_model", "checkpoints/stage2_sft/instruct/final")
    logger.info(f"Loading model from: {stage2_checkpoint}")
    
    if dry_run:
        logger.info("Dry run mode - validating configuration only")
        return
    
    # Load tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(stage2_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load current and reference models
    model = AutoModelForCausalLM.from_pretrained(
        stage2_checkpoint,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        stage2_checkpoint,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Initialize preference labeler
    preference_labeler = VerifierBasedPreferenceLabeler(config["data"])
    
    # Generate preference dataset
    logger.info("Generating preference dataset...")
    
    # Load prompts (simplified - in production, load from dataset)
    prompts = [
        "Write a Python function to calculate factorial",
        "Generate a tool schema for file reading",
        "Fix this bug: [code snippet]",
        # Add more prompts from dataset
    ]
    
    preference_pairs = []
    rollout_generator = RolloutGenerator(model, tokenizer, config["data"])
    
    for prompt in prompts:
        # Generate rollouts
        rollouts = rollout_generator.generate_rollouts(
            prompt,
            num_rollouts=config["training"].get("num_rollouts_per_prompt", 4)
        )
        
        # Label rollouts
        labeled = preference_labeler.label_rollouts(prompt, rollouts)
        
        # Create preference pairs (winner vs loser)
        if len(labeled) >= 2:
            winner = labeled[0]
            loser = labeled[-1]
            
            preference_pairs.append({
                "prompt": prompt,
                "chosen": winner["response"],
                "rejected": loser["response"]
            })
    
    logger.info(f"Generated {len(preference_pairs)} preference pairs")
    
    # Save preference pairs
    preference_log_path = config["output"].get("preference_logs", "logs/stage3_preferences.jsonl")
    os.makedirs(os.path.dirname(preference_log_path), exist_ok=True)
    
    with open(preference_log_path, "w") as f:
        for pair in preference_pairs:
            f.write(json.dumps(pair) + "\n")
    
    # DPO Training
    dpo_config = DPOConfig(
        output_dir=config["output"]["checkpoint_dir"],
        num_train_epochs=config["training"].get("num_epochs", 3),
        per_device_train_batch_size=config["training"].get("per_device_batch_size", 1),
        learning_rate=config["training"].get("learning_rate", 1e-6),
        beta=config["training"].get("beta", 0.1),
        logging_steps=50,
        save_steps=500,
        fp16=True,
    )
    
    # Create dataset from preference pairs
    # Simplified - in production, use proper dataset class
    class PreferenceDataset:
        def __init__(self, pairs, tokenizer):
            self.pairs = pairs
            self.tokenizer = tokenizer
        
        def __len__(self):
            return len(self.pairs)
        
        def __getitem__(self, idx):
            pair = self.pairs[idx]
            prompt = pair["prompt"]
            chosen = pair["chosen"]
            rejected = pair["rejected"]
            
            # Tokenize
            prompt_enc = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            chosen_enc = self.tokenizer(chosen, return_tensors="pt", padding="max_length", truncation=True, max_length=2048)
            rejected_enc = self.tokenizer(rejected, return_tensors="pt", padding="max_length", truncation=True, max_length=2048)
            
            return {
                "prompt_input_ids": prompt_enc["input_ids"].squeeze(0),
                "prompt_attention_mask": prompt_enc["attention_mask"].squeeze(0),
                "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
                "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
                "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
                "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
            }
    
    dataset = PreferenceDataset(preference_pairs, tokenizer)
    
    # Initialize DPO trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        beta=config["training"].get("beta", 0.1),
    )
    
    # Train
    logger.info("Starting DPO training...")
    dpo_trainer.train()
    
    # Save model
    final_path = os.path.join(config["output"]["checkpoint_dir"], "final")
    dpo_trainer.save_model(final_path)
    logger.info(f"Saved final model to {final_path}")
    
    logger.info("=" * 80)
    logger.info("✓ Stage 3 complete")
    logger.info("=" * 80)

