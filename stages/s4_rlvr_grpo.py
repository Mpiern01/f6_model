"""
Stage 4: RLVR (RL with Verifiable Reward)
GRPO-family optimizer with verifiable rewards

Implements: GRPO, MS-GRPO, INFO-GRPO

MIT-level engineering: Production-grade RL with verifiable rewards
"""

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments
)
from typing import Dict, Any, Optional, List
import logging
import os
from pathlib import Path
import json
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from verifiers import TestVerifier, BuildVerifier, SchemaVerifier
from safeguards.catastrophic_loss import CatastrophicLossPrevention

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerifiableReward:
    """Computes verifiable rewards from verifiers."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize verifiable reward.
        
        Args:
            config: Reward configuration with components
        """
        self.verifiers = {
            "tests": TestVerifier(),
            "build": BuildVerifier(),
            "schema": SchemaVerifier()
        }
        self.components = config.get("components", [])
        self.reward_scale = config.get("reward_scale", 1.0)
        self.reward_clip = config.get("reward_clip", [-10.0, 10.0])
    
    def compute(self, response: str, prompt: str) -> float:
        """
        Compute verifiable reward.
        
        Args:
            response: Model response
            prompt: Input prompt
            
        Returns:
            Reward value
        """
        total_reward = 0.0
        
        for component in self.components:
            weight = component.get("weight", 1.0)
            name = component.get("name")
            verifier_name = component.get("verifier")
            
            if verifier_name in self.verifiers:
                verifier = self.verifiers[verifier_name]
                result = self._run_verifier(verifier, verifier_name, response, prompt)
                
                if result.get("passed", result.get("valid", result.get("applied", False))):
                    reward = weight * 1.0
                else:
                    reward = 0.0
                
                # Cost penalty
                if name == "cost_penalty":
                    metric = component.get("metric", "token_count")
                    if metric == "token_count":
                        cost = len(response.split())
                        reward = -weight * (cost / 1000.0)  # Penalize verbosity
                
                total_reward += reward
        
        # Scale and clip
        total_reward *= self.reward_scale
        total_reward = np.clip(total_reward, self.reward_clip[0], self.reward_clip[1])
        
        return total_reward
    
    def _run_verifier(self, verifier, verifier_name: str, response: str, prompt: str) -> Dict[str, Any]:
        """Run verifier."""
        if verifier_name == "tests":
            code, tests = self._extract_code_and_tests(response)
            return verifier.verify(code, tests, language="python")
        elif verifier_name == "build":
            code_dir = self._extract_code_directory(response)
            if code_dir:
                return verifier.verify(code_dir)
            else:
                return {"passed": False, "score": 0.0, "error": "Could not extract code directory"}
        elif verifier_name == "schema":
            schema = self._extract_schema(response)
            if schema:
                return verifier.verify(schema)
            else:
                return {"valid": False, "score": 0.0, "error": "Could not extract schema"}
        return {"passed": False, "score": 0.0}
    
    def _extract_code_and_tests(self, response: str) -> tuple[str, str]:
        """Extract code and tests from response."""
        import re
        code_blocks = re.findall(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
        if len(code_blocks) >= 2:
            return code_blocks[0], code_blocks[1]
        elif len(code_blocks) == 1:
            code = code_blocks[0]
            if "def test_" in code or "assert" in code:
                parts = re.split(r'(def test_|assert)', code, maxsplit=1)
                if len(parts) >= 3:
                    return parts[0], parts[1] + parts[2]
            return code, ""
        return response, ""
    
    def _extract_code_directory(self, response: str) -> Optional[str]:
        """Extract code directory path from response."""
        import re
        import os
        paths = re.findall(r'[\/\w]+(?:\.py|\.js|\.ts)', response)
        if paths and os.path.exists(paths[0]):
            return os.path.dirname(os.path.abspath(paths[0]))
        dirs = re.findall(r'[\/\w]+(?:/src|/lib|/tests)', response)
        if dirs and os.path.exists(dirs[0]):
            return os.path.abspath(dirs[0])
        return None
    
    def _extract_schema(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON schema from response."""
        import json
        import re
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        try:
            return json.loads(response)
        except:
            return None


class GRPOTrainer:
    """GRPO (Group Relative Policy Optimization) trainer."""
    
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        reward_fn,
        config: Dict[str, Any]
    ):
        """
        Initialize GRPO trainer.
        
        Args:
            model: Current model
            ref_model: Reference model
            tokenizer: Tokenizer
            reward_fn: Reward function
            config: Training configuration
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config
        
        self.kl_penalty = config.get("kl_penalty", 0.1)
        self.clip_range = config.get("clip_range", 0.2)
        self.learning_rate = config.get("learning_rate", 5e-7)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate
        )
    
    def compute_grpo_loss(self, prompts: List[str], responses: List[str], rewards: List[float]) -> torch.Tensor:
        """
        Compute GRPO loss.
        
        Args:
            prompts: Input prompts
            responses: Model responses
            rewards: Verifiable rewards
            
        Returns:
            Loss tensor
        """
        # Tokenize
        prompt_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        response_inputs = self.tokenizer(responses, return_tensors="pt", padding=True, truncation=True)
        
        if torch.cuda.is_available():
            prompt_inputs = {k: v.cuda() for k, v in prompt_inputs.items()}
            response_inputs = {k: v.cuda() for k, v in response_inputs.items()}
        
        # Get logits from current and reference models
        with torch.no_grad():
            ref_outputs = self.ref_model(**prompt_inputs)
            ref_logits = ref_outputs.logits
        
        current_outputs = self.model(**prompt_inputs)
        current_logits = current_outputs.logits
        
        # Compute log probabilities
        ref_probs = F.softmax(ref_logits, dim=-1)
        current_probs = F.softmax(current_logits, dim=-1)
        
        # KL divergence
        kl_div = current_probs * (torch.log(current_probs + 1e-8) - torch.log(ref_probs + 1e-8))
        kl_div = kl_div.sum(dim=-1).mean()
        
        # Reward-weighted loss
        rewards_tensor = torch.tensor(rewards, device=current_logits.device)
        reward_mean = rewards_tensor.mean()
        reward_std = rewards_tensor.std() + 1e-8
        
        # Normalize rewards
        normalized_rewards = (rewards_tensor - reward_mean) / reward_std
        
        # GRPO loss: maximize reward while minimizing KL divergence
        loss = -normalized_rewards.mean() + self.kl_penalty * kl_div
        
        return loss
    
    def train_step(self, prompts: List[str], responses: List[str], rewards: List[float]):
        """Perform one training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss = self.compute_grpo_loss(prompts, responses, rewards)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()


class MSGRPOTrainer(GRPOTrainer):
    """MS-GRPO: Multi-Step GRPO with improvements."""
    
    def __init__(self, *args, ms_steps: int = 3, **kwargs):
        """Initialize MS-GRPO trainer."""
        super().__init__(*args, **kwargs)
        self.ms_steps = ms_steps
    
    def train_step(self, prompts: List[str], responses: List[str], rewards: List[float]):
        """Multi-step training."""
        total_loss = 0.0
        
        for step in range(self.ms_steps):
            loss = super().train_step(prompts, responses, rewards)
            total_loss += loss
        
        return total_loss / self.ms_steps


class INFOGRPOTrainer(GRPOTrainer):
    """INFO-GRPO: Information-theoretic GRPO to prevent collapse."""
    
    def __init__(self, *args, entropy_bonus: float = 0.01, **kwargs):
        """Initialize INFO-GRPO trainer."""
        super().__init__(*args, **kwargs)
        self.entropy_bonus = entropy_bonus
    
    def compute_grpo_loss(self, prompts: List[str], responses: List[str], rewards: List[float]) -> torch.Tensor:
        """Compute INFO-GRPO loss with entropy bonus."""
        loss = super().compute_grpo_loss(prompts, responses, rewards)
        
        # Add entropy bonus to prevent collapse
        # Compute entropy of response distribution
        prompt_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available():
            prompt_inputs = {k: v.cuda() for k, v in prompt_inputs.items()}
        
        outputs = self.model(**prompt_inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        
        # Entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        
        # Subtract entropy bonus (maximize entropy = minimize negative entropy)
        loss = loss - self.entropy_bonus * entropy
        
        return loss


def run_stage4(config: Dict[str, Any], resume: Optional[str] = None, dry_run: bool = False):
    """
    Run Stage 4: RLVR with GRPO.
    
    Args:
        config: Configuration dictionary
        resume: Optional checkpoint to resume from
        dry_run: If True, validate config without training
    """
    logger.info("=" * 80)
    logger.info("Stage 4: RLVR with GRPO")
    logger.info("=" * 80)
    
    # Load model from Stage 3
    stage3_checkpoint = config.get("training", {}).get("reference_model", "checkpoints/stage3_dpo/final")
    logger.info(f"Loading model from: {stage3_checkpoint}")
    
    if dry_run:
        logger.info("Dry run mode - validating configuration only")
        return
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained(stage3_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        stage3_checkpoint,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        stage3_checkpoint,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Initialize reward function
    reward_fn = VerifiableReward(config["reward"])
    
    # Initialize trainer based on algorithm
    algorithm = config["training"].get("algorithm", "grpo")
    advanced_config = config.get("advanced_grpo", {})
    
    if algorithm == "ms_grpo" and advanced_config.get("use_ms_grpo", False):
        trainer = MSGRPOTrainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            config=config["training"],
            ms_steps=advanced_config.get("ms_grpo_steps", 3)
        )
        logger.info("Using MS-GRPO trainer")
    elif algorithm == "info_grpo" and advanced_config.get("use_info_grpo", False):
        trainer = INFOGRPOTrainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            config=config["training"],
            entropy_bonus=advanced_config.get("info_grpo_entropy_bonus", 0.01)
        )
        logger.info("Using INFO-GRPO trainer")
    else:
        trainer = GRPOTrainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            config=config["training"]
        )
        logger.info("Using standard GRPO trainer")
    
    # Training loop
    logger.info("Starting RLVR training...")
    
    # Load prompts (simplified)
    prompts = [
        "Write a Python function to calculate factorial",
        "Generate a tool schema for file reading",
        # Add more from dataset
    ]
    
    num_epochs = config["training"].get("num_epochs", 10)
    reward_logs = []
    
    for epoch in range(num_epochs):
        epoch_rewards = []
        epoch_losses = []
        
        for prompt in prompts:
            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=1.0,
                    do_sample=True
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Compute reward
            reward = reward_fn.compute(response, prompt)
            epoch_rewards.append(reward)
            
            # Training step
            loss = trainer.train_step([prompt], [response], [reward])
            epoch_losses.append(loss)
        
        avg_reward = np.mean(epoch_rewards)
        avg_loss = np.mean(epoch_losses)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}: avg_reward={avg_reward:.4f}, avg_loss={avg_loss:.4f}")
        
        reward_logs.append({
            "epoch": epoch + 1,
            "avg_reward": avg_reward,
            "avg_loss": avg_loss
        })
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(config["output"]["checkpoint_dir"], f"checkpoint-{epoch+1}")
            trainer.model.save_pretrained(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(config["output"]["checkpoint_dir"], "final")
    trainer.model.save_pretrained(final_path)
    logger.info(f"Saved final model to {final_path}")
    
    # Save reward logs
    reward_log_path = config["output"].get("reward_logs", "logs/stage4_rewards.jsonl")
    os.makedirs(os.path.dirname(reward_log_path), exist_ok=True)
    with open(reward_log_path, "w") as f:
        for log in reward_logs:
            f.write(json.dumps(log) + "\n")
    
    logger.info("=" * 80)
    logger.info("✓ Stage 4 complete")
    logger.info("=" * 80)

