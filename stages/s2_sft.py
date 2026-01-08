"""
Stage 2: Supervised Fine-Tuning (SFT)
Format: (spec → plan → tool calls → patch → tests → final summary)

High-quality SWE trajectories for deterministic behavior and tool schemas.

MIT-level engineering: Production-grade SFT with variant generation
"""

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, Any, Optional, List
import logging
import os
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming.hf_stream import StreamingDataLoader
from streaming.formats.swe_trace import SWETraceFormatter, validate_swe_trace
from safeguards.catastrophic_loss import CatastrophicLossPrevention
from safeguards.anchor_regression import AnchorRegression
from safeguards.gradient_safety import GradientSafety

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SFTDataset(IterableDataset):
    """Iterable dataset for SFT with SWE trajectories."""
    
    def __init__(self, data_source, tokenizer, max_length: int = 8192, variant: str = "instruct"):
        """
        Initialize SFT dataset.
        
        Args:
            data_source: Data source (streamer or list)
            tokenizer: Tokenizer
            max_length: Maximum sequence length
            variant: "instruct" or "thinking"
        """
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.variant = variant
        self.formatter = SWETraceFormatter()
    
    def __iter__(self):
        """Iterate over SWE trajectories."""
        for sample in self.data_source:
            # Validate SWE trace
            if not validate_swe_trace(sample):
                continue
            
            # Format based on variant
            if self.variant == "instruct":
                # Concise, policy-compliant format
                text = self._format_instruct(sample)
            else:  # thinking
                # Deeper reasoning traces
                text = self._format_thinking(sample)
            
            # Tokenize
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Create labels
            encoded["labels"] = encoded["input_ids"].clone()
            
            yield {k: v.squeeze(0) for k, v in encoded.items()}
    
    def _format_instruct(self, sample: Dict[str, Any]) -> str:
        """Format for Instruct variant (concise)."""
        return self.formatter.format_trace(sample)
    
    def _format_thinking(self, sample: Dict[str, Any]) -> str:
        """Format for Thinking variant (deeper reasoning)."""
        base = self.formatter.format_trace(sample)
        
        # Add reasoning traces
        reasoning = f"""
<reasoning>
Let me think through this step by step:

1. Understanding the spec: {sample.get('spec', '')[:100]}...
2. Planning approach: {sample.get('plan', '')[:200]}...
3. Tool selection rationale: Why these tools? What alternatives exist?
4. Patch generation: How does this solve the problem? Edge cases?
5. Test design: What scenarios need coverage?
6. Final validation: Does this meet all requirements?
</reasoning>

{base}
"""
        return reasoning


class SFTTrainer(Trainer):
    """Custom trainer for SFT."""
    
    def __init__(self, gradient_safety: GradientSafety, *args, **kwargs):
        """Initialize SFT trainer."""
        super().__init__(*args, **kwargs)
        self.gradient_safety = gradient_safety
    
    def training_step(self, model, inputs):
        """Override training step to add gradient safety."""
        loss = super().training_step(model, inputs)
        self.gradient_safety.clip_gradients(model)
        return loss


class SFTCallback(TrainerCallback):
    """Callback for SFT monitoring."""
    
    def __init__(self, catastrophic_loss: CatastrophicLossPrevention):
        """Initialize SFT callback."""
        self.catastrophic_loss = catastrophic_loss
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Monitor loss."""
        if logs and "loss" in logs:
            loss_check = self.catastrophic_loss.check_loss(logs["loss"], state.global_step)
            if loss_check["alert"]:
                logger.warning(f"Loss alert: {loss_check['reason']}")


def run_stage2(config: Dict[str, Any], resume: Optional[str] = None, dry_run: bool = False):
    """
    Run Stage 2: Supervised Fine-Tuning.
    
    Args:
        config: Configuration dictionary
        resume: Optional checkpoint to resume from
        dry_run: If True, validate config without training
    """
    logger.info("=" * 80)
    logger.info("Stage 2: Supervised Fine-Tuning (SFT)")
    logger.info("=" * 80)
    
    # Load model from Stage 1 or base
    stage1_checkpoint = config.get("stage1_checkpoint", "checkpoints/stage1_midtrain/final")
    base_model_path = config["model"]["base_model"]
    
    model_path = stage1_checkpoint if os.path.exists(stage1_checkpoint) else base_model_path
    logger.info(f"Loading model from: {model_path}")
    
    if dry_run:
        logger.info("Dry run mode - validating configuration only")
        return
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model (with LoRA if exists)
    training_config = config["training"]
    if training_config.get("use_lora", True):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        # Apply LoRA if not already applied
        if not hasattr(model, "peft_config"):
            lora_config = LoraConfig(
                r=training_config.get("lora_r", 16),
                lora_alpha=training_config.get("lora_alpha", 32),
                target_modules=training_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                lora_dropout=training_config.get("lora_dropout", 0.05),
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            model = get_peft_model(model, lora_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    # Initialize safeguards
    catastrophic_loss = CatastrophicLossPrevention(
        loss_spike_threshold=config["safeguards"].get("loss_spike_threshold", 2.0),
        checkpoint_dir="checkpoints/safeguards"
    )
    
    gradient_safety = GradientSafety(
        max_grad_norm=config["training"].get("max_grad_norm", 1.0)
    )
    
    # Setup data
    data_config = config["data"]
    datasets = data_config.get("datasets", [])
    
    # Train both variants
    variants = config["output"].get("variants", ["instruct", "thinking"])
    
    for variant in variants:
        logger.info(f"\nTraining {variant} variant...")
        
        # Create dataset streamer
        dataset_configs = []
        for dataset_cfg in datasets:
            loader = StreamingDataLoader(
                dataset_name=dataset_cfg["name"],
                split=dataset_cfg.get("split", "train"),
                streaming=True
            )
            dataset_configs.append(loader)
        
        # Create dataset
        dataset = SFTDataset(
            data_source=dataset_configs[0].stream() if dataset_configs else [],
            tokenizer=tokenizer,
            max_length=config["training"].get("max_length", 8192),
            variant=variant
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(config["output"]["checkpoint_dir"], variant),
            num_train_epochs=config["training"].get("num_epochs", 5),
            per_device_train_batch_size=training_config.get("per_device_batch_size", 1),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
            learning_rate=config["training"].get("learning_rate", 5e-6),
            warmup_steps=config["training"].get("warmup_steps", 200),
            logging_steps=50,
            save_steps=500,
            save_total_limit=3,
            gradient_checkpointing=True,
            fp16=True,
            dataloader_num_workers=0,
        )
        
        # Initialize trainer
        trainer = SFTTrainer(
            gradient_safety=gradient_safety,
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            callbacks=[SFTCallback(catastrophic_loss)]
        )
        
        # Train
        if resume:
            trainer.train(resume_from_checkpoint=resume)
        else:
            trainer.train()
        
        # Save variant
        variant_path = os.path.join(config["output"]["checkpoint_dir"], variant, "final")
        trainer.save_model(variant_path)
        logger.info(f"Saved {variant} variant to {variant_path}")
    
    logger.info("=" * 80)
    logger.info("✓ Stage 2 complete")
    logger.info("=" * 80)

