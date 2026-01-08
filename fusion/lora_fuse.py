"""
LoRA Fusion
Safe, rollback-capable LoRA adapter merging

MIT-level engineering: Production-grade LoRA fusion
"""

import torch
from peft import PeftModel
from typing import Dict, Any, List, Optional
import logging
import os
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRAFusion:
    """
    Safe LoRA fusion with rollback capability.
    """
    
    def __init__(self, base_model_path: str):
        """
        Initialize LoRA fusion.
        
        Args:
            base_model_path: Path to base model
        """
        self.base_model_path = base_model_path
        self.backup_path = None
    
    def fuse(
        self,
        lora_paths: List[str],
        weights: Optional[List[float]] = None,
        output_path: str = "fused_model",
        create_backup: bool = True
    ) -> Dict[str, Any]:
        """
        Fuse multiple LoRA adapters.
        
        Args:
            lora_paths: List of LoRA adapter paths
            weights: Optional weights for each adapter (uniform if None)
            output_path: Output path for fused model
            create_backup: Whether to create backup
            
        Returns:
            Fusion result dictionary
        """
        logger.info(f"Fusing {len(lora_paths)} LoRA adapters...")
        
        if weights is None:
            weights = [1.0 / len(lora_paths)] * len(lora_paths)
        
        if len(weights) != len(lora_paths):
            raise ValueError("Number of weights must match number of LoRA paths")
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Create backup
        if create_backup:
            self.backup_path = f"{output_path}_backup"
            if os.path.exists(output_path):
                shutil.copytree(output_path, self.backup_path)
                logger.info(f"Created backup: {self.backup_path}")
        
        try:
            # Load base model
            from transformers import AutoModelForCausalLM
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_path)
            
            # Load and merge LoRA adapters
            merged_state_dict = {}
            
            for lora_path, weight in zip(lora_paths, weights):
                logger.info(f"Loading LoRA: {lora_path} (weight={weight:.3f})")
                
                # Load LoRA adapter
                peft_model = PeftModel.from_pretrained(base_model, lora_path)
                lora_state_dict = peft_model.get_peft_state_dict()
                
                # Merge with weighted sum
                for key, value in lora_state_dict.items():
                    if key in merged_state_dict:
                        merged_state_dict[key] += weight * value
                    else:
                        merged_state_dict[key] = weight * value
            
            # Apply merged adapters to base model
            base_model.load_state_dict(merged_state_dict, strict=False)
            
            # Save fused model
            os.makedirs(output_path, exist_ok=True)
            base_model.save_pretrained(output_path)
            
            logger.info(f"✓ LoRA fusion complete: {output_path}")
            
            return {
                "success": True,
                "output_path": output_path,
                "backup_path": self.backup_path,
                "num_adapters": len(lora_paths),
                "weights": weights
            }
            
        except Exception as e:
            logger.error(f"LoRA fusion failed: {e}")
            
            # Rollback if backup exists
            if self.backup_path and os.path.exists(self.backup_path):
                logger.info("Rolling back to backup...")
                if os.path.exists(output_path):
                    shutil.rmtree(output_path)
                shutil.copytree(self.backup_path, output_path)
                logger.info("Rollback complete")
            
            return {
                "success": False,
                "error": str(e),
                "backup_path": self.backup_path
            }
    
    def rollback(self, output_path: str):
        """
        Rollback to backup.
        
        Args:
            output_path: Path to rollback
        """
        if self.backup_path and os.path.exists(self.backup_path):
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            shutil.copytree(self.backup_path, output_path)
            logger.info(f"Rolled back: {output_path}")
        else:
            logger.warning("No backup found for rollback")

