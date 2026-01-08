"""
Gradient Safety
Gradient clipping and anomaly detection

MIT-level engineering: Production-grade gradient handling
"""

import torch
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradientSafety:
    """
    Ensures gradient safety through:
    - Gradient clipping
    - Gradient norm monitoring
    - NaN/Inf detection
    """
    
    def __init__(
        self,
        max_grad_norm: float = 1.0,
        clip_mode: str = "norm"
    ):
        """
        Initialize gradient safety.
        
        Args:
            max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
            clip_mode: Clipping mode: "norm" or "value" (default: "norm")
        """
        self.max_grad_norm = max_grad_norm
        self.clip_mode = clip_mode
        
        if clip_mode not in ["norm", "value"]:
            raise ValueError(f"Invalid clip_mode: {clip_mode}. Must be 'norm' or 'value'")
    
    def clip_gradients(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Clip gradients to prevent explosion.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with:
                - clipped: bool
                - original_norm: float
                - clipped_norm: float
                - nan_detected: bool
                - inf_detected: bool
        """
        # Check for NaN/Inf first
        nan_detected = False
        inf_detected = False
        
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_detected = True
                if torch.isinf(param.grad).any():
                    inf_detected = True
        
        if nan_detected or inf_detected:
            logger.warning(f"NaN ({nan_detected}) or Inf ({inf_detected}) detected in gradients")
            # Zero out NaN/Inf gradients
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = torch.where(
                        torch.isnan(param.grad) | torch.isinf(param.grad),
                        torch.zeros_like(param.grad),
                        param.grad
                    )
        
        # Compute gradient norm
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        original_norm = total_norm
        
        # Clip gradients
        clipped = False
        if self.clip_mode == "norm":
            if total_norm > self.max_grad_norm:
                clip_coef = self.max_grad_norm / (total_norm + 1e-6)
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.mul_(clip_coef)
                clipped = True
                total_norm = self.max_grad_norm
        elif self.clip_mode == "value":
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-self.max_grad_norm, self.max_grad_norm)
                    clipped = True
        
        if clipped:
            logger.debug(f"Gradients clipped: {original_norm:.4f} -> {total_norm:.4f}")
        
        return {
            "clipped": clipped,
            "original_norm": original_norm,
            "clipped_norm": total_norm,
            "nan_detected": nan_detected,
            "inf_detected": inf_detected
        }
    
    def check_gradient_health(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Check gradient health without clipping.
        
        Args:
            model: PyTorch model
            
        Returns:
            Health status
        """
        total_norm = 0.0
        param_count = 0
        nan_count = 0
        inf_count = 0
        zero_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                if torch.isnan(param.grad).any():
                    nan_count += 1
                if torch.isinf(param.grad).any():
                    inf_count += 1
                if param_norm.item() < 1e-8:
                    zero_count += 1
        
        total_norm = total_norm ** (1. / 2)
        
        # Health status
        healthy = (
            not nan_count and
            not inf_count and
            total_norm < self.max_grad_norm * 10 and
            total_norm > 1e-8  # Not all zeros
        )
        
        return {
            "healthy": healthy,
            "gradient_norm": total_norm,
            "param_count": param_count,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "zero_count": zero_count,
            "max_grad_norm": self.max_grad_norm
        }
    
    def update_max_grad_norm(self, new_max: float):
        """
        Update maximum gradient norm.
        
        Args:
            new_max: New maximum gradient norm
        """
        self.max_grad_norm = new_max
        logger.info(f"Updated max_grad_norm to {new_max}")

