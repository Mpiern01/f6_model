"""
Catastrophic Loss Prevention
Comprehensive safeguards against training failures

MIT-level engineering: Multi-layered protection
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Callable
import logging
from collections import deque
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CatastrophicLossPrevention:
    """
    Prevents catastrophic loss through multiple mechanisms:
    - Loss spike detection
    - Gradient anomaly detection
    - Model checkpoint validation
    - Automatic rollback
    """
    
    def __init__(
        self,
        loss_spike_threshold: float = 2.0,
        max_loss_value: float = 100.0,
        window_size: int = 100,
        anomaly_threshold: float = 10.0,
        checkpoint_dir: str = "checkpoints/safeguards"
    ):
        """
        Initialize catastrophic loss prevention.
        
        Args:
            loss_spike_threshold: Factor by which loss can spike before alert (default: 2.0)
            max_loss_value: Absolute maximum loss value (default: 100.0)
            window_size: Window size for loss trend analysis (default: 100)
            anomaly_threshold: Standard deviation threshold for anomalies (default: 10.0)
            checkpoint_dir: Directory for safeguard checkpoints
        """
        self.loss_spike_threshold = loss_spike_threshold
        self.max_loss_value = max_loss_value
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.checkpoint_dir = checkpoint_dir
        
        # Loss history
        self.loss_history = deque(maxlen=window_size)
        self.baseline_loss = None
        
        # Gradient history
        self.gradient_norms = deque(maxlen=window_size)
        
        # Alert tracking
        self.alerts = []
        self.rollback_count = 0
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def check_loss(self, loss_value: float, step: int) -> Dict[str, Any]:
        """
        Check loss value for anomalies.
        
        Args:
            loss_value: Current loss value
            step: Training step number
            
        Returns:
            Check result with:
                - safe: bool
                - alert: bool
                - action: str (one of: "continue", "warn", "stop", "rollback")
                - reason: str
        """
        # Update history
        self.loss_history.append(loss_value)
        
        # Check absolute maximum
        if loss_value > self.max_loss_value:
            return {
                "safe": False,
                "alert": True,
                "action": "stop",
                "reason": f"Loss exceeds maximum: {loss_value} > {self.max_loss_value}",
                "loss_value": loss_value,
                "step": step
            }
        
        # Check for NaN or Inf
        if not torch.isfinite(torch.tensor(loss_value)):
            return {
                "safe": False,
                "alert": True,
                "action": "rollback",
                "reason": f"Loss is NaN or Inf: {loss_value}",
                "loss_value": loss_value,
                "step": step
            }
        
        # Establish baseline
        if self.baseline_loss is None and len(self.loss_history) >= 10:
            self.baseline_loss = np.mean(list(self.loss_history)[:10])
        
        # Check for spike
        if self.baseline_loss is not None:
            spike_ratio = loss_value / self.baseline_loss
            
            if spike_ratio > self.loss_spike_threshold:
                return {
                    "safe": False,
                    "alert": True,
                    "action": "warn",
                    "reason": f"Loss spike detected: {spike_ratio:.2f}x baseline",
                    "loss_value": loss_value,
                    "baseline": self.baseline_loss,
                    "spike_ratio": spike_ratio,
                    "step": step
                }
        
        # Check for trend (increasing loss)
        if len(self.loss_history) >= self.window_size:
            recent_losses = list(self.loss_history)[-self.window_size:]
            trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            
            if trend > 0.1:  # Increasing trend
                return {
                    "safe": True,
                    "alert": True,
                    "action": "warn",
                    "reason": f"Increasing loss trend detected: {trend:.4f}",
                    "loss_value": loss_value,
                    "trend": trend,
                    "step": step
                }
        
        # Check for anomalies using statistical methods
        if len(self.loss_history) >= self.window_size:
            recent_losses = list(self.loss_history)[-self.window_size:]
            mean_loss = np.mean(recent_losses)
            std_loss = np.std(recent_losses)
            
            if std_loss > 0:
                z_score = abs((loss_value - mean_loss) / std_loss)
                
                if z_score > self.anomaly_threshold:
                    return {
                        "safe": False,
                        "alert": True,
                        "action": "warn",
                        "reason": f"Loss anomaly detected: z-score={z_score:.2f}",
                        "loss_value": loss_value,
                        "mean": mean_loss,
                        "std": std_loss,
                        "z_score": z_score,
                        "step": step
                    }
        
        # Update baseline if loss is stable
        if len(self.loss_history) >= 50:
            recent_mean = np.mean(list(self.loss_history)[-50:])
            if self.baseline_loss is None or abs(recent_mean - self.baseline_loss) / max(self.baseline_loss, 1e-6) < 0.1:
                self.baseline_loss = recent_mean
        
        return {
            "safe": True,
            "alert": False,
            "action": "continue",
            "reason": "Loss within normal range",
            "loss_value": loss_value,
            "step": step
        }
    
    def check_gradients(self, model: torch.nn.Module, step: int) -> Dict[str, Any]:
        """
        Check gradients for anomalies.
        
        Args:
            model: PyTorch model
            step: Training step number
            
        Returns:
            Check result
        """
        total_norm = 0.0
        param_count = 0
        nan_count = 0
        inf_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Check for NaN/Inf
                if torch.isnan(param.grad).any():
                    nan_count += 1
                if torch.isinf(param.grad).any():
                    inf_count += 1
        
        total_norm = total_norm ** (1. / 2)
        self.gradient_norms.append(total_norm)
        
        # Check for NaN/Inf gradients
        if nan_count > 0 or inf_count > 0:
            return {
                "safe": False,
                "alert": True,
                "action": "stop",
                "reason": f"Gradient contains NaN ({nan_count}) or Inf ({inf_count})",
                "gradient_norm": total_norm,
                "nan_count": nan_count,
                "inf_count": inf_count,
                "step": step
            }
        
        # Check gradient norm
        if total_norm > 100.0:  # Very large gradient
            return {
                "safe": False,
                "alert": True,
                "action": "warn",
                "reason": f"Gradient norm very large: {total_norm:.2f}",
                "gradient_norm": total_norm,
                "step": step
            }
        
        # Check for gradient explosion
        if len(self.gradient_norms) >= 10:
            recent_norms = list(self.gradient_norms)[-10:]
            if max(recent_norms) > 10 * min(recent_norms):
                return {
                    "safe": False,
                    "alert": True,
                    "action": "warn",
                    "reason": "Potential gradient explosion detected",
                    "gradient_norm": total_norm,
                    "step": step
                }
        
        return {
            "safe": True,
            "alert": False,
            "action": "continue",
            "reason": "Gradients normal",
            "gradient_norm": total_norm,
            "step": step
        }
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                       step: int, loss: float) -> str:
        """
        Save safeguard checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            step: Training step
            loss: Current loss
            
        Returns:
            Checkpoint path
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"safeguard_checkpoint_step_{step}.pt"
        )
        
        torch.save({
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "loss_history": list(self.loss_history),
            "gradient_norms": list(self.gradient_norms),
        }, checkpoint_path)
        
        logger.info(f"Saved safeguard checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load safeguard checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Checkpoint data
        """
        checkpoint = torch.load(checkpoint_path)
        
        # Restore history
        self.loss_history = deque(checkpoint.get("loss_history", []), maxlen=self.window_size)
        self.gradient_norms = deque(checkpoint.get("gradient_norms", []), maxlen=self.window_size)
        
        if self.loss_history:
            self.baseline_loss = np.mean(list(self.loss_history)[:min(10, len(self.loss_history))])
        
        return checkpoint
    
    def rollback(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 checkpoint_path: str) -> bool:
        """
        Rollback to previous checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            checkpoint_path: Path to checkpoint to rollback to
            
        Returns:
            True if rollback successful
        """
        try:
            checkpoint = self.load_checkpoint(checkpoint_path)
            
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            self.rollback_count += 1
            logger.warning(f"Rolled back to checkpoint: {checkpoint_path} (rollback #{self.rollback_count})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current safeguard status."""
        return {
            "loss_history_length": len(self.loss_history),
            "baseline_loss": self.baseline_loss,
            "recent_loss_mean": np.mean(list(self.loss_history)) if self.loss_history else None,
            "recent_loss_std": np.std(list(self.loss_history)) if len(self.loss_history) > 1 else None,
            "gradient_norms_length": len(self.gradient_norms),
            "recent_gradient_mean": np.mean(list(self.gradient_norms)) if self.gradient_norms else None,
            "alerts_count": len(self.alerts),
            "rollback_count": self.rollback_count
        }

