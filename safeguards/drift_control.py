"""
Drift Control
KL divergence regularization to prevent catastrophic drift

MIT-level engineering: Sophisticated drift prevention
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftControl:
    """
    Controls model drift using KL divergence regularization.
    Keeps model close to base/reference model.
    """
    
    def __init__(
        self,
        base_model: Optional[torch.nn.Module] = None,
        kl_lambda: float = 0.1,
        temperature: float = 1.0
    ):
        """
        Initialize drift control.
        
        Args:
            base_model: Base/reference model (frozen)
            kl_lambda: KL divergence weight (default: 0.1)
            temperature: Temperature for logits (default: 1.0)
        """
        self.base_model = base_model
        if self.base_model is not None:
            self.base_model.eval()
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        self.kl_lambda = kl_lambda
        self.temperature = temperature
    
    def compute_kl_loss(
        self,
        current_model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Compute KL divergence loss between current and base model.
        
        Args:
            current_model: Current model being trained
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for computing KL on next-token predictions
            
        Returns:
            Dictionary with:
                - kl_loss: KL divergence loss
                - kl_divergence: Raw KL divergence value
                - regularization_strength: Effective regularization strength
        """
        if self.base_model is None:
            return {
                "kl_loss": torch.tensor(0.0, device=input_ids.device),
                "kl_divergence": torch.tensor(0.0, device=input_ids.device),
                "regularization_strength": 0.0
            }
        
        current_model.eval()
        self.base_model.eval()
        
        with torch.no_grad():
            # Get logits from both models
            current_outputs = current_model(input_ids=input_ids, attention_mask=attention_mask)
            base_outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            
            current_logits = current_outputs.logits
            base_logits = base_outputs.logits
        
        current_model.train()
        
        # Apply temperature
        current_logits = current_logits / self.temperature
        base_logits = base_logits / self.temperature
        
        # Compute KL divergence
        # KL(P_current || P_base) = sum(P_current * log(P_current / P_base))
        current_probs = F.softmax(current_logits, dim=-1)
        base_probs = F.softmax(base_logits, dim=-1)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        current_probs = current_probs + eps
        base_probs = base_probs + eps
        current_probs = current_probs / current_probs.sum(dim=-1, keepdim=True)
        base_probs = base_probs / base_probs.sum(dim=-1, keepdim=True)
        
        # Compute KL divergence
        kl_div = current_probs * (torch.log(current_probs) - torch.log(base_probs))
        
        # Mask out padding tokens if labels provided
        if labels is not None:
            mask = (labels != -100).float()
            kl_div = kl_div * mask.unsqueeze(-1)
            kl_div = kl_div.sum(dim=-1) / (mask.sum(dim=-1, keepdim=True) + eps)
        else:
            kl_div = kl_div.sum(dim=-1)
        
        # Average over batch and sequence
        kl_divergence = kl_div.mean()
        
        # Apply regularization weight
        kl_loss = self.kl_lambda * kl_divergence
        
        return {
            "kl_loss": kl_loss,
            "kl_divergence": kl_divergence,
            "regularization_strength": self.kl_lambda
        }
    
    def update_kl_lambda(self, new_lambda: float):
        """
        Update KL divergence weight.
        
        Args:
            new_lambda: New KL weight
        """
        self.kl_lambda = new_lambda
        logger.info(f"Updated KL lambda to {new_lambda}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current drift control status."""
        return {
            "kl_lambda": self.kl_lambda,
            "temperature": self.temperature,
            "base_model_loaded": self.base_model is not None
        }

