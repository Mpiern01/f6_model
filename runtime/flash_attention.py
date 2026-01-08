"""
Flash Attention Integration
Memory-efficient attention computation

2025/2026 optimization: Flash Attention 2, variable length sequences

MIT-level engineering: Production-grade attention optimization
"""

import torch
import logging
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlashAttentionWrapper:
    """
    Wrapper for Flash Attention.
    
    Uses flash-attn library for memory-efficient attention computation.
    Falls back to standard attention if flash-attn not available.
    """
    
    def __init__(self, enable_flash: bool = True):
        """
        Initialize Flash Attention wrapper.
        
        Args:
            enable_flash: Enable Flash Attention if available
        """
        self.enable_flash = enable_flash
        self.flash_attn_available = False
        
        if enable_flash:
            self._check_flash_attn()
    
    def _check_flash_attn(self):
        """Check if flash-attn is available."""
        try:
            import flash_attn
            self.flash_attn_available = True
            logger.info("Flash Attention available")
        except ImportError:
            logger.warning(
                "flash-attn not available. Install with: "
                "pip install flash-attn --no-build-isolation"
            )
            self.flash_attn_available = False
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
        softmax_scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute attention using Flash Attention if available.
        
        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim]
            k: Key tensor [batch, seq_len, num_heads, head_dim]
            v: Value tensor [batch, seq_len, num_heads, head_dim]
            causal: Whether to use causal masking
            softmax_scale: Softmax scale (1/sqrt(head_dim) if None)
            
        Returns:
            Attention output tensor
        """
        if self.flash_attn_available and self.enable_flash:
            return self._flash_attention_forward(q, k, v, causal, softmax_scale)
        else:
            return self._standard_attention_forward(q, k, v, causal, softmax_scale)
    
    def _flash_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        softmax_scale: Optional[float]
    ) -> torch.Tensor:
        """Forward pass using Flash Attention."""
        try:
            from flash_attn import flash_attn_func
            
            # Flash Attention expects [batch, seq_len, num_heads, head_dim]
            # and returns [batch, seq_len, num_heads, head_dim]
            
            if softmax_scale is None:
                head_dim = q.shape[-1]
                softmax_scale = 1.0 / (head_dim ** 0.5)
            
            output = flash_attn_func(
                q, k, v,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=causal
            )
            
            return output
        except Exception as e:
            logger.warning(f"Flash Attention failed, falling back to standard: {e}")
            return self._standard_attention_forward(q, k, v, causal, softmax_scale)
    
    def _standard_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        softmax_scale: Optional[float]
    ) -> torch.Tensor:
        """Standard attention computation (fallback)."""
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        if softmax_scale is None:
            head_dim = q.shape[-1]
            softmax_scale = 1.0 / (head_dim ** 0.5)
        
        scores = scores * softmax_scale
        
        # Apply causal mask if needed
        if causal:
            seq_len = q.shape[1]
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=q.dtype),
                diagonal=1
            ) * -1e9
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply to values
        output = torch.matmul(attn_weights, v)
        
        return output

