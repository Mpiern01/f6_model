"""
2026 PHD-Level Training Improvements
Integrates latest research for small data → big benchmark gains

Techniques:
- QuaRot: Quantization with rotations
- BitNet v2: 4-bit activation quantization
- AMXFP4: Asymmetric microscaling
- Distilling Step-by-Step
- Selective Pre-training
- Mixture-of-Depths: Conditional computation (2026)
- Speculative Decoding 2.0: Mixture of Attentions (2026)
- Data-Centric Training: Quality over quantity (2026)

MIT-level engineering: State-of-the-art training techniques
PhD-level math: Optimal routing, curriculum learning, speculative decoding
"""

from .quarot import QuaRotQuantization
from .bitnet_v2 import BitNetV2Quantization
from .amxfp4 import AMXFP4Quantization
from .distilling import StepByStepDistillation
from .selective_pretraining import SelectivePreTraining
from .mixture_of_depths import MixtureOfDepthsRouter, MixtureOfDepthsLayer, apply_mixture_of_depths
from .speculative_decoding import SpeculativeDecoder
from .data_centric_2026 import DataQualityFilter, CurriculumScheduler, DataQualityMetrics

__all__ = [
    "QuaRotQuantization",
    "BitNetV2Quantization",
    "AMXFP4Quantization",
    "StepByStepDistillation",
    "SelectivePreTraining",
    "MixtureOfDepthsRouter",
    "MixtureOfDepthsLayer",
    "apply_mixture_of_depths",
    "SpeculativeDecoder",
    "DataQualityFilter",
    "CurriculumScheduler",
    "DataQualityMetrics",
]

