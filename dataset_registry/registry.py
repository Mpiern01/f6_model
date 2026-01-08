"""
Dataset Registry
Comprehensive registry of all datasets for F6 StreamTrain

MIT-level engineering: Production-grade dataset management
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

class DatasetPriority(Enum):
    """Dataset priority levels."""
    ALL = "all"  # Use all samples
    HEAVY = "heavy"  # Use heavy portion
    MEDIUM = "medium"  # Use medium portion
    LIGHT = "light"  # Use light portion
    EVALUATE = "evaluate"  # Evaluate first, then decide

@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    hf_path: str
    priority: DatasetPriority
    category: str  # code, math, reasoning, vision, etc.
    description: str
    estimated_size: Optional[int] = None
    split: str = "train"
    streaming: bool = True
    weight: float = 1.0  # Relative weight for mixing

# ALL datasets marked with <- ALL
ALL_DATASETS = [
    "a-m-team/AM-DeepSeek-Distilled-40M",
    "a-m-team/AM-DeepSeek-R1-Distilled-1.4M",
    "QuixiAI/dolphin-r1",
    "QuixiAI/dolphin-coder",
    "microsoft/rStar-Coder",
    "AI-MO/NuminaMath-CoT",
    "TeichAI/MiniMax-M2.1-8800x",
    "TeichAI/claude-haiku-4.5-1700x",
    "TeichAI/claude-haiku-4.5-high-reasoning-1700x",
    "TeichAI/gemini-3-flash-preview-1000x",
    "TeichAI/gpt-5.1-codex-max-1000x",
    "TeichAI/gpt-5.2-high-reasoning-250x",
    "TeichAI/deepseek-v3.2-speciale-openr1-math-3k",
    "TeichAI/deepseek-v3.2-speciale-1000x",
    "TeichAI/claude-4.5-opus-high-reasoning-250x",
    "TeichAI/gemini-2.5-flash-11000x",
    "sequelbox/UML-Generator-Dataset-DeepSeek-V3.2",
    "smcleod/golang-coder",
    "livesweagent/gpt-5-2_swebench_verified_traj",
    "livesweagent/claude-opus-4-5_swebench_verified_traj",
    "livesweagent/gemini_3_pro_swebench_verified_traj",
]

DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    # Code & Programming
    "qwen3-coder": DatasetConfig(
        name="qwen3-coder-30b",
        hf_path="Nilaksh404/qwen3-coder-30b",
        priority=DatasetPriority.HEAVY,
        category="code",
        description="Qwen3 coder dataset"
    ),
    "nvidia-python-code": DatasetConfig(
        name="nvidia-python-code",
        hf_path="kye/all-nvidia-python-code",
        priority=DatasetPriority.HEAVY,
        category="code",
        description="NVIDIA Python code dataset"
    ),
    "nvidia-code-reasoning": DatasetConfig(
        name="nvidia-code-reasoning",
        hf_path="ykarout/nvidia-code-reasoning-clean",
        priority=DatasetPriority.HEAVY,
        category="code",
        description="NVIDIA code reasoning dataset"
    ),
    "nvidia-open-code": DatasetConfig(
        name="nvidia-open-code",
        hf_path="ykarout/nvidia-open-code-15k-25k",
        priority=DatasetPriority.HEAVY,
        category="code",
        description="NVIDIA open code dataset"
    ),
    "rstar-coder": DatasetConfig(
        name="rstar-coder",
        hf_path="microsoft/rStar-Coder",
        priority=DatasetPriority.ALL,
        category="code",
        description="Microsoft rStar Coder dataset"
    ),
    "dolphin-coder": DatasetConfig(
        name="dolphin-coder",
        hf_path="QuixiAI/dolphin-coder",
        priority=DatasetPriority.ALL,
        category="code",
        description="Dolphin coder dataset"
    ),
    "golang-coder": DatasetConfig(
        name="golang-coder",
        hf_path="smcleod/golang-coder",
        priority=DatasetPriority.ALL,
        category="code",
        description="Golang coder dataset"
    ),
    
    # Math & Reasoning
    "openmath-reasoning": DatasetConfig(
        name="openmath-reasoning",
        hf_path="nvidia/OpenMathReasoning",
        priority=DatasetPriority.HEAVY,
        category="math",
        description="NVIDIA OpenMath reasoning"
    ),
    "openmath-instruct": DatasetConfig(
        name="openmath-instruct",
        hf_path="nvidia/OpenMathInstruct-2",
        priority=DatasetPriority.HEAVY,
        category="math",
        description="NVIDIA OpenMath instruct"
    ),
    "nemotron-math-v1": DatasetConfig(
        name="nemotron-math-v1",
        hf_path="nvidia/Nemotron-CC-Math-v1",
        priority=DatasetPriority.HEAVY,
        category="math",
        description="NVIDIA Nemotron math v1"
    ),
    "nemotron-math-v2": DatasetConfig(
        name="nemotron-math-v2",
        hf_path="nvidia/Nemotron-Math-v2",
        priority=DatasetPriority.HEAVY,
        category="math",
        description="NVIDIA Nemotron math v2"
    ),
    "numinamath-cot": DatasetConfig(
        name="numinamath-cot",
        hf_path="AI-MO/NuminaMath-CoT",
        priority=DatasetPriority.ALL,
        category="math",
        description="NuminaMath CoT dataset"
    ),
    "gsm8k": DatasetConfig(
        name="gsm8k",
        hf_path="openai/gsm8k",
        priority=DatasetPriority.MEDIUM,
        category="math",
        description="GSM8K math dataset"
    ),
    
    # Reasoning & Thinking
    "am-deepseek-distilled-40m": DatasetConfig(
        name="am-deepseek-distilled-40m",
        hf_path="a-m-team/AM-DeepSeek-Distilled-40M",
        priority=DatasetPriority.ALL,
        category="reasoning",
        description="AM DeepSeek distilled 40M"
    ),
    "am-deepseek-r1-distilled": DatasetConfig(
        name="am-deepseek-r1-distilled",
        hf_path="a-m-team/AM-DeepSeek-R1-Distilled-1.4M",
        priority=DatasetPriority.ALL,
        category="reasoning",
        description="AM DeepSeek R1 distilled"
    ),
    "dolphin-r1": DatasetConfig(
        name="dolphin-r1",
        hf_path="QuixiAI/dolphin-r1",
        priority=DatasetPriority.ALL,
        category="reasoning",
        description="Dolphin R1 dataset"
    ),
    "thinker-datasets": [
        DatasetConfig(
            name="lapo-thinker",
            hf_path="xytian1008/LAPO-Thinker-SFT-train100k",
            priority=DatasetPriority.HEAVY,
            category="reasoning",
            description="LAPO Thinker dataset"
        ),
        DatasetConfig(
            name="thinker",
            hf_path="catsaresupercool/thinker",
            priority=DatasetPriority.HEAVY,
            category="reasoning",
            description="Thinker dataset"
        ),
        DatasetConfig(
            name="kag-thinker",
            hf_path="OpenSPG/KAG-Thinker-training-dataset",
            priority=DatasetPriority.HEAVY,
            category="reasoning",
            description="KAG Thinker dataset"
        ),
    ],
    
    # Long-Horizon & GUI
    "long-horizon-gui": DatasetConfig(
        name="long-horizon-gui",
        hf_path="CVC2233/Long-Horizon-GUI-Dataset",
        priority=DatasetPriority.HEAVY,
        category="long_horizon",
        description="Long-horizon GUI dataset"
    ),
    "long-horizon-execution": DatasetConfig(
        name="long-horizon-execution",
        hf_path="arvindh75/Long-Horizon-Execution",
        priority=DatasetPriority.HEAVY,
        category="long_horizon",
        description="Long-horizon execution dataset"
    ),
    
    # SWE Bench
    "swebench-verified": DatasetConfig(
        name="swebench-verified",
        hf_path="princeton-nlp/SWE-bench_Verified",
        priority=DatasetPriority.HEAVY,
        category="swe",
        description="SWE-bench verified"
    ),
    "swebench-pro": DatasetConfig(
        name="swebench-pro",
        hf_path="ScaleAI/SWE-bench_Pro",
        priority=DatasetPriority.HEAVY,
        category="swe",
        description="SWE-bench Pro"
    ),
    "swebench-trajectories": [
        DatasetConfig(
            name="gpt-5-2-swebench",
            hf_path="livesweagent/gpt-5-2_swebench_verified_traj",
            priority=DatasetPriority.ALL,
            category="swe",
            description="GPT-5-2 SWE-bench trajectories"
        ),
        DatasetConfig(
            name="claude-opus-swebench",
            hf_path="livesweagent/claude-opus-4-5_swebench_verified_traj",
            priority=DatasetPriority.ALL,
            category="swe",
            description="Claude Opus SWE-bench trajectories"
        ),
        DatasetConfig(
            name="gemini-3-pro-swebench",
            hf_path="livesweagent/gemini_3_pro_swebench_verified_traj",
            priority=DatasetPriority.ALL,
            category="swe",
            description="Gemini 3 Pro SWE-bench trajectories"
        ),
    ],
    
    # Vision & Multimodal
    "mmmu": DatasetConfig(
        name="mmmu",
        hf_path="MMMU/MMMU",
        priority=DatasetPriority.MEDIUM,
        category="vision",
        description="MMMU multimodal dataset"
    ),
    "mmmu-pro": DatasetConfig(
        name="mmmu-pro",
        hf_path="MMMU/MMMU_Pro",
        priority=DatasetPriority.MEDIUM,
        category="vision",
        description="MMMU Pro dataset"
    ),
    "deepseek-ocr": DatasetConfig(
        name="deepseek-ocr",
        hf_path="rickxt/DeepSeek-OCR",
        priority=DatasetPriority.MEDIUM,
        category="vision",
        description="DeepSeek OCR dataset"
    ),
    
    # ARC-AGI
    "arc-agi-datasets": [
        DatasetConfig(
            name="arc-agi-2",
            hf_path="zhmz90/arc-agi-2",
            priority=DatasetPriority.HEAVY,
            category="reasoning",
            description="ARC-AGI 2"
        ),
        DatasetConfig(
            name="arc-agi-ultra",
            hf_path="ayjays132/ARC_AGI_V1_ULTRA",
            priority=DatasetPriority.HEAVY,
            category="reasoning",
            description="ARC-AGI Ultra"
        ),
    ],
    
    # Tool Use
    "toolscale": DatasetConfig(
        name="toolscale",
        hf_path="nvidia/ToolScale",
        priority=DatasetPriority.HEAVY,
        category="tool_use",
        description="NVIDIA ToolScale dataset"
    ),
    "xlam-function-calling": DatasetConfig(
        name="xlam-function-calling",
        hf_path="Salesforce/xlam-function-calling-60k",
        priority=DatasetPriority.MEDIUM,
        category="tool_use",
        description="XLAM function calling"
    ),
    
    # TeichAI High-Quality Datasets (ALL)
    "teichai-datasets": [
        DatasetConfig(
            name="minimax-m2.1",
            hf_path="TeichAI/MiniMax-M2.1-8800x",
            priority=DatasetPriority.ALL,
            category="reasoning",
            description="MiniMax M2.1 8800x",
            weight=2.0
        ),
        DatasetConfig(
            name="claude-haiku-4.5",
            hf_path="TeichAI/claude-haiku-4.5-1700x",
            priority=DatasetPriority.ALL,
            category="reasoning",
            description="Claude Haiku 4.5",
            weight=2.0
        ),
        DatasetConfig(
            name="claude-haiku-reasoning",
            hf_path="TeichAI/claude-haiku-4.5-high-reasoning-1700x",
            priority=DatasetPriority.ALL,
            category="reasoning",
            description="Claude Haiku high reasoning",
            weight=2.0
        ),
        DatasetConfig(
            name="gemini-3-flash",
            hf_path="TeichAI/gemini-3-flash-preview-1000x",
            priority=DatasetPriority.ALL,
            category="reasoning",
            description="Gemini 3 Flash",
            weight=2.0
        ),
        DatasetConfig(
            name="gpt-5.1-codex",
            hf_path="TeichAI/gpt-5.1-codex-max-1000x",
            priority=DatasetPriority.ALL,
            category="code",
            description="GPT-5.1 Codex",
            weight=2.0
        ),
        DatasetConfig(
            name="gpt-5.2-reasoning",
            hf_path="TeichAI/gpt-5.2-high-reasoning-250x",
            priority=DatasetPriority.ALL,
            category="reasoning",
            description="GPT-5.2 high reasoning",
            weight=2.0
        ),
        DatasetConfig(
            name="deepseek-v3.2-math",
            hf_path="TeichAI/deepseek-v3.2-speciale-openr1-math-3k",
            priority=DatasetPriority.ALL,
            category="math",
            description="DeepSeek V3.2 math",
            weight=2.0
        ),
        DatasetConfig(
            name="deepseek-v3.2-speciale",
            hf_path="TeichAI/deepseek-v3.2-speciale-1000x",
            priority=DatasetPriority.ALL,
            category="reasoning",
            description="DeepSeek V3.2 speciale",
            weight=2.0
        ),
        DatasetConfig(
            name="claude-4.5-opus",
            hf_path="TeichAI/claude-4.5-opus-high-reasoning-250x",
            priority=DatasetPriority.ALL,
            category="reasoning",
            description="Claude 4.5 Opus reasoning",
            weight=2.0
        ),
        DatasetConfig(
            name="gemini-2.5-flash",
            hf_path="TeichAI/gemini-2.5-flash-11000x",
            priority=DatasetPriority.ALL,
            category="reasoning",
            description="Gemini 2.5 Flash",
            weight=2.0
        ),
    ],
    
    # Multimodal Reasoning Lab
    "multimodal-reasoning": [
        DatasetConfig(
            name="zebra-cot",
            hf_path="multimodal-reasoning-lab/Zebra-CoT",
            priority=DatasetPriority.MEDIUM,
            category="reasoning",
            description="Zebra CoT"
        ),
        DatasetConfig(
            name="chess",
            hf_path="multimodal-reasoning-lab/Chess",
            priority=DatasetPriority.MEDIUM,
            category="reasoning",
            description="Chess reasoning"
        ),
        DatasetConfig(
            name="arc-agi-mrl",
            hf_path="multimodal-reasoning-lab/ARC-AGI",
            priority=DatasetPriority.HEAVY,
            category="reasoning",
            description="ARC-AGI from MRL"
        ),
    ],
    
    # Code Contests
    "code-contests": DatasetConfig(
        name="code-contests",
        hf_path="deepmind/code_contests",
        priority=DatasetPriority.HEAVY,
        category="code",
        description="DeepMind code contests"
    ),
    
    # Prime Intellect
    "prime-intellect": [
        DatasetConfig(
            name="stackv1",
            hf_path="PrimeIntellect/StackV1-popular",
            priority=DatasetPriority.MEDIUM,
            category="code",
            description="Stack V1 popular"
        ),
        DatasetConfig(
            name="intellect-3-sft",
            hf_path="PrimeIntellect/INTELLECT-3-SFT",
            priority=DatasetPriority.HEAVY,
            category="reasoning",
            description="Intellect 3 SFT"
        ),
    ],
}

def get_all_datasets() -> List[DatasetConfig]:
    """Get all datasets as flat list."""
    datasets = []
    for key, value in DATASET_REGISTRY.items():
        if isinstance(value, list):
            datasets.extend(value)
        else:
            datasets.append(value)
    
    # Add additional datasets from gap analysis (lazy import to avoid circular)
    try:
        from .additional_datasets import get_additional_datasets
        datasets.extend(get_additional_datasets())
    except ImportError:
        pass  # Additional datasets not available
    
    return datasets

def get_datasets_by_priority(priority: DatasetPriority) -> List[DatasetConfig]:
    """Get datasets by priority."""
    all_ds = get_all_datasets()
    return [ds for ds in all_ds if ds.priority == priority]

def get_datasets_by_category(category: str) -> List[DatasetConfig]:
    """Get datasets by category."""
    all_ds = get_all_datasets()
    return [ds for ds in all_ds if ds.category == category]

