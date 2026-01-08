"""
Additional Datasets for Gap Filling
2026 PHD-level techniques and multimodal capabilities

MIT-level engineering: Comprehensive dataset coverage
"""

from typing import List
from dataset_registry.registry import DatasetConfig, DatasetPriority

# Additional datasets to fill gaps
ADDITIONAL_DATASETS = {
    # Audio Generation & Speech
    "common-voice": DatasetConfig(
        name="common-voice",
        hf_path="mozilla-foundation/common_voice",
        priority=DatasetPriority.HEAVY,
        category="audio",
        description="Multilingual speech dataset"
    ),
    "fleurs": DatasetConfig(
        name="fleurs",
        hf_path="google/fleurs",
        priority=DatasetPriority.HEAVY,
        category="audio",
        description="FLEURS multilingual speech"
    ),
    "voxpopuli": DatasetConfig(
        name="voxpopuli",
        hf_path="facebook/voxpopuli",
        priority=DatasetPriority.MEDIUM,
        category="audio",
        description="VoxPopuli multilingual speech"
    ),
    "librispeech": DatasetConfig(
        name="librispeech",
        hf_path="librispeech_asr",
        priority=DatasetPriority.MEDIUM,
        category="audio",
        description="LibriSpeech ASR"
    ),
    "people-speech": DatasetConfig(
        name="people-speech",
        hf_path="mlcommons/peoples_speech",
        priority=DatasetPriority.HEAVY,
        category="audio",
        description="People's Speech (1M+ hours)"
    ),
    
    # Video Generation & Understanding
    "activitynet": DatasetConfig(
        name="activitynet",
        hf_path="ActivityNet/ActivityNet",
        priority=DatasetPriority.HEAVY,
        category="video",
        description="ActivityNet video understanding"
    ),
    "kinetics": DatasetConfig(
        name="kinetics",
        hf_path="deepmind/kinetics",
        priority=DatasetPriority.HEAVY,
        category="video",
        description="Kinetics video dataset"
    ),
    "webvid": DatasetConfig(
        name="webvid",
        hf_path="webvid",
        priority=DatasetPriority.HEAVY,
        category="video",
        description="WebVid video-text dataset"
    ),
    "something-something": DatasetConfig(
        name="something-something",
        hf_path="something-something-v2",
        priority=DatasetPriority.MEDIUM,
        category="video",
        description="Something-Something video"
    ),
    
    # Robotics & Embodied AI
    "robocasa": DatasetConfig(
        name="robocasa",
        hf_path="robocasa/robocasa",
        priority=DatasetPriority.HEAVY,
        category="robotics",
        description="RoboCasa robotics dataset"
    ),
    "bridge-v2": DatasetConfig(
        name="bridge-v2",
        hf_path="rail-berkeley/bridge-v2",
        priority=DatasetPriority.HEAVY,
        category="robotics",
        description="Bridge V2 robotics"
    ),
    "roboset": DatasetConfig(
        name="roboset",
        hf_path="roboset",
        priority=DatasetPriority.MEDIUM,
        category="robotics",
        description="RoboSet robotics"
    ),
    
    # Agentic AI & Distillation
    "agent-bench": DatasetConfig(
        name="agent-bench",
        hf_path="THUDM/AgentBench",
        priority=DatasetPriority.HEAVY,
        category="agentic_ai",
        description="AgentBench agent evaluation"
    ),
    "agentic-distillation": DatasetConfig(
        name="agentic-distillation",
        hf_path="davidheineman/iclr-2026",
        priority=DatasetPriority.ALL,
        category="agentic_ai",
        description="Agentic distillation dataset",
        weight=2.0
    ),
    
    # 2026 Techniques
    "pika-alignment": DatasetConfig(
        name="pika-alignment",
        hf_path="PiKa/alignment-dataset",
        priority=DatasetPriority.ALL,
        category="reasoning",
        description="PiKa expert-level alignment (30k examples)",
        weight=2.0
    ),
    "semods": DatasetConfig(
        name="semods",
        hf_path="semods/dataset",
        priority=DatasetPriority.HEAVY,
        category="swe",
        description="SEMODS: 3,427 software engineering models"
    ),
    "taco": DatasetConfig(
        name="taco",
        hf_path="TACO/dataset",
        priority=DatasetPriority.HEAVY,
        category="code",
        description="TACO: Topics in Algorithmic Code generation"
    ),
    "leetcode": DatasetConfig(
        name="leetcode",
        hf_path="bigcode/leetcode",
        priority=DatasetPriority.HEAVY,
        category="code",
        description="LeetCode programming problems"
    ),
    
    # Multilingual
    "bloom-multilingual": DatasetConfig(
        name="bloom-multilingual",
        hf_path="bigscience/bloom",
        priority=DatasetPriority.MEDIUM,
        category="multilingual",
        description="BLOOM multilingual dataset"
    ),
    "multilingual-cc": DatasetConfig(
        name="multilingual-cc",
        hf_path="allenai/multilingual-cc",
        priority=DatasetPriority.MEDIUM,
        category="multilingual",
        description="Multilingual Common Crawl"
    ),
    
    # Financial
    "nifty-financial": DatasetConfig(
        name="nifty-financial",
        hf_path="financial/nifty",
        priority=DatasetPriority.MEDIUM,
        category="financial",
        description="NIFTY financial news headlines"
    ),
    
    # Physics & Scientific
    "collider-ml": DatasetConfig(
        name="collider-ml",
        hf_path="physics/collider-ml",
        priority=DatasetPriority.MEDIUM,
        category="physics",
        description="ColliderML: High-Luminosity Physics"
    ),
    "plaid-physics": DatasetConfig(
        name="plaid-physics",
        hf_path="physics/plaid",
        priority=DatasetPriority.MEDIUM,
        category="physics",
        description="PLAID: Physics-Learning AI Datamodel"
    ),
    
    # Tabular Data
    "tabular-benchmark": DatasetConfig(
        name="tabular-benchmark",
        hf_path="tabular/benchmark",
        priority=DatasetPriority.LIGHT,
        category="tabular",
        description="Tabular data benchmark"
    ),
    
    # Additional Vision
    "coco": DatasetConfig(
        name="coco",
        hf_path="detection-datasets/coco",
        priority=DatasetPriority.HEAVY,
        category="vision",
        description="COCO object detection"
    ),
    "imagenet": DatasetConfig(
        name="imagenet",
        hf_path="imagenet-1k",
        priority=DatasetPriority.MEDIUM,
        category="vision",
        description="ImageNet classification"
    ),
    "laion": DatasetConfig(
        name="laion",
        hf_path="laion/laion2B-en",
        priority=DatasetPriority.HEAVY,
        category="vision",
        description="LAION image-text dataset"
    ),
    
    # Long-Context & Reasoning
    "pile": DatasetConfig(
        name="pile",
        hf_path="EleutherAI/pile",
        priority=DatasetPriority.MEDIUM,
        category="reasoning",
        description="The Pile: 886GB diverse text"
    ),
    "mteb": DatasetConfig(
        name="mteb",
        hf_path="mteb/benchmark",
        priority=DatasetPriority.MEDIUM,
        category="reasoning",
        description="MTEB: Massive Text Embedding Benchmark"
    ),
}

def get_additional_datasets() -> List[DatasetConfig]:
    """Get all additional datasets."""
    return list(ADDITIONAL_DATASETS.values())

