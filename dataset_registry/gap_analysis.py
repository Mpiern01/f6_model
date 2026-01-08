"""
Dataset Gap Analysis
Identifies missing datasets and categories

MIT-level engineering: Comprehensive gap analysis
"""

from typing import Dict, Any, List
from dataset_registry.registry import get_all_datasets, get_datasets_by_category

# Missing categories and required datasets
GAP_ANALYSIS = {
    "audio": {
        "description": "Audio generation, speech synthesis, music generation",
        "required": True,
        "datasets": [
            "mozilla-foundation/common_voice",
            "google/fleurs",
            "facebook/voxpopuli",
            "librispeech_asr",
            "openslr",
        ]
    },
    "video": {
        "description": "Video generation, video understanding",
        "required": True,
        "datasets": [
            "ActivityNet",
            "Something-Something",
            "Kinetics",
            "WebVid",
        ]
    },
    "robotics": {
        "description": "Robotics, embodied AI, manipulation",
        "required": True,
        "datasets": [
            "robocasa/robocasa",
            "roboset",
            "bridge-v2",
        ]
    },
    "tabular": {
        "description": "Tabular data, structured data",
        "required": False,
        "datasets": [
            "tabular-benchmark",
            "openml",
        ]
    },
    "financial": {
        "description": "Financial data, market forecasting",
        "required": False,
        "datasets": [
            "financial-news",
            "stock-prices",
        ]
    },
    "physics": {
        "description": "Physics simulations, scientific computing",
        "required": False,
        "datasets": [
            "collider-ml",
            "plaid-physics",
        ]
    },
    "agentic_distillation": {
        "description": "Agentic distillation, teacher-student interaction",
        "required": True,
        "datasets": [
            "agentic-distillation",
            "teacher-student-interaction",
        ]
    },
    "federated_learning": {
        "description": "Federated learning scenarios",
        "required": False,
        "datasets": [
            "federated-benchmark",
        ]
    },
    "multilingual": {
        "description": "Multilingual datasets",
        "required": True,
        "datasets": [
            "bigscience/bloom",
            "multilingual-cc",
        ]
    },
    "agentic_ai": {
        "description": "Agentic AI, autonomous agents",
        "required": True,
        "datasets": [
            "agent-bench",
            "agentic-workflows",
        ]
    }
}

def analyze_gaps() -> Dict[str, Any]:
    """
    Analyze gaps in current dataset collection.
    
    Returns:
        Gap analysis results
    """
    all_datasets = get_all_datasets()
    current_categories = set(ds.category for ds in all_datasets)
    
    gaps = {}
    for category, info in GAP_ANALYSIS.items():
        if category not in current_categories:
            gaps[category] = {
                "missing": True,
                "required": info["required"],
                "description": info["description"],
                "suggested_datasets": info["datasets"]
            }
    
    return {
        "current_categories": list(current_categories),
        "missing_categories": list(gaps.keys()),
        "gaps": gaps,
        "required_gaps": [cat for cat, info in gaps.items() if info["required"]]
    }

