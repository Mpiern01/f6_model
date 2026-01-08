"""
Dataset Evaluator
Evaluate datasets to determine quality and usage

MIT-level engineering: Production-grade dataset evaluation
"""

import logging
from typing import Dict, Any, List, Optional
from datasets import load_dataset
from streaming.hf_stream import StreamingDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetEvaluator:
    """
    Evaluates datasets to determine quality scores and usage recommendations.
    """
    
    def __init__(self):
        """Initialize dataset evaluator."""
        pass
    
    def evaluate_dataset(
        self,
        dataset_name: str,
        sample_size: int = 100,
        streaming: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a dataset.
        
        Args:
            dataset_name: HuggingFace dataset name
            sample_size: Number of samples to evaluate
            streaming: Whether to use streaming
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating dataset: {dataset_name}")
        
        try:
            loader = StreamingDataLoader(dataset_name, streaming=streaming)
            
            scores = {
                "quality": 0.0,
                "diversity": 0.0,
                "relevance": 0.0,
                "size_estimate": 0,
                "format_quality": 0.0
            }
            
            samples = []
            for i, sample in enumerate(loader.stream(max_samples=sample_size)):
                samples.append(sample)
                if i >= sample_size:
                    break
            
            if not samples:
                return {
                    "dataset": dataset_name,
                    "error": "No samples found",
                    "scores": scores
                }
            
            # Evaluate quality metrics
            scores["quality"] = self._evaluate_quality(samples)
            scores["diversity"] = self._evaluate_diversity(samples)
            scores["relevance"] = self._evaluate_relevance(samples)
            scores["format_quality"] = self._evaluate_format(samples)
            
            return {
                "dataset": dataset_name,
                "scores": scores,
                "sample_count": len(samples),
                "recommendation": self._get_recommendation(scores)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating {dataset_name}: {e}")
            return {
                "dataset": dataset_name,
                "error": str(e),
                "scores": {}
            }
    
    def _evaluate_quality(self, samples: List[Dict[str, Any]]) -> float:
        """Evaluate sample quality."""
        # Check for required fields, non-empty content, etc.
        quality_scores = []
        
        for sample in samples:
            score = 0.0
            
            # Check if sample has content
            if sample and len(str(sample)) > 10:
                score += 0.3
            
            # Check for common quality indicators
            sample_str = str(sample).lower()
            if any(keyword in sample_str for keyword in ["code", "function", "def", "class"]):
                score += 0.2
            
            if any(keyword in sample_str for keyword in ["reasoning", "step", "solution"]):
                score += 0.2
            
            # Check structure
            if isinstance(sample, dict) and len(sample) > 0:
                score += 0.3
            
            quality_scores.append(score)
        
        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    def _evaluate_diversity(self, samples: List[Dict[str, Any]]) -> float:
        """Evaluate sample diversity."""
        if len(samples) < 2:
            return 0.5
        
        # Simple diversity: check for unique content
        unique_content = set()
        for sample in samples:
            content_hash = hash(str(sample)[:100])  # First 100 chars
            unique_content.add(content_hash)
        
        diversity = len(unique_content) / len(samples)
        return diversity
    
    def _evaluate_relevance(self, samples: List[Dict[str, Any]]) -> float:
        """Evaluate relevance to training goals."""
        relevance_keywords = [
            "code", "function", "class", "def", "import",
            "reasoning", "step", "solution", "problem",
            "test", "bug", "fix", "error"
        ]
        
        relevance_scores = []
        for sample in samples:
            sample_str = str(sample).lower()
            matches = sum(1 for keyword in relevance_keywords if keyword in sample_str)
            relevance_scores.append(min(1.0, matches / len(relevance_keywords)))
        
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    
    def _evaluate_format(self, samples: List[Dict[str, Any]]) -> float:
        """Evaluate format quality."""
        format_scores = []
        
        for sample in samples:
            score = 0.0
            
            # Check if it's a dict (structured)
            if isinstance(sample, dict):
                score += 0.5
                
                # Check for common fields
                common_fields = ["text", "input", "output", "instruction", "response", "code"]
                if any(field in sample for field in common_fields):
                    score += 0.5
            
            format_scores.append(score)
        
        return sum(format_scores) / len(format_scores) if format_scores else 0.0
    
    def _get_recommendation(self, scores: Dict[str, float]) -> str:
        """Get usage recommendation based on scores."""
        overall = (
            scores.get("quality", 0.0) * 0.3 +
            scores.get("diversity", 0.0) * 0.2 +
            scores.get("relevance", 0.0) * 0.3 +
            scores.get("format_quality", 0.0) * 0.2
        )
        
        if overall >= 0.8:
            return "ALL"  # Use all samples
        elif overall >= 0.6:
            return "HEAVY"  # Use heavy portion
        elif overall >= 0.4:
            return "MEDIUM"  # Use medium portion
        else:
            return "LIGHT"  # Use light portion or skip

