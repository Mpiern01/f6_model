"""
Selective Pre-training for Private Fine-tuning
Carefully pre-train on subset of public datasets guided by private datasets

Reference: Selective Pre-training for Private Fine-tuning
arXiv: 2305.13865

MIT-level engineering: Production-grade selective pre-training
"""

import torch
from typing import Dict, Any, List, Optional
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SelectivePreTraining:
    """
    Selective Pre-training: Pre-train on public data subset
    guided by private dataset characteristics.
    """
    
    def __init__(self, private_dataset_analyzer=None):
        """
        Initialize selective pre-training.
        
        Args:
            private_dataset_analyzer: Analyzer for private dataset characteristics
        """
        self.private_dataset_analyzer = private_dataset_analyzer
        self.selection_criteria = {}
    
    def analyze_private_dataset(self, private_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze private dataset to extract characteristics.
        
        Args:
            private_dataset: Private dataset samples
            
        Returns:
            Characteristics dictionary
        """
        logger.info("Analyzing private dataset...")
        
        # Extract characteristics
        characteristics = {
            "domain": self._extract_domain(private_dataset),
            "language": self._extract_language(private_dataset),
            "code_patterns": self._extract_code_patterns(private_dataset),
            "complexity": self._extract_complexity(private_dataset),
            "topics": self._extract_topics(private_dataset)
        }
        
        logger.info(f"Private dataset characteristics: {characteristics}")
        return characteristics
    
    def select_public_data(
        self,
        public_datasets: List[Dict[str, Any]],
        private_characteristics: Dict[str, Any],
        selection_ratio: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Select subset of public data based on private characteristics.
        
        Args:
            public_datasets: List of public datasets
            private_characteristics: Characteristics from private dataset
            selection_ratio: Ratio of public data to select
            
        Returns:
            Selected public data
        """
        logger.info(f"Selecting {selection_ratio*100}% of public data...")
        
        # Score each public sample based on similarity to private characteristics
        scored_samples = []
        
        for dataset in public_datasets:
            for sample in dataset.get("samples", []):
                score = self._compute_similarity_score(sample, private_characteristics)
                scored_samples.append((sample, score))
        
        # Sort by score and select top
        scored_samples.sort(key=lambda x: x[1], reverse=True)
        num_select = int(len(scored_samples) * selection_ratio)
        selected = [sample for sample, _ in scored_samples[:num_select]]
        
        logger.info(f"Selected {len(selected)} samples from public data")
        return selected
    
    def _extract_domain(self, dataset: List[Dict[str, Any]]) -> str:
        """Extract domain from dataset using content analysis."""
        if not dataset:
            return "unknown"
        
        # Analyze sample content
        domain_keywords = {
            "software_engineering": ["code", "function", "class", "import", "def", "return", "test", "bug", "fix"],
            "mathematics": ["equation", "solve", "calculate", "theorem", "proof", "formula", "integral", "derivative"],
            "reasoning": ["reason", "logic", "conclusion", "premise", "infer", "deduce", "analyze"],
            "natural_language": ["sentence", "paragraph", "text", "language", "grammar", "syntax"],
            "science": ["experiment", "hypothesis", "data", "analysis", "research", "study"]
        }
        
        # Count keyword matches
        domain_scores = {domain: 0 for domain in domain_keywords}
        
        for sample in dataset[:100]:  # Sample first 100
            text = str(sample).lower()
            for domain, keywords in domain_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        domain_scores[domain] += 1
        
        # Return domain with highest score
        if max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)
        else:
            return "general"
    
    def _extract_language(self, dataset: List[Dict[str, Any]]) -> str:
        """Extract programming language from dataset."""
        languages = []
        for sample in dataset:
            lang = sample.get("language", "unknown")
            languages.append(lang)
        
        if languages:
            return Counter(languages).most_common(1)[0][0]
        return "python"
    
    def _extract_code_patterns(self, dataset: List[Dict[str, Any]]) -> List[str]:
        """Extract common code patterns."""
        # Simplified - in production, use AST analysis
        return ["function_definitions", "class_definitions", "error_handling"]
    
    def _extract_complexity(self, dataset: List[Dict[str, Any]]) -> float:
        """Extract average complexity."""
        complexities = []
        for sample in dataset:
            code = sample.get("code", "")
            # Simplified complexity metric
            complexity = len(code.split("\n")) / 100.0
            complexities.append(complexity)
        
        return sum(complexities) / len(complexities) if complexities else 1.0
    
    def _extract_topics(self, dataset: List[Dict[str, Any]]) -> List[str]:
        """Extract topics/themes."""
        # Simplified - in production, use topic modeling
        return ["code_generation", "bug_fixing", "refactoring"]
    
    def _compute_similarity_score(
        self,
        sample: Dict[str, Any],
        private_characteristics: Dict[str, Any]
    ) -> float:
        """Compute similarity score between sample and private characteristics."""
        score = 0.0
        
        # Domain match
        sample_domain = self._extract_domain([sample])
        if sample_domain == private_characteristics.get("domain"):
            score += 0.3
        
        # Language match
        sample_lang = sample.get("language", "")
        if sample_lang == private_characteristics.get("language"):
            score += 0.3
        
        # Complexity match
        sample_complexity = self._extract_complexity([sample])
        private_complexity = private_characteristics.get("complexity", 1.0)
        complexity_diff = abs(sample_complexity - private_complexity)
        score += max(0, 0.2 * (1 - complexity_diff))
        
        # Pattern match (simplified)
        score += 0.2
        
        return score

