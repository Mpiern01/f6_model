"""
Dataset Manager
Manages dataset selection, evaluation, and streaming

MIT-level engineering: Production-grade dataset management
"""

import logging
from typing import Dict, Any, List, Optional
from dataset_registry.registry import get_all_datasets, get_datasets_by_priority, get_datasets_by_category, DatasetPriority
from dataset_registry.evaluator import DatasetEvaluator
from streaming.mixers import IntelligentDatasetMixer
from streaming.hf_stream import StreamingDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Manages datasets for training pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dataset manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.evaluator = DatasetEvaluator()
        self.evaluated_datasets = {}
    
    def get_datasets_for_stage(self, stage: str) -> List[Dict[str, Any]]:
        """
        Get datasets for specific training stage.
        
        Args:
            stage: Stage name (e.g., "stage1_midtrain")
            
        Returns:
            List of dataset configurations
        """
        stage_config = self.config.get("stages", {}).get(stage, {})
        
        categories = stage_config.get("categories", ["all"])
        priorities = stage_config.get("priorities", ["ALL", "HEAVY"])
        exclude_categories = stage_config.get("exclude_categories", [])
        
        # Get all datasets
        all_datasets = get_all_datasets()
        
        # Filter by category
        if "all" not in categories:
            filtered = []
            for ds in all_datasets:
                if ds.category in categories:
                    filtered.append(ds)
            all_datasets = filtered
        
        # Filter by priority
        priority_enums = [DatasetPriority[p] for p in priorities]
        filtered = [ds for ds in all_datasets if ds.priority in priority_enums]
        
        # Exclude categories
        if exclude_categories:
            filtered = [ds for ds in filtered if ds.category not in exclude_categories]
        
        logger.info(f"Selected {len(filtered)} datasets for {stage}")
        return [ds.__dict__ for ds in filtered]
    
    def evaluate_datasets(
        self,
        dataset_names: Optional[List[str]] = None,
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Evaluate datasets.
        
        Args:
            dataset_names: List of dataset names to evaluate (all if None)
            sample_size: Sample size for evaluation
            
        Returns:
            Evaluation results
        """
        if dataset_names is None:
            all_ds = get_all_datasets()
            dataset_names = [ds.hf_path for ds in all_ds]
        
        results = {}
        for dataset_name in dataset_names:
            result = self.evaluator.evaluate_dataset(dataset_name, sample_size)
            results[dataset_name] = result
            self.evaluated_datasets[dataset_name] = result
        
        return results
    
    def create_mixer(
        self,
        stage: str,
        total_samples: Optional[int] = None
    ) -> IntelligentDatasetMixer:
        """
        Create dataset mixer for stage.
        
        Args:
            stage: Stage name
            total_samples: Total samples (for scheduling)
            
        Returns:
            IntelligentDatasetMixer instance
        """
        # Get datasets for stage
        dataset_dicts = self.get_datasets_for_stage(stage)
        
        # Convert to DatasetConfig objects
        from dataset_registry.registry import DatasetConfig
        dataset_configs = []
        for ds_dict in dataset_dicts:
            config = DatasetConfig(**ds_dict)
            dataset_configs.append(config)
        
        # Get mixing ratios
        mixing_config = self.config.get("mixing", {})
        
        mixer = IntelligentDatasetMixer(
            dataset_configs=dataset_configs,
            all_datasets_ratio=mixing_config.get("all_datasets_ratio", 0.4),
            heavy_datasets_ratio=mixing_config.get("heavy_datasets_ratio", 0.3),
            medium_datasets_ratio=mixing_config.get("medium_datasets_ratio", 0.2),
            light_datasets_ratio=mixing_config.get("light_datasets_ratio", 0.1)
        )
        
        return mixer
    
    def stream_for_stage(
        self,
        stage: str,
        max_samples: Optional[int] = None
    ):
        """
        Stream datasets for training stage.
        
        Args:
            stage: Stage name
            max_samples: Maximum samples to stream
            
        Yields:
            Mixed dataset samples
        """
        # Create mixer
        mixer = self.create_mixer(stage, max_samples)
        
        # Create schedule
        schedule = mixer.create_streaming_schedule(max_samples or 1000000)
        
        # Stream mixed
        for sample in mixer.stream_mixed(schedule, max_total_samples=max_samples):
            yield sample

