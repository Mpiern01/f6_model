"""
Intelligent Dataset Mixer
Mixes datasets based on priority, quality scores, and weights

Handles "<- ALL" datasets specially (uses all samples)

MIT-level engineering: Production-grade dataset mixing
"""

import logging
import random
from typing import Dict, Any, List, Optional, Iterator
from dataset_registry.registry import DatasetConfig, DatasetPriority, get_all_datasets, get_datasets_by_priority

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelligentDatasetMixer:
    """
    Intelligently mixes datasets with support for ALL priority datasets.
    """
    
    def __init__(
        self,
        dataset_configs: List[DatasetConfig],
        all_datasets_ratio: float = 0.4,  # 40% from ALL datasets
        heavy_datasets_ratio: float = 0.3,  # 30% from HEAVY
        medium_datasets_ratio: float = 0.2,  # 20% from MEDIUM
        light_datasets_ratio: float = 0.1   # 10% from LIGHT
    ):
        """
        Initialize intelligent dataset mixer.
        
        Args:
            dataset_configs: List of dataset configurations
            all_datasets_ratio: Ratio for ALL priority datasets
            heavy_datasets_ratio: Ratio for HEAVY priority datasets
            medium_datasets_ratio: Ratio for MEDIUM priority datasets
            light_datasets_ratio: Ratio for LIGHT priority datasets
        """
        self.dataset_configs = dataset_configs
        
        # Separate by priority
        self.all_datasets = [ds for ds in dataset_configs if ds.priority == DatasetPriority.ALL]
        self.heavy_datasets = [ds for ds in dataset_configs if ds.priority == DatasetPriority.HEAVY]
        self.medium_datasets = [ds for ds in dataset_configs if ds.priority == DatasetPriority.MEDIUM]
        self.light_datasets = [ds for ds in dataset_configs if ds.priority == DatasetPriority.LIGHT]
        
        # Ratios
        self.all_ratio = all_datasets_ratio
        self.heavy_ratio = heavy_datasets_ratio
        self.medium_ratio = medium_datasets_ratio
        self.light_ratio = light_datasets_ratio
        
        # Normalize ratios
        total = all_datasets_ratio + heavy_datasets_ratio + medium_datasets_ratio + light_datasets_ratio
        if total != 1.0:
            self.all_ratio /= total
            self.heavy_ratio /= total
            self.medium_ratio /= total
            self.light_ratio /= total
        
        logger.info(f"Dataset mixer initialized:")
        logger.info(f"  ALL datasets: {len(self.all_datasets)}")
        logger.info(f"  HEAVY datasets: {len(self.heavy_datasets)}")
        logger.info(f"  MEDIUM datasets: {len(self.medium_datasets)}")
        logger.info(f"  LIGHT datasets: {len(self.light_datasets)}")
    
    def create_streaming_schedule(self, total_samples: int) -> List[Dict[str, Any]]:
        """
        Create streaming schedule based on priorities and ratios.
        
        Args:
            total_samples: Total number of samples to stream
            
        Returns:
            List of dataset assignments with sample counts
        """
        schedule = []
        
        # Calculate sample counts per priority
        all_samples = int(total_samples * self.all_ratio)
        heavy_samples = int(total_samples * self.heavy_ratio)
        medium_samples = int(total_samples * self.medium_ratio)
        light_samples = int(total_samples * self.light_ratio)
        
        # Distribute samples within each priority group
        # ALL datasets: distribute evenly (or by weight)
        if self.all_datasets:
            all_per_dataset = all_samples // len(self.all_datasets)
            for ds in self.all_datasets:
                schedule.append({
                    "dataset": ds,
                    "samples": all_per_dataset * int(ds.weight),  # Weighted
                    "priority": "ALL",
                    "use_all": True  # Mark for full usage
                })
        
        # HEAVY datasets: distribute by weight
        if self.heavy_datasets:
            total_weight = sum(ds.weight for ds in self.heavy_datasets)
            for ds in self.heavy_datasets:
                ratio = ds.weight / total_weight
                schedule.append({
                    "dataset": ds,
                    "samples": int(heavy_samples * ratio),
                    "priority": "HEAVY",
                    "use_all": False
                })
        
        # MEDIUM datasets: distribute by weight
        if self.medium_datasets:
            total_weight = sum(ds.weight for ds in self.medium_datasets)
            for ds in self.medium_datasets:
                ratio = ds.weight / total_weight
                schedule.append({
                    "dataset": ds,
                    "samples": int(medium_samples * ratio),
                    "priority": "MEDIUM",
                    "use_all": False
                })
        
        # LIGHT datasets: distribute by weight
        if self.light_datasets:
            total_weight = sum(ds.weight for ds in self.light_datasets)
            for ds in self.light_datasets:
                ratio = ds.weight / total_weight
                schedule.append({
                    "dataset": ds,
                    "samples": int(light_samples * ratio),
                    "priority": "LIGHT",
                    "use_all": False
                })
        
        return schedule
    
    def stream_mixed(
        self,
        schedule: List[Dict[str, Any]],
        max_total_samples: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream mixed datasets according to schedule.
        
        Args:
            schedule: Streaming schedule from create_streaming_schedule
            max_total_samples: Maximum total samples to stream
            
        Yields:
            Mixed dataset samples
        """
        from streaming.hf_stream import StreamingDataLoader
        
        # Create loaders for each dataset
        loaders = {}
        for item in schedule:
            ds = item["dataset"]
            loader = StreamingDataLoader(
                dataset_name=ds.hf_path,
                split=ds.split,
                streaming=ds.streaming
            )
            loaders[ds.hf_path] = {
                "loader": loader,
                "schedule": item,
                "iterator": None
            }
        
        # Initialize iterators
        for key, value in loaders.items():
            schedule_item = value["schedule"]
            if schedule_item["use_all"]:
                # ALL datasets: stream all available
                value["iterator"] = value["loader"].stream(max_samples=None)
            else:
                # Other priorities: stream up to scheduled count
                value["iterator"] = value["loader"].stream(max_samples=schedule_item["samples"])
        
        # Mix streams
        total_yielded = 0
        active_loaders = {k: v for k, v in loaders.items() if v["iterator"] is not None}
        
        # Create round-robin schedule based on ratios
        round_robin = []
        for item in schedule:
            ds = item["dataset"]
            if ds.hf_path in active_loaders:
                # Add to round-robin proportional to samples
                count = item["samples"] if not item["use_all"] else 100  # High count for ALL
                round_robin.extend([ds.hf_path] * count)
        
        random.shuffle(round_robin)
        round_robin_iter = iter(round_robin * (max_total_samples // len(round_robin) + 1))
        
        while active_loaders and (max_total_samples is None or total_yielded < max_total_samples):
            try:
                # Get next dataset from round-robin
                dataset_name = next(round_robin_iter)
                
                if dataset_name not in active_loaders:
                    continue
                
                loader_info = active_loaders[dataset_name]
                
                # Get next sample
                try:
                    sample = next(loader_info["iterator"])
                    sample["_dataset_source"] = dataset_name
                    sample["_dataset_priority"] = loader_info["schedule"]["priority"]
                    yield sample
                    total_yielded += 1
                except StopIteration:
                    # Dataset exhausted
                    logger.info(f"Dataset exhausted: {dataset_name}")
                    del active_loaders[dataset_name]
                    continue
                    
            except StopIteration:
                # Round-robin exhausted
                break
        
        logger.info(f"Streamed {total_yielded} mixed samples")

