"""
HuggingFace Streaming Data Pipeline
Guarantees: streaming=True, no dataset materialization

MIT-level engineering: Production-grade streaming with error handling
"""

import os
from typing import Iterator, Dict, Any, Optional, List
from datasets import load_dataset, IterableDataset, Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingDataLoader:
    """Streaming data loader with no-storage guarantee."""
    
    def __init__(self, dataset_name: str, split: str = "train", 
                 streaming: bool = True, **kwargs):
        """
        Initialize streaming data loader.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split (default: "train")
            streaming: Force streaming mode (default: True)
            **kwargs: Additional arguments for load_dataset
        """
        self.dataset_name = dataset_name
        self.split = split
        self.streaming = streaming
        self.kwargs = kwargs
        
        # Verify streaming mode
        if not streaming:
            logger.warning("Streaming disabled - this violates no-storage policy!")
            
        # Verify cache is ephemeral
        cache_dir = os.environ.get("HF_DATASETS_CACHE", None)
        if cache_dir and not cache_dir.startswith(("/tmp", "/Volumes/RAMDisk")):
            logger.warning(f"Cache directory {cache_dir} may not be ephemeral!")
    
    def load(self) -> IterableDataset:
        """
        Load dataset in streaming mode.
        
        Returns:
            IterableDataset for streaming iteration
        """
        logger.info(f"Loading dataset: {self.dataset_name} (streaming={self.streaming})")
        
        try:
            dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                streaming=self.streaming,
                **self.kwargs
            )
            
            if not isinstance(dataset, IterableDataset):
                raise ValueError(f"Expected IterableDataset, got {type(dataset)}")
                
            logger.info(f"✓ Loaded {self.dataset_name} in streaming mode")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name}: {e}")
            raise
    
    def stream(self, max_samples: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """
        Stream dataset samples.
        
        Args:
            max_samples: Maximum number of samples to stream (None for unlimited)
            
        Yields:
            Dataset samples as dictionaries
        """
        dataset = self.load()
        
        count = 0
        for sample in dataset:
            if max_samples and count >= max_samples:
                break
                
            yield sample
            count += 1
            
            if count % 1000 == 0:
                logger.debug(f"Streamed {count} samples from {self.dataset_name}")
    
    def verify_no_storage(self):
        """Verify that no dataset files were materialized."""
        cache_dir = os.environ.get("HF_DATASETS_CACHE", "")
        
        if cache_dir:
            # Check for arrow/parquet files (materialized datasets)
            import glob
            arrow_files = glob.glob(os.path.join(cache_dir, "**", "*.arrow"), recursive=True)
            parquet_files = glob.glob(os.path.join(cache_dir, "**", "*.parquet"), recursive=True)
            
            if arrow_files or parquet_files:
                logger.warning(f"Found materialized dataset files in cache!")
                logger.warning(f"  Arrow files: {len(arrow_files)}")
                logger.warning(f"  Parquet files: {len(parquet_files)}")
                return False
        
        return True


class MultiDatasetStreamer:
    """Stream multiple datasets with ratio-based mixing."""
    
    def __init__(self, dataset_configs: List[Dict[str, Any]]):
        """
        Initialize multi-dataset streamer.
        
        Args:
            dataset_configs: List of dataset configs, each with:
                - name: Dataset identifier
                - ratio: Sampling ratio (0.0-1.0)
                - split: Dataset split
                - kwargs: Additional load_dataset arguments
        """
        self.dataset_configs = dataset_configs
        self.loaders = {}
        
        # Normalize ratios
        total_ratio = sum(cfg.get("ratio", 0.0) for cfg in dataset_configs)
        if abs(total_ratio - 1.0) > 0.01:
            logger.warning(f"Ratios sum to {total_ratio}, normalizing...")
            for cfg in dataset_configs:
                cfg["ratio"] = cfg.get("ratio", 0.0) / total_ratio
        
        # Initialize loaders
        for cfg in dataset_configs:
            name = cfg["name"]
            self.loaders[name] = StreamingDataLoader(
                dataset_name=name,
                split=cfg.get("split", "train"),
                streaming=True,
                **cfg.get("kwargs", {})
            )
    
    def stream_mixed(self, max_samples: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """
        Stream mixed dataset samples according to ratios.
        
        Uses round-robin with ratio-based sampling.
        
        Args:
            max_samples: Maximum total samples to stream
            
        Yields:
            Mixed dataset samples
        """
        import random
        
        # Create iterators for each dataset
        iterators = {
            name: loader.stream(max_samples=None)
            for name, loader in self.loaders.items()
        }
        
        # Build ratio-based sampling schedule
        schedule = []
        for cfg in self.dataset_configs:
            name = cfg["name"]
            ratio = cfg["ratio"]
            # Add dataset name to schedule proportional to ratio
            schedule.extend([name] * int(ratio * 100))
        
        random.shuffle(schedule)
        schedule_iter = iter(schedule * (max_samples // len(schedule) + 1))
        
        count = 0
        for dataset_name in schedule_iter:
            if max_samples and count >= max_samples:
                break
            
            try:
                sample = next(iterators[dataset_name])
                sample["_dataset_source"] = dataset_name  # Tag source
                yield sample
                count += 1
            except StopIteration:
                # Dataset exhausted, remove from schedule
                logger.warning(f"Dataset {dataset_name} exhausted")
                if dataset_name in iterators:
                    del iterators[dataset_name]
                if not iterators:
                    break
                    
            if count % 1000 == 0:
                logger.debug(f"Streamed {count} mixed samples")


def load_commitpack_stream(max_samples: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """
    Load CommitPack dataset in streaming mode.
    
    Args:
        max_samples: Maximum samples to stream
        
    Yields:
        CommitPack samples with commit_before, commit_message, commit_after
    """
    loader = StreamingDataLoader(
        dataset_name="bigcode/commitpack-subset-cf",
        split="train",
        streaming=True
    )
    
    return loader.stream(max_samples=max_samples)

