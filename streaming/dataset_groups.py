"""
Dataset Groups with Tracking
Groups datasets and tracks training progress per group

MIT-level engineering: Production-grade dataset group tracking
"""

import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict
from streaming.dataset_tracker import DatasetTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetGroup:
    """
    Represents a group of datasets with shared characteristics.
    """
    
    def __init__(
        self,
        name: str,
        datasets: List[str],
        category: str,
        priority: str,
        target_ratio: float = 1.0
    ):
        """
        Initialize dataset group.
        
        Args:
            name: Group name
            datasets: List of dataset identifiers
            category: Dataset category
            priority: Dataset priority
            target_ratio: Target ratio in training mix
        """
        self.name = name
        self.datasets = datasets
        self.category = category
        self.priority = priority
        self.target_ratio = target_ratio
        
        # Tracking
        self.samples_seen = 0
        self.steps_trained = 0
        self.avg_loss = 0.0
        self.losses = []
    
    def track_sample(self, loss: Optional[float] = None):
        """Track a sample from this group."""
        self.samples_seen += 1
        if loss is not None:
            self.losses.append(loss)
            if len(self.losses) > 1000:
                self.losses = self.losses[-1000:]
            self.avg_loss = sum(self.losses) / len(self.losses)
    
    def track_step(self):
        """Track a training step."""
        self.steps_trained += 1
    
    def get_stats(self, total_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Get group statistics.
        
        Args:
            total_samples: Total samples across all groups (for ratio calculation)
        """
        actual_ratio = 0.0
        if total_samples and total_samples > 0:
            actual_ratio = self.samples_seen / total_samples
        
        return {
            "name": self.name,
            "category": self.category,
            "priority": self.priority,
            "num_datasets": len(self.datasets),
            "samples_seen": self.samples_seen,
            "steps_trained": self.steps_trained,
            "avg_loss": self.avg_loss,
            "target_ratio": self.target_ratio,
            "actual_ratio": actual_ratio
        }


class DatasetGroupManager:
    """
    Manages dataset groups and tracks training progress.
    """
    
    def __init__(self, tracker: Optional[DatasetTracker] = None):
        """
        Initialize group manager.
        
        Args:
            tracker: Optional dataset tracker
        """
        self.tracker = tracker or DatasetTracker()
        self.groups: Dict[str, DatasetGroup] = {}
        self._initialize_groups()
    
    def _initialize_groups(self):
        """Initialize dataset groups."""
        from dataset_registry.registry import get_all_datasets, DatasetPriority
        
        all_datasets = get_all_datasets()
        
        # Group by category and priority
        grouped = defaultdict(list)
        for ds in all_datasets:
            key = f"{ds.category}_{ds.priority.value}"
            grouped[key].append(ds.hf_path)
        
        # Create groups
        for key, datasets in grouped.items():
            category, priority = key.split("_")
            group = DatasetGroup(
                name=key,
                datasets=datasets,
                category=category,
                priority=priority,
                target_ratio=self._calculate_target_ratio(priority)
            )
            self.groups[key] = group
        
        logger.info(f"Initialized {len(self.groups)} dataset groups")
    
    def _calculate_target_ratio(self, priority: str) -> float:
        """Calculate target ratio based on priority."""
        ratios = {
            "all": 0.40,
            "heavy": 0.30,
            "medium": 0.20,
            "light": 0.10
        }
        return ratios.get(priority.lower(), 0.10)
    
    def track_sample(self, dataset_name: str, loss: Optional[float] = None):
        """
        Track a sample from a dataset.
        
        Args:
            dataset_name: Dataset identifier
            loss: Optional loss value
        """
        # Find group
        group = self._find_group(dataset_name)
        if group:
            group.track_sample(loss)
        
        # Track in tracker
        if group:
            self.tracker.track_sample(
                dataset_name=dataset_name,
                category=group.category,
                priority=group.priority,
                loss=loss
            )
    
    def track_step(self, dataset_name: str):
        """Track a training step."""
        group = self._find_group(dataset_name)
        if group:
            group.track_step()
            self.tracker.track_step(dataset_name)
    
    def _find_group(self, dataset_name: str) -> Optional[DatasetGroup]:
        """Find group for dataset."""
        for group in self.groups.values():
            if dataset_name in group.datasets:
                return group
        return None
    
    def get_group_stats(self) -> Dict[str, Any]:
        """Get statistics for all groups."""
        total_samples = sum(g.samples_seen for g in self.groups.values())
        return {
            group.name: group.get_stats(total_samples=total_samples)
            for group in self.groups.values()
        }
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get overall training progress."""
        total_samples = sum(g.samples_seen for g in self.groups.values())
        total_steps = sum(g.steps_trained for g in self.groups.values())
        
        return {
            "total_samples": total_samples,
            "total_steps": total_steps,
            "groups": self.get_group_stats(),
            "tracker_stats": self.tracker.get_all_stats()
        }
    
    def save_progress(self, filepath: str):
        """Save training progress."""
        import json
        progress = self.get_training_progress()
        with open(filepath, "w") as f:
            json.dump(progress, f, indent=2)
        logger.info(f"Saved training progress to {filepath}")

