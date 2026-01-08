"""
Dataset Tracking System
Tracks dataset usage, samples seen, and training progress per dataset group

MIT-level engineering: Production-grade tracking
"""

import json
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetTracker:
    """
    Tracks dataset usage during training.
    
    Groups datasets by category/priority and tracks:
    - Samples seen per dataset
    - Training steps per dataset
    - Loss per dataset group
    - Quality metrics
    """
    
    def __init__(self, tracking_dir: str = "tracking"):
        """
        Initialize dataset tracker.
        
        Args:
            tracking_dir: Directory for tracking files
        """
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(exist_ok=True)
        
        # Tracking data
        self.dataset_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "samples_seen": 0,
            "steps": 0,
            "losses": [],
            "last_seen": None,
            "category": None,
            "priority": None
        })
        
        self.group_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_samples": 0,
            "total_steps": 0,
            "avg_loss": 0.0,
            "datasets": []
        })
        
        # Load existing tracking
        self._load_tracking()
    
    def track_sample(
        self,
        dataset_name: str,
        category: str,
        priority: str,
        loss: Optional[float] = None
    ):
        """
        Track a sample from a dataset.
        
        Args:
            dataset_name: Dataset identifier
            category: Dataset category
            priority: Dataset priority
            loss: Optional loss value
        """
        stats = self.dataset_stats[dataset_name]
        stats["samples_seen"] += 1
        stats["last_seen"] = datetime.now().isoformat()
        stats["category"] = category
        stats["priority"] = priority
        
        if loss is not None:
            stats["losses"].append(loss)
            # Keep only last 1000 losses
            if len(stats["losses"]) > 1000:
                stats["losses"] = stats["losses"][-1000:]
        
        # Update group stats
        group_key = f"{category}_{priority}"
        group_stats = self.group_stats[group_key]
        group_stats["total_samples"] += 1
        group_stats["datasets"] = list(set(group_stats["datasets"] + [dataset_name]))
        
        if loss is not None:
            # Update average loss
            all_losses = []
            for ds in group_stats["datasets"]:
                all_losses.extend(self.dataset_stats[ds]["losses"])
            if all_losses:
                group_stats["avg_loss"] = sum(all_losses) / len(all_losses)
    
    def track_step(self, dataset_name: str):
        """Track a training step for a dataset."""
        self.dataset_stats[dataset_name]["steps"] += 1
        group_key = f"{self.dataset_stats[dataset_name]['category']}_{self.dataset_stats[dataset_name]['priority']}"
        self.group_stats[group_key]["total_steps"] += 1
    
    def get_dataset_stats(self, dataset_name: str) -> Dict[str, Any]:
        """Get statistics for a dataset."""
        stats = self.dataset_stats[dataset_name].copy()
        if stats["losses"]:
            stats["avg_loss"] = sum(stats["losses"]) / len(stats["losses"])
            stats["min_loss"] = min(stats["losses"])
            stats["max_loss"] = max(stats["losses"])
        return stats
    
    def get_group_stats(self, category: str, priority: str) -> Dict[str, Any]:
        """Get statistics for a dataset group."""
        group_key = f"{category}_{priority}"
        return self.group_stats[group_key].copy()
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all tracking statistics."""
        return {
            "datasets": {k: self.get_dataset_stats(k) for k in self.dataset_stats},
            "groups": dict(self.group_stats),
            "summary": self._generate_summary()
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        total_samples = sum(s["samples_seen"] for s in self.dataset_stats.values())
        total_steps = sum(s["steps"] for s in self.dataset_stats.values())
        
        by_category = defaultdict(int)
        by_priority = defaultdict(int)
        
        for stats in self.dataset_stats.values():
            by_category[stats["category"]] += stats["samples_seen"]
            by_priority[stats["priority"]] += stats["samples_seen"]
        
        return {
            "total_samples": total_samples,
            "total_steps": total_steps,
            "unique_datasets": len(self.dataset_stats),
            "by_category": dict(by_category),
            "by_priority": dict(by_priority)
        }
    
    def save_tracking(self, filename: Optional[str] = None):
        """Save tracking data to file."""
        if filename is None:
            filename = f"tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.tracking_dir / filename
        with open(filepath, "w") as f:
            json.dump(self.get_all_stats(), f, indent=2)
        
        logger.info(f"Saved tracking data to {filepath}")
    
    def _load_tracking(self):
        """Load existing tracking data."""
        # Find latest tracking file
        tracking_files = sorted(self.tracking_dir.glob("tracking_*.json"), reverse=True)
        if tracking_files:
            try:
                with open(tracking_files[0], "r") as f:
                    data = json.load(f)
                    
                    # Restore dataset stats
                    for name, stats in data.get("datasets", {}).items():
                        self.dataset_stats[name] = stats
                    
                    # Restore group stats
                    for key, stats in data.get("groups", {}).items():
                        self.group_stats[key] = stats
                    
                    logger.info(f"Loaded tracking data from {tracking_files[0]}")
            except Exception as e:
                logger.warning(f"Failed to load tracking: {e}")

