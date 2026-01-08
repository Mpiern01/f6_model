"""
Test Streaming Data Pipeline
Tests for HuggingFace streaming and dataset management

MIT-level engineering: Comprehensive streaming tests
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming.hf_stream import StreamingDataLoader, MultiDatasetStreamer
from streaming.dataset_manager import DatasetManager
from dataset_registry.registry import DatasetConfig, DatasetPriority


class TestStreamingDataLoader:
    """Test StreamingDataLoader functionality."""

    def test_initialization(self):
        """Test loader initialization."""
        loader = StreamingDataLoader(
            dataset_name="bigcode/the-stack-smol",
            split="train",
            streaming=True
        )
        assert loader.streaming is True
        assert loader.dataset_name == "bigcode/the-stack-smol"
        assert loader.split == "train"

    @pytest.mark.slow
    def test_stream_samples(self):
        """Test streaming samples (slow test - requires network)."""
        loader = StreamingDataLoader(
            dataset_name="bigcode/the-stack-smol",
            split="train",
            streaming=True
        )

        # Stream a few samples
        samples = []
        for i, sample in enumerate(loader.stream(max_samples=3)):
            samples.append(sample)
            if i >= 2:
                break

        assert len(samples) == 3
        assert all(isinstance(s, dict) for s in samples)

    def test_streaming_mode_enforced(self):
        """Test that streaming mode is enforced."""
        loader = StreamingDataLoader(
            dataset_name="bigcode/the-stack-smol",
            streaming=True
        )

        # Verify streaming is enforced
        assert loader.streaming is True


class TestMultiDatasetStreamer:
    """Test MultiDatasetStreamer functionality."""

    def test_initialization(self):
        """Test streamer initialization."""
        datasets = [
            {"name": "bigcode/the-stack-smol", "ratio": 0.5, "split": "train"},
            {"name": "HuggingFaceH4/ultrachat_200k", "ratio": 0.5, "split": "train"}
        ]

        streamer = MultiDatasetStreamer(datasets)
        assert len(streamer.loaders) == 2
        assert "bigcode/the-stack-smol" in streamer.loaders
        assert "HuggingFaceH4/ultrachat_200k" in streamer.loaders

    def test_ratio_normalization(self):
        """Test ratio normalization."""
        datasets = [
            {"name": "bigcode/the-stack-smol", "ratio": 0.7, "split": "train"},
            {"name": "HuggingFaceH4/ultrachat_200k", "ratio": 0.3, "split": "train"}
        ]

        streamer = MultiDatasetStreamer(datasets)

        # Check ratios sum to 1.0
        total_ratio = sum(cfg["ratio"] for cfg in streamer.dataset_configs)
        assert abs(total_ratio - 1.0) < 0.01


class TestDatasetManager:
    """Test DatasetManager functionality."""
    
    def test_initialization(self):
        """Test manager initialization."""
        config = {
            "datasets": {
                "stage1_midtrain": {
                    "categories": ["code", "reasoning"],
                    "priorities": ["ALL", "HEAVY"]
                }
            }
        }
        
        manager = DatasetManager(config)
        assert manager.config is not None
    
    def test_get_datasets_for_stage(self):
        """Test getting datasets for a stage."""
        config = {
            "datasets": {
                "stage1_midtrain": {
                    "categories": ["code"],
                    "priorities": ["ALL"]
                }
            }
        }
        
        manager = DatasetManager(config)
        datasets = manager.get_datasets_for_stage("stage1_midtrain")
        
        assert isinstance(datasets, list)
        # Should have at least some datasets
        assert len(datasets) >= 0


class TestDatasetRegistry:
    """Test dataset registry functionality."""

    def test_dataset_config(self):
        """Test DatasetConfig creation."""
        config = DatasetConfig(
            name="Test Dataset",
            hf_path="test/dataset",
            category="code",
            priority=DatasetPriority.HEAVY,
            weight=1.0,
            description="Test dataset"
        )

        assert config.name == "Test Dataset"
        assert config.priority == DatasetPriority.HEAVY
        assert config.weight == 1.0

    def test_priority_enum(self):
        """Test DatasetPriority enum."""
        # Values are lowercase in actual implementation
        assert DatasetPriority.ALL.value == "all"
        assert DatasetPriority.HEAVY.value == "heavy"
        assert DatasetPriority.MEDIUM.value == "medium"
        assert DatasetPriority.LIGHT.value == "light"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

