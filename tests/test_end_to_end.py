"""
End-to-End Tests
Complete integration tests for the F6 StreamTrain pipeline

MIT-level engineering: Comprehensive E2E testing
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stages.s0_env_bootstrap import EphemeralCacheManager
from streaming.hf_stream import StreamingDataLoader, MultiDatasetStreamer
from dataset_registry.registry import get_all_datasets, DatasetPriority
from safeguards.catastrophic_loss import CatastrophicLossPrevention
from safeguards.gradient_safety import GradientSafety
import torch


class TestEnvironmentBootstrap:
    """Test environment bootstrap end-to-end."""

    def test_ephemeral_cache_setup(self):
        """Test ephemeral cache setup."""
        cache_manager = EphemeralCacheManager()

        # Verify cache directory exists
        assert cache_manager.cache_dir is not None
        assert os.path.exists(cache_manager.cache_dir)

        # Verify cache is in /tmp or /var/folders (macOS temp)
        assert cache_manager.cache_dir.startswith("/tmp") or cache_manager.cache_dir.startswith("/var/folders")

    def test_cache_cleanup(self):
        """Test cache cleanup."""
        cache_manager = EphemeralCacheManager()
        cache_dir = cache_manager.cache_dir

        # Verify cache exists
        assert os.path.exists(cache_dir)

        # Cleanup
        cache_manager.cleanup()

        # Verify cache is removed
        assert not os.path.exists(cache_dir)


class TestStreamingPipeline:
    """Test complete streaming pipeline."""
    
    @pytest.mark.slow
    def test_single_dataset_streaming(self):
        """Test streaming from a single dataset."""
        # Use a small, fast dataset
        loader = StreamingDataLoader(
            dataset_name="QuixiAI/dolphin-coder",
            split="train",
            streaming=True
        )
        
        # Stream a few samples
        samples = list(loader.stream(max_samples=3))
        
        assert len(samples) == 3
        assert all(isinstance(s, dict) for s in samples)
    
    def test_dataset_registry_integration(self):
        """Test dataset registry integration."""
        # Get all datasets
        all_datasets = get_all_datasets()

        assert len(all_datasets) > 0

        # Check structure - datasets are DatasetConfig objects
        for dataset in all_datasets[:5]:  # Check first 5
            assert hasattr(dataset, "name")
            assert hasattr(dataset, "hf_path")
            assert hasattr(dataset, "category")
            assert hasattr(dataset, "priority")


class TestSafeguardsIntegration:
    """Test safeguards integration."""
    
    def test_loss_and_gradient_monitoring(self):
        """Test loss and gradient monitoring together."""
        # Create safeguards
        loss_prevention = CatastrophicLossPrevention()
        gradient_safety = GradientSafety(max_grad_norm=1.0)
        
        # Create dummy model
        model = torch.nn.Linear(10, 10)
        
        # Simulate training step
        # 1. Forward pass (simulated)
        loss = 1.5
        
        # 2. Check loss
        loss_result = loss_prevention.check_loss(loss, step=1)
        assert "safe" in loss_result
        
        # 3. Backward pass (simulated)
        for param in model.parameters():
            param.grad = torch.randn_like(param) * 0.5
        
        # 4. Check gradients
        grad_result = gradient_safety.clip_gradients(model)
        assert "clipped" in grad_result
        
        # 5. Check gradient health
        health = gradient_safety.check_gradient_health(model)
        assert health["healthy"] is True
    
    def test_catastrophic_loss_recovery(self):
        """Test catastrophic loss recovery workflow."""
        loss_prevention = CatastrophicLossPrevention(
            loss_spike_threshold=2.0,
            max_loss_value=10.0
        )
        
        # Normal training
        for i in range(10):
            result = loss_prevention.check_loss(1.0 + i * 0.1, step=i)
            assert result["safe"] is True or result["action"] == "warn"
        
        # Catastrophic loss
        result = loss_prevention.check_loss(100.0, step=11)
        assert result["safe"] is False
        assert result["action"] == "stop"


class TestConfigurationLoading:
    """Test configuration loading."""
    
    def test_load_quick_test_config(self):
        """Test loading quick test configuration."""
        import yaml
        
        config_path = Path(__file__).parent.parent / "configs" / "quick_test.yaml"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert "model" in config
        assert "training" in config
        assert "data" in config
        
        # Check model config
        assert config["model"]["model_type"] == "qwen3_vl"
        
        # Check training config
        assert config["training"]["max_steps"] == 3
        
        # Check data config
        assert config["data"]["streaming"] is True


class TestDatasetGroups:
    """Test dataset grouping functionality."""
    
    def test_get_datasets_by_category(self):
        """Test getting datasets by category."""
        from dataset_registry.registry import get_datasets_by_category

        code_datasets = get_datasets_by_category("code")
        assert len(code_datasets) > 0

        # All should be code category - datasets are DatasetConfig objects
        for ds in code_datasets:
            assert ds.category == "code"

    def test_get_datasets_by_priority(self):
        """Test getting datasets by priority."""
        from dataset_registry.registry import get_datasets_by_priority

        heavy_datasets = get_datasets_by_priority(DatasetPriority.HEAVY)
        assert len(heavy_datasets) > 0

        # All should be HEAVY priority - datasets are DatasetConfig objects
        for ds in heavy_datasets:
            assert ds.priority == DatasetPriority.HEAVY


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])

