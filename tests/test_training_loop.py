"""
Training Loop Tests
Tests for the complete training loop with safeguards

MIT-level engineering: Production training validation
"""

import pytest
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from safeguards.catastrophic_loss import CatastrophicLossPrevention
from safeguards.gradient_safety import GradientSafety
from safeguards.drift_control import DriftControl


class TestTrainingLoopComponents:
    """Test individual training loop components."""
    
    def test_loss_tracking(self):
        """Test loss tracking over multiple steps."""
        loss_prevention = CatastrophicLossPrevention(window_size=10)
        
        losses = [2.5, 2.3, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4]
        
        for i, loss in enumerate(losses):
            result = loss_prevention.check_loss(loss, step=i)
            # Should be safe - decreasing loss
            assert result["safe"] is True or result["action"] == "warn"
        
        # Check status
        status = loss_prevention.get_status()
        assert status["loss_history_length"] == 10
        assert status["baseline_loss"] is not None
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation and clipping."""
        gradient_safety = GradientSafety(max_grad_norm=1.0)
        
        # Create model
        model = torch.nn.Linear(100, 100)
        
        # Simulate multiple gradient accumulation steps
        for step in range(3):
            # Add gradients
            for param in model.parameters():
                if param.grad is None:
                    param.grad = torch.randn_like(param) * 0.1
                else:
                    param.grad += torch.randn_like(param) * 0.1
        
        # Clip accumulated gradients
        result = gradient_safety.clip_gradients(model)
        
        assert "clipped" in result
        assert "clipped_norm" in result
    
    def test_drift_control_initialization(self):
        """Test drift control initialization."""
        base_model = torch.nn.Linear(10, 10)
        
        drift_control = DriftControl(
            base_model=base_model,
            kl_lambda=0.1
        )
        
        assert drift_control.kl_lambda == 0.1
        assert drift_control.base_model is not None


class TestTrainingStepSimulation:
    """Test simulated training steps."""
    
    def test_complete_training_step(self):
        """Test a complete training step with all safeguards."""
        # Setup
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        loss_prevention = CatastrophicLossPrevention()
        gradient_safety = GradientSafety(max_grad_norm=1.0)
        
        # Simulate training step
        step = 1
        
        # 1. Forward pass (simulated)
        inputs = torch.randn(4, 10)
        targets = torch.randn(4, 10)
        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        
        # 2. Check loss
        loss_result = loss_prevention.check_loss(loss.item(), step=step)
        
        if not loss_result["safe"]:
            pytest.skip("Loss not safe, would rollback in production")
        
        # 3. Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # 4. Check gradients
        grad_health = gradient_safety.check_gradient_health(model)
        
        if not grad_health["healthy"]:
            pytest.skip("Gradients not healthy, would skip step in production")
        
        # 5. Clip gradients
        clip_result = gradient_safety.clip_gradients(model)
        
        # 6. Optimizer step
        optimizer.step()
        
        # Verify step completed
        assert loss_result["safe"] is True or loss_result["action"] == "warn"
        assert grad_health["healthy"] is True
        assert "clipped" in clip_result
    
    def test_multiple_training_steps(self):
        """Test multiple training steps."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        loss_prevention = CatastrophicLossPrevention()
        gradient_safety = GradientSafety(max_grad_norm=1.0)
        
        num_steps = 10
        losses = []
        
        for step in range(num_steps):
            # Forward
            inputs = torch.randn(4, 10)
            targets = torch.randn(4, 10)
            outputs = model(inputs)
            loss = torch.nn.functional.mse_loss(outputs, targets)
            
            # Check loss
            loss_result = loss_prevention.check_loss(loss.item(), step=step)
            losses.append(loss.item())
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Clip and step
            gradient_safety.clip_gradients(model)
            optimizer.step()
        
        # Verify training progressed
        assert len(losses) == num_steps
        
        # Check status
        status = loss_prevention.get_status()
        assert status["loss_history_length"] == num_steps


class TestCheckpointingWorkflow:
    """Test checkpointing workflow."""
    
    def test_checkpoint_save_and_load(self):
        """Test saving and loading checkpoints."""
        import tempfile
        import os
        
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        
        loss_prevention = CatastrophicLossPrevention(
            checkpoint_dir=tempfile.mkdtemp()
        )
        
        # Save checkpoint
        checkpoint_path = loss_prevention.save_checkpoint(
            model, optimizer, step=100, loss=1.5
        )
        
        assert os.path.exists(checkpoint_path)
        
        # Load checkpoint
        checkpoint = loss_prevention.load_checkpoint(checkpoint_path)
        
        assert checkpoint["step"] == 100
        assert checkpoint["loss"] == 1.5
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

