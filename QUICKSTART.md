# F6 StreamTrain - Quick Start Guide

## Overview

F6 StreamTrain is a production-grade training pipeline for building frontier vision-language models based on `janhq/Jan-v2-VL-high`. It implements a Phi-4-style staged training pipeline with comprehensive safeguards against catastrophic loss.

## Prerequisites

- Python 3.9+
- PyTorch 2.1+
- CUDA-capable GPU (or Apple Silicon for MLX)
- HuggingFace account (for model access)

## Installation

```bash
# Clone or navigate to project directory
cd /Users/marcpierne/Desktop/f6_model

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

## Quick Start

### 1. Bootstrap Environment (Stage 0)

```bash
python main.py --stage 0
```

This sets up:
- Ephemeral cache directories (no dataset storage)
- HuggingFace streaming mode
- Environment variables for no-storage guarantee

### 2. Run Mid-Training (Stage 1)

```bash
python main.py --stage 1 --config configs/stage1_midtrain.yaml
```

This stage:
- Loads `janhq/Jan-v2-VL-high` as base model
- Applies LoRA for efficient training
- Streams CommitPack and other datasets
- Applies KL divergence regularization
- Monitors anchor regression
- Prevents catastrophic loss

### 3. Monitor Training

Training logs will show:
- Loss values with safeguard checks
- Anchor regression results
- Gradient norms
- Checkpoint saves

Watch for alerts:
- Loss spikes
- Anchor regression failures
- Gradient anomalies

### 4. Resume Training

If training is interrupted:

```bash
python main.py --stage 1 --config configs/stage1_midtrain.yaml --resume checkpoints/stage1_midtrain/checkpoint-500
```

## Configuration

### Key Configuration Files

- `configs/base.yaml` - Base configuration
- `configs/stage1_midtrain.yaml` - Mid-training config
- `configs/stage2_sft.yaml` - SFT config
- `configs/stage3_dpo.yaml` - DPO config
- `configs/stage4_rlvr.yaml` - RLVR config
- `configs/stage5_inftool.yaml` - InfTool loop config

### Important Settings

**Safeguards:**
```yaml
safeguards:
  loss_spike_threshold: 2.0  # Alert if loss spikes 2x
  anomaly_threshold: 10.0    # Max loss value
  anchor_regression_threshold: 0.05  # Max 5% degradation
```

**Training:**
```yaml
training:
  use_lora: true              # LoRA for efficiency
  kl_lambda: 0.1              # KL divergence weight
  max_grad_norm: 1.0          # Gradient clipping
```

**Data:**
```yaml
data:
  streaming: true              # No storage guarantee
  commitpack_ratio: 0.60      # 60% CommitPack data
```

## Safeguards

The pipeline includes multiple safeguard mechanisms:

1. **Catastrophic Loss Prevention**
   - Loss spike detection
   - Anomaly detection
   - Automatic checkpointing
   - Rollback capability

2. **Anchor Regression**
   - Immutable test prompts
   - Performance monitoring
   - Drift detection

3. **Drift Control**
   - KL divergence regularization
   - Base model comparison

4. **Gradient Safety**
   - Gradient clipping
   - NaN/Inf detection
   - Norm monitoring

## Data Streaming

All data is streamed (no storage):
- CommitPack (code-flow learning)
- Long-context repo snapshots
- Synthetic tool traces

Cache is ephemeral (cleaned on exit).

## Model Variants

After training, you'll have:
- **F6-StreamTrain-Instruct**: Concise, policy-compliant
- **F6-StreamTrain-Thinking**: Deeper reasoning traces

## MLX Conversion (Mac)

```bash
# Convert to MLX
python mlx/export_mlx_vlm.py --model-path checkpoints/stage1_midtrain/final

# Quantize to 4-bit
python mlx/quantize_4bit.py --mlx-path Jan-v2-VL-high-bf16-mlx
```

## Troubleshooting

### Out of Memory
- Reduce `micro_batch_size` in config
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing`

### Loss Spikes
- Check data quality
- Reduce learning rate
- Increase KL lambda

### Anchor Regression Failures
- Reduce learning rate
- Increase anchor lambda
- Check base model compatibility

## Next Steps

1. Complete Stage 1 training
2. Run Stage 2 (SFT)
3. Run Stage 3 (DPO)
4. Run Stage 4 (RLVR)
5. Run Stage 5 (InfTool loop)

## Support

See `PROGRESS.md` for implementation status and `README.md` for detailed documentation.

