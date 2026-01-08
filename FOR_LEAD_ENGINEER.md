# F6 StreamTrain: Complete System Documentation
## For Lead Engineer

**Project**: F6 StreamTrain - Frontier Model-Fusion Training Pipeline  
**Base Model**: `janhq/Jan-v2-VL-high` (8B parameter vision-language model)  
**Objective**: Train, convert to MLX, and 4-bit quantize for Mac deployment

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Directory Structure](#directory-structure)
3. [Model Paths and Standardization](#model-paths-and-standardization)
4. [Complete Pipeline Execution](#complete-pipeline-execution)
5. [Training Stages](#training-stages)
6. [MLX Conversion and Quantization](#mlx-conversion-and-quantization)
7. [Configuration Files](#configuration-files)
8. [Troubleshooting](#troubleshooting)
9. [File Path Reference](#file-path-reference)

---

## System Overview

F6 StreamTrain is a production-grade training pipeline that:

1. **Pulls** the base model from HuggingFace (`janhq/Jan-v2-VL-high`)
2. **Trains** the model through 5 stages (mid-training → SFT → DPO → RLVR → InfTool)
3. **Converts** trained model to MLX format for Apple Silicon
4. **Quantizes** MLX model to 4-bit using advanced methods (QuaRot, BitNet v2, AMXFP4)

### Key Features

- **No-storage guarantee**: All datasets streamed, no local storage
- **Catastrophic loss prevention**: Multi-layer safeguards (KL divergence, anchor regression, gradient clipping)
- **Mac-first deployment**: MLX conversion and 4-bit quantization
- **Production-ready**: MIT-level engineering, comprehensive error handling

---

## Directory Structure

```
/Users/marcpierne/Desktop/f6_model/
├── models/                          # Model storage
│   └── Jan-v2-VL-high/             # Base model (standardized path)
│
├── checkpoints/                     # Training checkpoints
│   ├── stage1_midtrain/            # Stage 1 checkpoints
│   │   ├── checkpoint-500/         # Individual checkpoints
│   │   └── final/                  # Final trained model
│   ├── stage2_sft/
│   ├── stage3_dpo/
│   ├── stage4_rlvr/
│   └── stage5_inftool/
│
├── configs/                         # Configuration files
│   ├── base.yaml                   # Base configuration
│   ├── stage1_midtrain.yaml        # Stage 1 config
│   ├── stage2_sft.yaml
│   ├── stage3_dpo.yaml
│   ├── stage4_rlvr.yaml
│   ├── stage5_inftool.yaml
│   └── datasets.yaml                # Dataset configuration
│
├── stages/                         # Training stage implementations
│   ├── s0_env_bootstrap.py         # Environment setup
│   ├── s1_midtrain.py              # Stage 1: Mid-training
│   ├── s2_sft.py                   # Stage 2: Supervised Fine-Tuning
│   ├── s3_rollout_dpo.py           # Stage 3: Rollout DPO
│   ├── s4_rlvr_grpo.py             # Stage 4: RL with Verifiable Reward
│   └── s5_inftool_loop.py           # Stage 5: InfTool closed-loop
│
├── mlx/                            # MLX conversion and quantization
│   ├── export_mlx_vlm.py           # Convert to MLX format
│   └── quantize_4bit.py            # 4-bit quantization
│
├── train_and_deploy.py             # Main pipeline script
├── run_complete_pipeline.py        # Automated pipeline runner
├── main.py                         # Stage-by-stage execution
│
├── streaming/                      # Data streaming infrastructure
├── safeguards/                     # Catastrophic loss prevention
├── verifiers/                      # Test/build verification
├── benchmarks/                    # Evaluation suites
└── agentic/                        # MCP/A2A protocols
```

---

## Model Paths and Standardization

### Standardized Model Path

**All model references use**: `models/Jan-v2-VL-high/`

This is the **single source of truth** for the base model. The training pipeline:
- Downloads to this path if model doesn't exist locally
- Uses this path for training initialization
- Falls back to HuggingFace path (`janhq/Jan-v2-VL-high`) if local path unavailable

### Path Resolution Logic

1. **Training Stage** (`stages/s1_midtrain.py`):
   - Reads `config["model"]["base_model"]` from YAML
   - Default: `"janhq/Jan-v2-VL-high"` (HuggingFace path)
   - If local path exists: `models/Jan-v2-VL-high` → uses local
   - If local path missing: `janhq/Jan-v2-VL-high` → downloads/caches via HuggingFace

2. **Model Pull** (`train_and_deploy.py`):
   - Downloads to: `models/Jan-v2-VL-high/` (standardized)
   - Returns this path for training

3. **MLX Conversion**:
   - Input: Trained model path (e.g., `checkpoints/stage1_midtrain/final`)
   - Output: `{model_path}_mlx` (e.g., `checkpoints/stage1_midtrain/final_mlx`)

4. **Quantization**:
   - Input: MLX model path
   - Output: `{mlx_path}_q4_{method}` (e.g., `final_mlx_q4_quarot`)

### Cleaning Up Duplicate Paths

If you see both:
- `models/Jan-v2-VL-high/` ✓ (correct, standardized)
- `models/janhq_Jan-v2-VL-high/` ✗ (old path, legacy)

**Action**: The system now uses `models/Jan-v2-VL-high/` only. The old path is not referenced anywhere in the codebase.

**To clean up** (after verifying both contain the same model):
```bash
# Verify they're the same model (both should be ~16GB)
du -sh models/*

# If same size and you've verified they're identical, remove old path:
rm -rf models/janhq_Jan-v2-VL-high
```

**Note**: The old path was created by an earlier version that replaced `/` with `_`. The new standardized path uses just the model name.

---

## Complete Pipeline Execution

### Option 1: Automated Pipeline (Recommended)

```bash
cd /Users/marcpierne/Desktop/f6_model
python3 run_complete_pipeline.py
```

This script:
1. Checks/installs dependencies
2. Pulls model to `models/Jan-v2-VL-high/`
3. Trains model (configurable steps)
4. Converts to MLX
5. Quantizes to 4-bit

### Option 2: Manual Step-by-Step

#### Step 1: Bootstrap Environment

```bash
python3 stages/s0_env_bootstrap.py
```

Sets up:
- Ephemeral cache directories (`/tmp/f6_ephemeral_cache`)
- HuggingFace streaming mode
- Environment variables for no-storage guarantee

#### Step 2: Pull Model

```bash
python3 train_and_deploy.py \
    --model-path "janhq/Jan-v2-VL-high" \
    --skip-training \
    --skip-mlx \
    --skip-quantization
```

Downloads model to: `models/Jan-v2-VL-high/`

#### Step 3: Train Model

```bash
# Full training
python3 train_and_deploy.py \
    --model-path "models/Jan-v2-VL-high" \
    --config "configs/stage1_midtrain.yaml"

# Limited steps (for testing)
python3 train_and_deploy.py \
    --model-path "models/Jan-v2-VL-high" \
    --config "configs/stage1_midtrain.yaml" \
    --max-steps 10
```

Output: `checkpoints/stage1_midtrain/final/`

#### Step 4: Convert to MLX

```bash
python3 mlx/export_mlx_vlm.py \
    --model-path "checkpoints/stage1_midtrain/final" \
    --output-path "checkpoints/stage1_midtrain/final_mlx" \
    --dtype bfloat16
```

**Requirements**: `pip install mlx mlx-vlm`

#### Step 5: 4-bit Quantization

```bash
# QuaRot method (recommended)
python3 mlx/quantize_4bit.py \
    --model-path "checkpoints/stage1_midtrain/final_mlx" \
    --output-path "checkpoints/stage1_midtrain/final_mlx_q4_quarot" \
    --method quarot

# Standard method (fallback)
python3 mlx/quantize_4bit.py \
    --model-path "checkpoints/stage1_midtrain/final_mlx" \
    --output-path "checkpoints/stage1_midtrain/final_mlx_q4_standard" \
    --method standard
```

**Available methods**: `quarot`, `bitnet_v2`, `amxfp4`, `standard`

### Option 3: Stage-by-Stage (Advanced)

```bash
# Stage 0: Bootstrap
python3 main.py --stage 0

# Stage 1: Mid-training
python3 main.py --stage 1 --config configs/stage1_midtrain.yaml

# Stage 2: SFT
python3 main.py --stage 2 --config configs/stage2_sft.yaml

# Stage 3: DPO
python3 main.py --stage 3 --config configs/stage3_dpo.yaml

# Stage 4: RLVR
python3 main.py --stage 4 --config configs/stage4_rlvr.yaml

# Stage 5: InfTool
python3 main.py --stage 5 --config configs/stage5_inftool.yaml
```

---

## Training Stages

### Stage 1: Mid-Training (Continued Pretraining)

**File**: `stages/s1_midtrain.py`  
**Config**: `configs/stage1_midtrain.yaml`  
**Output**: `checkpoints/stage1_midtrain/`

**Purpose**: Inject repo-evolution priors without catastrophic drift

**Key Features**:
- LoRA adapters for efficient training
- KL divergence regularization (prevents drift)
- Anchor regression (immutable prompts)
- Streaming datasets (CommitPack, long-context, tool traces)

**Safeguards**:
- Loss spike detection (threshold: 2.0x)
- Anchor evaluation every 100 steps
- Gradient clipping (max_norm: 1.0)
- Checkpoint validation

### Stage 2: Supervised Fine-Tuning (SFT)

**File**: `stages/s2_sft.py`  
**Config**: `configs/stage2_sft.yaml`  
**Output**: `checkpoints/stage2_sft/`

**Purpose**: High-quality instruction following

**Input**: Stage 1 final model  
**Data**: High-quality CoT trajectories, SWE traces

### Stage 3: Rollout DPO

**File**: `stages/s3_rollout_dpo.py`  
**Config**: `configs/stage3_dpo.yaml`  
**Output**: `checkpoints/stage3_dpo/`

**Purpose**: Preference learning from verifier feedback

**Process**:
1. Generate multiple rollouts per prompt
2. Verify each rollout (tests, builds, schemas)
3. Label preferences based on verification
4. Train with DPO loss

### Stage 4: RL with Verifiable Reward (RLVR)

**File**: `stages/s4_rlvr_grpo.py`  
**Config**: `configs/stage4_rlvr.yaml`  
**Output**: `checkpoints/stage4_rlvr/`

**Purpose**: Reinforcement learning with verifiable rewards

**Optimizers**: GRPO, MS-GRPO, INFO-GRPO

### Stage 5: InfTool Closed-Loop

**File**: `stages/s5_inftool_loop.py`  
**Config**: `configs/stage5_inftool.yaml`  
**Output**: `checkpoints/stage5_inftool/`

**Purpose**: Self-improving tool-use data generation

**Process**:
- Multi-agent role-play (User Simulator, Tool Assistant, MCP Server)
- Synthesize tool-use scenarios
- Quality gates (verification, schema validation)
- Closed-loop improvement

---

## MLX Conversion and Quantization

### MLX Conversion

**File**: `mlx/export_mlx_vlm.py`

**Usage**:
```bash
python3 mlx/export_mlx_vlm.py \
    --model-path <trained_model_path> \
    --output-path <mlx_output_path> \
    --dtype bfloat16
```

**Requirements**:
```bash
pip install mlx mlx-vlm
```

**Input**: HuggingFace model (local path or HF identifier)  
**Output**: MLX model directory with:
- `weights.safetensors` or `weights.npz`
- `config.json`
- Tokenizer files

### 4-bit Quantization

**File**: `mlx/quantize_4bit.py`

**Methods**:

1. **QuaRot** (Recommended)
   - Quantization with Rotations
   - Removes outliers before quantization
   - Best quality retention

2. **BitNet v2**
   - Hadamard transformation
   - Native 4-bit activation quantization

3. **AMXFP4**
   - Asymmetric microscaling
   - Better quantization range utilization

4. **Standard**
   - Uses `mlx-lm` quantize function
   - Fallback method

**Usage**:
```bash
python3 mlx/quantize_4bit.py \
    --model-path <mlx_model_path> \
    --output-path <quantized_output_path> \
    --method quarot \
    --bits 4
```

**Output**: Quantized model with:
- `weights.safetensors` (4-bit quantized)
- `quantization_metadata.json` (scales, zeros, method)
- `config.json`

---

## Configuration Files

### Base Configuration (`configs/base.yaml`)

**Model Settings**:
```yaml
model:
  base_model: "janhq/Jan-v2-VL-high"  # Can be local: "models/Jan-v2-VL-high"
  model_type: "qwen3_vl"
  max_seq_length: 32768
  vision_max_length: 2048
```

**Training Settings**:
```yaml
training:
  use_lora: true
  lora_r: 16
  lora_alpha: 32
  gradient_checkpointing: true
  gradient_accumulation_steps: 4
  micro_batch_size: 1
  learning_rate: 2e-5
  kl_lambda: 0.1      # KL divergence weight
  anchor_lambda: 0.05 # Anchor loss weight
```

**Safeguards**:
```yaml
safeguards:
  enable_anomaly_detection: true
  anomaly_threshold: 10.0
  anchor_eval_frequency: 100
  anchor_regression_threshold: 0.05
```

### Stage-Specific Configs

Each stage extends `base.yaml` and overrides:
- Learning rate
- Number of epochs
- Dataset mix ratios
- Stage-specific parameters

---

## Troubleshooting

### Issue: Model Path Not Found

**Error**: `OSError: Model path not found: models/Jan-v2-VL-high`

**Solution**:
```bash
# Pull model explicitly
python3 train_and_deploy.py \
    --model-path "janhq/Jan-v2-VL-high" \
    --skip-training \
    --skip-mlx \
    --skip-quantization
```

The system will download to `models/Jan-v2-VL-high/` automatically.

### Issue: Duplicate Model Directories

**Problem**: Both `models/Jan-v2-VL-high/` and `models/janhq_Jan-v2-VL-high/` exist

**Solution**:
```bash
# Check sizes
du -sh models/*

# Remove duplicate (keep standardized path)
rm -rf models/janhq_Jan-v2-VL-high
```

### Issue: MLX Conversion Fails

**Error**: `ImportError: No module named 'mlx'`

**Solution**:
```bash
pip install mlx mlx-vlm mlx-lm
```

**Note**: MLX requires macOS with Apple Silicon (M1/M2/M3)

### Issue: Training Fails with Out of Memory

**Solution**:
1. Reduce `micro_batch_size` in config (default: 1)
2. Increase `gradient_accumulation_steps` (default: 4)
3. Enable `gradient_checkpointing: true` (default: enabled)

### Issue: Dependencies Missing

**Error**: `ImportError: No module named 'transformers'`

**Solution**:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install torch transformers datasets peft accelerate pyyaml
```

### Issue: HuggingFace Authentication

**Error**: `401 Client Error: Unauthorized`

**Solution**:
```bash
huggingface-cli login
```

Or set token:
```bash
export HF_TOKEN=your_token_here
```

---

## File Path Reference

### Model Paths

| Purpose | Path | Description |
|---------|------|-------------|
| Base Model (Local) | `models/Jan-v2-VL-high/` | Standardized local model path |
| Base Model (HF) | `janhq/Jan-v2-VL-high` | HuggingFace identifier |
| Stage 1 Output | `checkpoints/stage1_midtrain/final/` | Trained model after Stage 1 |
| Stage 2 Output | `checkpoints/stage2_sft/final/` | Trained model after Stage 2 |
| MLX Model | `{model_path}_mlx/` | MLX-converted model |
| Quantized Model | `{mlx_path}_q4_{method}/` | 4-bit quantized model |

### Configuration Paths

| Config | Path | Purpose |
|--------|------|---------|
| Base | `configs/base.yaml` | Base configuration |
| Stage 1 | `configs/stage1_midtrain.yaml` | Mid-training config |
| Stage 2 | `configs/stage2_sft.yaml` | SFT config |
| Stage 3 | `configs/stage3_dpo.yaml` | DPO config |
| Stage 4 | `configs/stage4_rlvr.yaml` | RLVR config |
| Stage 5 | `configs/stage5_inftool.yaml` | InfTool config |
| Datasets | `configs/datasets.yaml` | Dataset configuration |

### Script Paths

| Script | Path | Purpose |
|--------|------|---------|
| Main Pipeline | `train_and_deploy.py` | Complete pipeline |
| Automated Runner | `run_complete_pipeline.py` | Automated execution |
| Stage Runner | `main.py` | Stage-by-stage execution |
| MLX Export | `mlx/export_mlx_vlm.py` | MLX conversion |
| Quantization | `mlx/quantize_4bit.py` | 4-bit quantization |

### Cache Paths

| Cache | Path | Purpose |
|-------|------|---------|
| Ephemeral Cache | `/tmp/f6_ephemeral_cache/` | Temporary dataset cache |
| HF Home | `/tmp/f6_hf_home/` | HuggingFace home |
| Datasets Cache | `/tmp/f6_datasets_cache/` | Dataset cache |

---

## Quick Start Commands

### Full Pipeline (One Command)

```bash
cd /Users/marcpierne/Desktop/f6_model
python3 run_complete_pipeline.py
```

### Manual Pipeline

```bash
# 1. Pull model
python3 train_and_deploy.py --model-path "janhq/Jan-v2-VL-high" --skip-training --skip-mlx --skip-quantization

# 2. Train (10 steps for testing)
python3 train_and_deploy.py --model-path "models/Jan-v2-VL-high" --max-steps 10

# 3. Convert to MLX
python3 mlx/export_mlx_vlm.py --model-path "checkpoints/stage1_midtrain/final" --output-path "checkpoints/stage1_midtrain/final_mlx"

# 4. Quantize
python3 mlx/quantize_4bit.py --model-path "checkpoints/stage1_midtrain/final_mlx" --output-path "checkpoints/stage1_midtrain/final_mlx_q4" --method quarot
```

---

## Summary

**Standardized Model Path**: `models/Jan-v2-VL-high/` (single source of truth)

**Pipeline Flow**:
1. Pull → `models/Jan-v2-VL-high/`
2. Train → `checkpoints/stage1_midtrain/final/`
3. MLX → `checkpoints/stage1_midtrain/final_mlx/`
4. Quantize → `checkpoints/stage1_midtrain/final_mlx_q4_quarot/`

**Key Files**:
- `train_and_deploy.py` - Main pipeline
- `configs/base.yaml` - Base configuration
- `mlx/export_mlx_vlm.py` - MLX conversion
- `mlx/quantize_4bit.py` - Quantization

**All paths are standardized and consistent throughout the system.**

