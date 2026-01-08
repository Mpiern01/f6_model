# F6 StreamTrain: Frontier Model Fine-Tuning Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-grade streaming training pipeline for frontier model fine-tuning with zero-storage guarantees.**

## 🎯 Overview

F6 StreamTrain is a specialized training pipeline based on `janhq/Jan-v2-VL-high` (Qwen3VL) that excels at:
- **Zero-storage streaming**: All data streaming, no dataset materialization
- **Long-horizon execution**: Real software environments (browsers/desktop)
- **Secure tool use**: MCP/function calling, codebase edits, test/build loops
- **Repository evolution**: Learning from commit history ("code-flow")
- **Mac deployment**: MLX conversion and 4-bit quantization

## ✨ Key Features

### 🚀 Zero-Storage Architecture
- **Ephemeral caching**: All HuggingFace data in `/tmp` with automatic cleanup
- **Streaming-first**: Enforces `streaming=True` for all dataset operations
- **No materialization**: Prevents dataset storage to disk
- **Memory efficient**: Mac-optimized with CPU training support

### 📊 Comprehensive Dataset Coverage
- **200+ datasets** across all modalities
- **Priority system**: CRITICAL, HEAVY, MEDIUM, LIGHT categories
- **Dataset groups**: Code evolution, reasoning chains, multimodal, etc.
- **Ratio-based mixing**: Configurable dataset sampling

### 🛡️ Production Safeguards
- **Catastrophic loss prevention**: Real-time loss spike detection and rollback
- **Drift control**: KL divergence monitoring vs. base model
- **Anchor regression**: Periodic evaluation on held-out anchor set
- **Gradient safety**: Gradient clipping, NaN/Inf detection, automatic recovery

### 🎓 Multi-Stage Training
- **Stage 0**: Environment bootstrap (ephemeral cache setup)
- **Stage 1**: Mid-training (continued pretraining with safeguards)
- **Stage 2**: Supervised fine-tuning (SFT)
- **Stage 3**: Direct Preference Optimization (DPO)
- **Stage 4**: Infinite tool use (long-horizon execution)

### 🍎 Mac-First Design
- **CPU training**: Optimized for Apple Silicon
- **Float32 precision**: No FP16 requirements
- **Small batch sizes**: Memory-efficient training
- **MLX export**: Native Mac deployment ready

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Mpiern01/f6_model.git
cd f6_model

# Install dependencies
pip install -r requirements.txt
```

### Test Streaming (No Model Required)

```bash
# Test streaming infrastructure
python test_streaming.py
```

Expected output:
```
✓ Environment bootstrap complete
✓ Loaded QuixiAI/dolphin-coder in streaming mode
✓ Successfully fetched 3 samples
```

### Run Training

```bash
# Quick 3-step test
python main.py --stage 1 --config configs/quick_test.yaml

# Full mid-training
python main.py --stage 1 --config configs/stage1_midtrain.yaml

# Supervised fine-tuning
python main.py --stage 2 --config configs/stage2_sft.yaml

# DPO alignment
python main.py --stage 3 --config configs/stage3_dpo.yaml
```

## 🏗️ Architecture

### Zero-Storage Design

```
┌─────────────────────────────────────┐
│  HuggingFace Datasets (Streaming)   │
└──────────────┬──────────────────────┘
               │ streaming=True
               ▼
┌─────────────────────────────────────┐
│  Ephemeral Cache (/tmp/f6_*)        │
│  - Auto-cleanup on exit             │
│  - No persistent storage            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  MultiDatasetStreamer               │
│  - Ratio-based mixing               │
│  - Round-robin sampling             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Training Loop                      │
│  - Safeguards active                │
│  - Checkpoints only                 │
└─────────────────────────────────────┘
```

### Project Structure

```
f6_model/
├── main.py                          # Entry point
├── test_streaming.py                # Streaming test (no model required)
├── stages/
│   ├── s0_env_bootstrap.py         # Environment setup ✅
│   ├── s1_midtrain.py              # Mid-training ✅
│   ├── s2_sft.py                   # Supervised fine-tuning
│   └── s3_dpo.py                   # DPO alignment
├── streaming/
│   ├── hf_stream.py                # Streaming infrastructure ✅
│   ├── dataset_groups.py           # Dataset grouping ✅
│   └── formats/                    # Format handlers
├── dataset_registry/
│   ├── registry.py                 # Main registry (200+ datasets) ✅
│   └── additional_datasets.py      # Extended datasets ✅
├── safeguards/
│   ├── catastrophic_loss.py        # Loss monitoring ✅
│   ├── drift_control.py            # KL divergence ✅
│   ├── anchor_regression.py        # Anchor evaluation ✅
│   └── gradient_safety.py          # Gradient monitoring ✅
├── configs/
│   ├── __init__.py                 # Config loader ✅
│   ├── quick_test.yaml             # 3-step test ✅
│   └── stage1_midtrain.yaml        # Full mid-training
└── docs/
    ├── IMPLEMENTATION_SUMMARY.md   # Implementation notes
    ├── DATASET_REVIEW_2026.md      # Dataset catalog
    └── ENGINEERING_REPORT_2026.md  # Technical deep-dive
```

## 📊 Dataset Coverage

- **Code**: 50+ datasets (CommitPack, StarCoder, CodeParrot, etc.)
- **Math**: 20+ datasets (GSM8K, MATH, MathInstruct, etc.)
- **Reasoning**: 15+ datasets (ARC, HellaSwag, PIQA, etc.)
- **Vision**: 30+ datasets (COCO, Visual Genome, LAION, etc.)
- **Audio**: 10+ datasets (Common Voice, LibriSpeech, etc.)
- **Multimodal**: 25+ datasets (LLaVA, BLIP, Flamingo, etc.)
- **Long-Context**: 10+ datasets (LongBench, InfiniteBench, etc.)

See [DATASET_REVIEW_2026.md](DATASET_REVIEW_2026.md) for the complete list.

## 🧪 Testing

```bash
# Test streaming (fast, no model required)
python test_streaming.py

# Test with quick config (3 steps)
python main.py --stage 1 --config configs/quick_test.yaml

# Full test suite
pytest tests/
```

## 📖 Documentation

- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Current implementation status
- [Dataset Review](DATASET_REVIEW_2026.md) - Complete dataset catalog
- [Engineering Report](ENGINEERING_REPORT_2026.md) - Technical architecture
- [Testing Guide](TESTING_GUIDE.md) - Comprehensive testing instructions

## 🔧 Configuration Example

`configs/quick_test.yaml`:

```yaml
model:
  base_model: "models/Jan-v2-VL-high"
  model_type: "qwen3_vl"
  max_seq_length: 512

training:
  use_lora: true
  max_steps: 3
  micro_batch_size: 1
  learning_rate: 2e-5

data:
  streaming: true
  datasets:
    - name: "QuixiAI/dolphin-coder"
      ratio: 1.0
      split: "train"
      streaming: true

safeguards:
  enabled: true
  loss_spike_threshold: 3.0
```

## ✅ Implementation Status

- ✅ **Environment Bootstrap**: Ephemeral cache, streaming enforcement
- ✅ **Streaming Infrastructure**: MultiDatasetStreamer, verified working
- ✅ **Dataset Registry**: 200+ datasets, priority system
- ✅ **Model Support**: Qwen3VL with trust_remote_code
- ✅ **Safeguards**: Loss prevention, drift control, anchor regression
- ⏳ **Full Training**: Model loading optimization needed for Mac CPU

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **HuggingFace** for datasets and transformers
- **Qwen Team** for Qwen3VL base model
- **Jan.ai** for Jan-v2-VL-high
- Research papers: "The Illusion of Diminishing Returns", "Direct Preference Optimization"

## 📞 Contact

- **GitHub**: [@Mpiern01](https://github.com/Mpiern01)
- **Issues**: [GitHub Issues](https://github.com/Mpiern01/f6_model/issues)

---

**Status**: ✅ Production-ready streaming infrastructure with comprehensive safeguards

**Last Updated**: January 2026

