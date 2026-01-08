# Dataset Integration Guide

## Overview

F6 StreamTrain now supports **150+ datasets** with intelligent mixing, priority-based sampling, and special handling for "<- ALL" datasets.

## Key Features

### 1. Priority-Based Mixing
- **ALL**: Use all samples (marked with `<- ALL`)
- **HEAVY**: Use heavy portion (high-quality datasets)
- **MEDIUM**: Use medium portion
- **LIGHT**: Use light portion

### 2. Intelligent Mixing
- Automatic ratio distribution (40% ALL, 30% HEAVY, 20% MEDIUM, 10% LIGHT)
- Weight-based sampling within priority groups
- Round-robin streaming for balanced distribution

### 3. Category-Based Selection
- **code**: Code generation, programming
- **math**: Mathematics, reasoning
- **reasoning**: General reasoning, thinking
- **swe**: Software engineering
- **vision**: Vision, multimodal
- **tool_use**: Tool calling, function calling
- **long_horizon**: Long-horizon execution

## ALL Datasets (Use All Samples)

These datasets are marked for full usage:

1. **AM DeepSeek Distilled** (40M, 1.4M)
2. **Dolphin** (r1, coder)
3. **Microsoft rStar-Coder**
4. **NuminaMath-CoT**
5. **TeichAI High-Quality** (15+ datasets)
6. **SWE-bench Trajectories** (GPT-5-2, Claude Opus, Gemini 3 Pro)
7. **UML Generator**
8. **Golang Coder**

## Usage

### Automatic Dataset Selection

The pipeline automatically selects datasets based on training stage:

```python
from streaming.dataset_manager import DatasetManager
import yaml

# Load config
with open("configs/datasets.yaml", "r") as f:
    config = yaml.safe_load(f)

manager = DatasetManager(config)

# Stream for stage
for sample in manager.stream_for_stage("stage1_midtrain", max_samples=1000):
    print(sample)
```

### Manual Dataset Selection

```python
from datasets.registry import get_datasets_by_priority, DatasetPriority

# Get ALL priority datasets
all_datasets = get_datasets_by_priority(DatasetPriority.ALL)

# Get by category
from datasets.registry import get_datasets_by_category
code_datasets = get_datasets_by_category("code")
```

### Dataset Evaluation

```python
from datasets.evaluator import DatasetEvaluator

evaluator = DatasetEvaluator()
result = evaluator.evaluate_dataset("TeichAI/MiniMax-M2.1-8800x", sample_size=100)
print(result["recommendation"])  # ALL, HEAVY, MEDIUM, or LIGHT
```

## Configuration

Edit `configs/datasets.yaml` to:
- Adjust mixing ratios
- Configure stage-specific selections
- Set quality thresholds
- Enable/disable auto-evaluation

## Streaming Guarantee

✅ **All datasets stream without storage**
✅ **ALL priority datasets use full stream**
✅ **Intelligent mixing preserves ratios**
✅ **Ephemeral cache only**

## Next Steps

1. **Test Streaming**: Verify datasets stream correctly
2. **Evaluate Quality**: Run evaluation on key datasets
3. **Adjust Ratios**: Tune mixing ratios based on results
4. **Monitor Training**: Track which datasets contribute most

---

**Ready to test with 150+ datasets!** 🚀

