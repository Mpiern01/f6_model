# Real Benchmark Evaluation Frameworks

## Overview

All benchmarks now use **real, official evaluation frameworks** used by frontier models. No custom implementations.

## Evaluation Frameworks Used

### 1. lm-eval-harness
**Official framework for frontier model evaluation**

- **MMLU**: Massive Multitask Language Understanding
- **HellaSwag**: Commonsense reasoning
- **GSM8K**: Math word problems
- **MATH**: Competition math
- **ARC**: AI2 Reasoning Challenge
- **TruthfulQA**: Truthfulness evaluation
- **Winogrande**: Commonsense reasoning
- **PIQA**: Physical reasoning

**Installation:**
```bash
pip install lm-eval
```

**Usage:**
```bash
lm_eval --model hf --model_args pretrained=model_path --tasks mmlu --num_fewshot 5
```

### 2. HumanEval (OpenAI)
**Official code generation evaluation**

- **HumanEval**: 164 Python programming problems
- **MBPP**: Mostly Basic Python Problems

**Installation:**
```bash
pip install human-eval
```

**Usage:**
```python
from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness
```

### 3. SWE-bench (Princeton)
**Official software engineering evaluation**

- **SWE-bench**: Real-world software engineering tasks

**Installation:**
```bash
pip install swebench
# Or from GitHub: https://github.com/princeton-nlp/SWE-bench
```

### 4. CodeXGLUE (Microsoft)
**Official code understanding benchmark**

- **Code-to-Text**: Code summarization
- **Text-to-Code**: Code generation

**Installation:**
```bash
# From Microsoft's repository
```

### 5. Long-Horizon Execution
**From "The Illusion of Diminishing Returns" paper**

- Uses official datasets:
  - `long_horizon_execution`
  - `CVC2233/Long-Horizon-GUI-Dataset`

## Benchmark Results Format

All benchmarks return standardized results:

```python
{
    "score": 0.85,  # Main metric
    "details": {...},  # Full results
    "status": "complete"  # or "failed"
}
```

## Running Benchmarks

```bash
# Run all benchmarks
python run_benchmarks.py --model-path checkpoints/stage5_inftool/final

# Skip long-horizon (faster)
python run_benchmarks.py --model-path checkpoints/stage5_inftool/final --skip-long-horizon
```

## Requirements

All real evaluation frameworks are in `requirements.txt`:

- `lm-eval>=0.4.0` - For frontier benchmarks
- `human-eval>=1.0.0` - For code generation
- `nltk>=3.8.0` - For BLEU scores
- `swebench` - For SWE evaluation (install separately)
- `codexglue` - For code understanding (install separately)

## Notes

- All benchmarks use **official evaluation frameworks**
- No custom implementations
- Results are comparable to published frontier model scores
- Follows standard evaluation protocols

---

**Status**: ✅ **All benchmarks use real evaluation frameworks**

