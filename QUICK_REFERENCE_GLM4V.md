# GLM-4.6V-Flash Quick Reference Card

**One-page reference for switching to GLM-4.6V-Flash**

---

## 🎯 2-Step Switch

### Step 1: Edit `configs/base.yaml`
```yaml
# Comment out Jan-v2-VL-high
# base_model: "janhq/Jan-v2-VL-high"
# model_type: "qwen3_vl"

# Uncomment GLM-4.6V-Flash
base_model: "zai-org/GLM-4.6V-Flash"
model_type: "glm4v"
```

### Step 2: Edit `stages/s1_midtrain.py`
Uncomment 3 sections:

**Line ~301-309**: Model loading
```python
# Uncomment this:
elif model_type == "glm4v":
    logger.info("Loading GLM-4.6V-Flash model...")
    model = AutoModel.from_pretrained(...)
```

**Line ~327**: Already includes `"glm4v"` ✅

**Line ~373-380**: Base model for drift
```python
# Uncomment this:
elif model_type == "glm4v":
    base_model = AutoModel.from_pretrained(...)
```

---

## 📋 All Files with GLM-4V Support

| File | Lines | What to Change |
|------|-------|----------------|
| `configs/base.yaml` | 5-11 | Uncomment GLM-4V option |
| `configs/quick_test.yaml` | 5-11 | Uncomment GLM-4V option |
| `stages/s1_midtrain.py` | 301-309 | Uncomment model loading |
| `stages/s1_midtrain.py` | 327 | Already includes `"glm4v"` ✅ |
| `stages/s1_midtrain.py` | 373-380 | Uncomment base model |
| `test_real_training.py` | 28-31 | Uncomment model path |
| `test_streaming_final.py` | 42-46 | Uncomment model path |
| `test_streaming_quick.py` | 42-46 | Uncomment model path |
| `run_complete_pipeline.py` | 84-98 | Uncomment model path |
| `run_benchmarks.py` | 77-81 | Uncomment model path |

---

## 🚀 Commands

### Pull Model
```bash
python train_and_deploy.py --model-path zai-org/GLM-4.6V-Flash --skip-training
```

### Train
```bash
python train_and_deploy.py --model-path zai-org/GLM-4.6V-Flash --config configs/base.yaml
```

### Test
```bash
python test_streaming_quick.py
```

---

## 📊 Model Comparison

| Feature | Jan-v2-VL-high | GLM-4.6V-Flash |
|---------|----------------|----------------|
| **Size** | 8B | 4.6B |
| **Speed** | Slower | Faster |
| **Memory** | ~16GB | ~9GB |
| **HF Path** | `janhq/Jan-v2-VL-high` | `zai-org/GLM-4.6V-Flash` |
| **model_type** | `qwen3_vl` | `glm4v` |

---

## 🔍 Verification

### Check Config
```bash
python -c "import yaml; print(yaml.safe_load(open('configs/base.yaml'))['model'])"
```

Expected output:
```
{'base_model': 'zai-org/GLM-4.6V-Flash', 'model_type': 'glm4v', ...}
```

### Run Tests
```bash
pytest tests/test_safeguards.py -v
```

---

## 🐛 Troubleshooting

### Error: Model not found
```bash
python train_and_deploy.py --model-path zai-org/GLM-4.6V-Flash --skip-training
```

### Error: model_type mismatch
Check that `model_type: "glm4v"` in config

### Error: Out of memory
Reduce `micro_batch_size` in config

---

## 📚 Full Documentation

- **Detailed Guide**: `docs/MODEL_SWITCHING_GUIDE.md`
- **Model Specs**: `docs/SUPPORTED_MODELS.md`
- **Migration Guide**: `JANV2_TO_GLM4V_MIGRATION.md`
- **Audit Report**: `AUDIT_REPORT_GLM4V_SUPPORT.md`
- **Complete Summary**: `COMPLETE_AUDIT_SUMMARY.md`

---

## ✅ Checklist

- [ ] Edit `configs/base.yaml` (uncomment GLM-4V)
- [ ] Edit `stages/s1_midtrain.py` (uncomment 3 sections)
- [ ] Pull model: `python train_and_deploy.py --model-path zai-org/GLM-4.6V-Flash`
- [ ] Test: `python test_streaming_quick.py`

---

**That's it! You're now using GLM-4.6V-Flash!**

