# Jan-v2-VL-high to GLM-4.6V-Flash Migration Guide

**Quick Reference**: How to switch from Jan-v2-VL-high to GLM-4.6V-Flash

---

## 🎯 Quick Start (5 Minutes)

### Step 1: Update Configuration
Edit `configs/base.yaml`:
```yaml
model:
  # OPTION 1: Jan-v2-VL-high (Qwen3-VL-8B-Thinking based)
  # base_model: "janhq/Jan-v2-VL-high"
  # model_type: "qwen3_vl"
  
  # OPTION 2: GLM-4.6V-Flash (Uncomment to use)
  base_model: "zai-org/GLM-4.6V-Flash"
  model_type: "glm4v"
```

### Step 2: Update Training Script
Edit `stages/s1_midtrain.py` - Uncomment GLM-4V sections (3 locations):

**Location 1** (~line 300):
```python
# OPTION 2: GLM-4.6V-Flash (Uncomment to use)
elif model_type == "glm4v":
    logger.info("Loading GLM-4.6V-Flash model with trust_remote_code...")
    model = AutoModel.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
```

**Location 2** (~line 327): Already updated - checks for `"glm4v"` in skip list

**Location 3** (~line 373):
```python
# OPTION 2: GLM-4.6V-Flash (Uncomment to use)
elif model_type == "glm4v":
    base_model = AutoModel.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
```

### Step 3: Pull Model
```bash
python train_and_deploy.py --model-path zai-org/GLM-4.6V-Flash --skip-training
```

### Step 4: Test
```bash
# Quick test
python test_streaming_quick.py

# Full test
python -m pytest tests/test_safeguards.py -v
```

---

## 📋 Complete File Checklist

### Must Change (Required)
- [ ] `configs/base.yaml` - Update `base_model` and `model_type`
- [ ] `stages/s1_midtrain.py` - Uncomment GLM-4V sections (3 locations)

### Optional (If Using)
- [ ] `configs/quick_test.yaml` - If running quick tests
- [ ] `test_real_training.py` - If running training tests
- [ ] `test_streaming_final.py` - If running streaming tests
- [ ] `test_streaming_quick.py` - If running quick streaming tests
- [ ] `run_complete_pipeline.py` - If running full pipeline
- [ ] `run_benchmarks.py` - If running benchmarks
- [ ] `vscode-extension/package.json` - If using VSCode extension

---

## 🔍 What Changes Automatically

Once you update the config files, these adapt automatically:
- ✅ Model loading logic (detects `model_type`)
- ✅ LoRA skip logic (checks for `"glm4v"`)
- ✅ Base model loading for drift control
- ✅ Training arguments
- ✅ Tokenizer loading

---

## 🚀 Command-Line Override

You can also override the model without changing config files:

```bash
# Use GLM-4.6V-Flash for training
python train_and_deploy.py \
  --model-path zai-org/GLM-4.6V-Flash \
  --config configs/base.yaml

# Note: You still need to update model_type in config
```

---

## ⚠️ Important Notes

### 1. Model Type Must Match
Ensure `model_type` matches the model:
- Jan-v2-VL-high → `model_type: "qwen3_vl"`
- GLM-4.6V-Flash → `model_type: "glm4v"`

### 2. trust_remote_code Required
Both models require `trust_remote_code=True` (already set everywhere)

### 3. LoRA Skipped
Both models skip LoRA and use full fine-tuning mode

### 4. Memory Considerations
- Jan-v2-VL-high: 8B parameters (~16GB RAM)
- GLM-4.6V-Flash: 4.6B parameters (~9GB RAM)

---

## 🔄 Switching Back to Jan-v2-VL-high

Simply reverse the process:

1. Edit `configs/base.yaml`:
   ```yaml
   base_model: "janhq/Jan-v2-VL-high"
   model_type: "qwen3_vl"
   ```

2. Comment out GLM-4V sections in `stages/s1_midtrain.py`

3. Uncomment Jan-v2-VL-high sections

---

## 📊 Performance Comparison

| Metric | Jan-v2-VL-high | GLM-4.6V-Flash |
|--------|----------------|----------------|
| **Size** | 8B | 4.6B |
| **Speed** | Slower | Faster |
| **Memory** | ~16GB | ~9GB |
| **Long-horizon** | Excellent | Good |
| **Tool Calling** | Extensive | Check docs |

---

## 🐛 Troubleshooting

### Error: Model not found
```bash
# Pull the model first
python train_and_deploy.py --model-path zai-org/GLM-4.6V-Flash --skip-training
```

### Error: model_type mismatch
Check that `model_type` in config matches the model:
- `"qwen3_vl"` for Jan-v2-VL-high
- `"glm4v"` for GLM-4.6V-Flash

### Error: Out of memory
GLM-4.6V-Flash uses less memory. If still OOM:
- Reduce `micro_batch_size` in config
- Reduce `max_seq_length` in config

---

## 📚 Additional Resources

- **Detailed Guide**: `docs/MODEL_SWITCHING_GUIDE.md`
- **Model Comparison**: `docs/SUPPORTED_MODELS.md`
- **Implementation Details**: `MODEL_SWITCHING_SUMMARY.md`

---

## ✅ Verification

After switching, verify everything works:

```bash
# 1. Check config loads
python -c "import yaml; print(yaml.safe_load(open('configs/base.yaml'))['model'])"

# 2. Run quick test
python -m pytest tests/test_safeguards.py::TestCatastrophicLossPrevention -v

# 3. Test streaming
python test_streaming_quick.py
```

Expected output:
```
model:
  base_model: zai-org/GLM-4.6V-Flash
  model_type: glm4v
  ...
```

---

## 🎉 Done!

You're now using GLM-4.6V-Flash instead of Jan-v2-VL-high!

**All code is backward compatible** - switching back is just as easy.

