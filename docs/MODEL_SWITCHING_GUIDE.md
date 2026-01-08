# Model Switching Guide: Jan-v2-VL-high ↔ GLM-4.6V-Flash

This guide explains how to switch between the two supported vision-language models in the F6 StreamTrain codebase.

---

## Supported Models

### OPTION 1: Jan-v2-VL-high (Default)
- **HuggingFace Path**: `janhq/Jan-v2-VL-high`
- **Base Architecture**: Qwen3-VL-8B-Thinking
- **Parameters**: 8B
- **Strengths**: Long-horizon tasks, agentic automation, tool calling
- **License**: Apache 2.0

### OPTION 2: GLM-4.6V-Flash
- **HuggingFace Path**: `zai-org/GLM-4.6V-Flash`
- **Base Architecture**: GLM-4V
- **Parameters**: ~4.6B
- **Strengths**: Fast inference, efficient vision-language understanding
- **License**: Check HuggingFace model card

---

## How to Switch Models

All code has been annotated with comments showing both options. To switch from Jan-v2-VL-high to GLM-4.6V-Flash:

### 1. Configuration Files

#### `configs/base.yaml`
```yaml
model:
  # OPTION 1: Jan-v2-VL-high (Qwen3-VL-8B-Thinking based)
  # base_model: "janhq/Jan-v2-VL-high"
  # model_type: "qwen3_vl"
  
  # OPTION 2: GLM-4.6V-Flash (Uncomment to use)
  base_model: "zai-org/GLM-4.6V-Flash"
  model_type: "glm4v"
```

#### `configs/quick_test.yaml`
```yaml
model:
  # OPTION 1: Jan-v2-VL-high (Qwen3-VL-8B-Thinking based)
  # base_model: "models/Jan-v2-VL-high"
  # model_type: "qwen3_vl"
  
  # OPTION 2: GLM-4.6V-Flash (Uncomment to use)
  base_model: "zai-org/GLM-4.6V-Flash"
  model_type: "glm4v"
```

### 2. Training Scripts

#### `stages/s1_midtrain.py`
The training script automatically detects the model type from config. Uncomment the GLM-4V sections:

```python
# OPTION 1: Qwen3VL (Jan-v2-VL-high)
# if model_type == "qwen3_vl":
#     logger.info("Loading Qwen3VL model (Jan-v2-VL-high)...")
#     model = AutoModel.from_pretrained(...)

# OPTION 2: GLM-4.6V-Flash (Uncomment to use)
elif model_type == "glm4v":
    logger.info("Loading GLM-4.6V-Flash model...")
    model = AutoModel.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
```

### 3. Test Scripts

#### `test_real_training.py`
```python
# OPTION 1: Jan-v2-VL-high (Qwen3VL)
# model_path = "models/Jan-v2-VL-high"

# OPTION 2: GLM-4.6V-Flash (Uncomment to use)
model_path = "zai-org/GLM-4.6V-Flash"
```

#### `test_streaming_final.py`
```python
# OPTION 1: Jan-v2-VL-high (Qwen3VL)
# model_path = "models/Jan-v2-VL-high"

# OPTION 2: GLM-4.6V-Flash (Uncomment to use)
model_path = "zai-org/GLM-4.6V-Flash"
```

### 4. Pipeline Scripts

#### `train_and_deploy.py`
```bash
# OPTION 1: Jan-v2-VL-high
python train_and_deploy.py --model-path janhq/Jan-v2-VL-high

# OPTION 2: GLM-4.6V-Flash
python train_and_deploy.py --model-path zai-org/GLM-4.6V-Flash
```

#### `run_complete_pipeline.py`
```python
# OPTION 1: Jan-v2-VL-high (Qwen3VL)
# model_path = pull_model("janhq/Jan-v2-VL-high")

# OPTION 2: GLM-4.6V-Flash (Uncomment to use)
model_path = pull_model("zai-org/GLM-4.6V-Flash")
```

### 5. VSCode Extension

#### `vscode-extension/package.json`
Update the default model in settings:
```json
{
  "f6JanAI.model": "GLM-4.6V-Flash"
}
```

Or change in VSCode settings UI:
1. Open VSCode Settings (Cmd+,)
2. Search for "f6JanAI.model"
3. Change to "GLM-4.6V-Flash"

### 6. MLX Export

```bash
# Export Jan-v2-VL-high to MLX
python mlx/export_mlx_vlm.py --model-path janhq/Jan-v2-VL-high --output-path Jan-v2-VL-high-mlx

# Export GLM-4.6V-Flash to MLX
python mlx/export_mlx_vlm.py --model-path zai-org/GLM-4.6V-Flash --output-path GLM-4.6V-Flash-mlx
```

---

## Quick Switch Checklist

To switch from Jan-v2-VL-high to GLM-4.6V-Flash:

- [ ] Update `configs/base.yaml` - change `base_model` and `model_type`
- [ ] Update `configs/quick_test.yaml` - change `base_model` and `model_type`
- [ ] Update test scripts if running tests directly
- [ ] Update VSCode extension settings if using the extension
- [ ] Pull the new model: `python train_and_deploy.py --model-path zai-org/GLM-4.6V-Flash`

---

## Model-Specific Considerations

### Jan-v2-VL-high (Qwen3VL)
- **LoRA**: Skipped (full fine-tuning mode)
- **dtype**: float32 on Mac, float16 on GPU
- **trust_remote_code**: Required
- **Special tokens**: Extensive tool calling tokens

### GLM-4.6V-Flash
- **LoRA**: Skipped (full fine-tuning mode)
- **dtype**: float32 on Mac, float16 on GPU
- **trust_remote_code**: Required
- **Special tokens**: Check model card for specifics

---

## Files Modified with Model Switching Support

All files below have been annotated with OPTION 1 and OPTION 2 comments:

1. **Configuration Files**
   - `configs/base.yaml`
   - `configs/quick_test.yaml`

2. **Training Scripts**
   - `stages/s1_midtrain.py`

3. **Test Scripts**
   - `test_real_training.py`
   - `test_streaming_final.py`
   - `test_streaming_quick.py`

4. **Pipeline Scripts**
   - `train_and_deploy.py`
   - `run_complete_pipeline.py`
   - `run_benchmarks.py`

5. **VSCode Extension**
   - `vscode-extension/package.json`
   - `vscode-extension/src/extension.ts`
   - `vscode-extension/README.md`

6. **MLX Export**
   - `mlx/export_mlx_vlm.py`

7. **Multimodal**
   - `multimodal/planner_executor.py`

---

## Testing After Switch

After switching models, run these tests to verify everything works:

```bash
# 1. Quick test (3 steps)
python -m pytest tests/test_safeguards.py -v

# 2. Streaming test
python test_streaming_quick.py

# 3. Full pipeline test
python run_complete_pipeline.py --skip-training
```

---

## Troubleshooting

### Issue: Model not found
**Solution**: Pull the model first:
```bash
python train_and_deploy.py --model-path zai-org/GLM-4.6V-Flash --skip-training
```

### Issue: trust_remote_code error
**Solution**: Both models require `trust_remote_code=True`. This is already set in all scripts.

### Issue: Out of memory
**Solution**: GLM-4.6V-Flash is smaller (4.6B vs 8B), so it should use less memory. Adjust batch size if needed.

---

**All code is ready to switch between models by simply uncommenting the desired option!**

