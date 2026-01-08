# Supported Models - F6 StreamTrain

F6 StreamTrain supports multiple vision-language models. This document provides a comprehensive reference.

---

## Currently Supported Models

### 1. Jan-v2-VL-high (Default)

**HuggingFace**: [`janhq/Jan-v2-VL-high`](https://huggingface.co/janhq/Jan-v2-VL-high)

#### Specifications
- **Base Model**: Qwen3-VL-8B-Thinking
- **Parameters**: 8B
- **Architecture**: Qwen3VL (vision-language)
- **Context Length**: 32K tokens
- **Vision**: Image + Video support
- **License**: Apache 2.0

#### Strengths
- ✅ Long-horizon task execution
- ✅ Agentic automation & UI control
- ✅ Tool calling with extensive special tokens
- ✅ Stepwise operation in browsers/desktop apps
- ✅ Screenshot grounding
- ✅ Minimal drift on multi-step tasks

#### Use Cases
- Agentic workflows
- Browser automation
- Desktop app control
- Multi-step reasoning
- Tool-calling tasks

#### Configuration
```yaml
model:
  base_model: "janhq/Jan-v2-VL-high"
  model_type: "qwen3_vl"
```

#### Special Tokens
- `<tool_call>`, `</tool_call>`
- `<tool_response>`, `</tool_response>`
- `<think>`, `</think>`
- `<|vision_start|>`, `<|vision_end|>`
- `<|image_pad|>`, `<|video_pad|>`
- And more (see `models/Jan-v2-VL-high/added_tokens.json`)

---

### 2. GLM-4.6V-Flash (Alternative)

**HuggingFace**: [`zai-org/GLM-4.6V-Flash`](https://huggingface.co/zai-org/GLM-4.6V-Flash)

#### Specifications
- **Base Model**: GLM-4V
- **Parameters**: ~4.6B
- **Architecture**: GLM-4V (vision-language)
- **Context Length**: Check model card
- **Vision**: Image support
- **License**: Check model card

#### Strengths
- ✅ Fast inference (smaller model)
- ✅ Efficient vision-language understanding
- ✅ Lower memory footprint
- ✅ Good for resource-constrained environments

#### Use Cases
- Fast vision-language inference
- Resource-constrained training
- Quick prototyping
- Efficient deployment

#### Configuration
```yaml
model:
  base_model: "zai-org/GLM-4.6V-Flash"
  model_type: "glm4v"
```

---

## Model Comparison

| Feature | Jan-v2-VL-high | GLM-4.6V-Flash |
|---------|----------------|----------------|
| **Parameters** | 8B | ~4.6B |
| **Base** | Qwen3-VL-8B-Thinking | GLM-4V |
| **Context** | 32K | Check model card |
| **Vision** | Image + Video | Image |
| **Tool Calling** | Extensive | Check model card |
| **Memory** | Higher | Lower |
| **Speed** | Slower | Faster |
| **Long-horizon** | Excellent | Good |
| **License** | Apache 2.0 | Check model card |

---

## Switching Between Models

See [`docs/MODEL_SWITCHING_GUIDE.md`](MODEL_SWITCHING_GUIDE.md) for detailed instructions.

### Quick Switch
1. Edit `configs/base.yaml`
2. Uncomment desired model
3. Comment out other model
4. Update `model_type` accordingly

---

## Model-Specific Implementation Details

### Both Models
- **trust_remote_code**: Required (`trust_remote_code=True`)
- **LoRA**: Skipped (full fine-tuning mode)
- **dtype**: float32 on Mac, float16 on GPU
- **device_map**: "cpu" on Mac, "auto" on GPU

### Jan-v2-VL-high Specific
```python
if model_type == "qwen3_vl":
    model = AutoModel.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
```

### GLM-4.6V-Flash Specific
```python
elif model_type == "glm4v":
    model = AutoModel.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
```

---

## Adding New Models

To add support for a new vision-language model:

1. **Add to configs**
   - Add new option in `configs/base.yaml`
   - Define new `model_type` (e.g., `"llava"`, `"cogvlm"`)

2. **Update training scripts**
   - Add model loading logic in `stages/s1_midtrain.py`
   - Add to LoRA skip list if needed
   - Add base model loading for drift control

3. **Update test scripts**
   - Add model path option in test files

4. **Update documentation**
   - Add to this file
   - Update `MODEL_SWITCHING_GUIDE.md`

5. **Test thoroughly**
   - Run quick test: `python test_streaming_quick.py`
   - Run full test suite: `pytest tests/ -v`

---

## Recommended Model Selection

### Choose Jan-v2-VL-high if:
- ✅ You need long-horizon task execution
- ✅ You're building agentic workflows
- ✅ You need extensive tool calling
- ✅ You have sufficient compute resources
- ✅ You need video understanding

### Choose GLM-4.6V-Flash if:
- ✅ You need fast inference
- ✅ You have limited compute resources
- ✅ You're prototyping quickly
- ✅ You only need image understanding
- ✅ You want lower memory usage

---

## Performance Benchmarks

### Jan-v2-VL-high
- **Illusion of Diminishing Returns**: Strong performance
- **Standard benchmarks**: No degradation vs base model
- **Long-horizon tasks**: Excellent

### GLM-4.6V-Flash
- Check model card for official benchmarks

---

## Resources

### Jan-v2-VL-high
- **Model Card**: https://huggingface.co/janhq/Jan-v2-VL-high
- **GitHub**: https://github.com/janhq/jan
- **Website**: https://jan.ai/

### GLM-4.6V-Flash
- **Model Card**: https://huggingface.co/zai-org/GLM-4.6V-Flash

---

## Support

For issues or questions:
1. Check `docs/MODEL_SWITCHING_GUIDE.md`
2. Check model-specific documentation on HuggingFace
3. Review `MODEL_SWITCHING_SUMMARY.md` for implementation details

---

**Last Updated**: January 7, 2026

