---
license: apache-2.0
language:
- en
base_model:
- Qwen/Qwen3-VL-8B-Thinking
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- agent
---
# Jan-v2-VL: Multimodal Agent for Long-Horizon Tasks

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/janhq/jan) 
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://opensource.org/licenses/Apache-2.0)
[![Jan App](https://img.shields.io/badge/Powered%20by-Jan%20App-purple?style=flat&logo=android)](https://jan.ai/) 

![image/gif](demo.gif)

## Overview

**Jan-v2-VL** is an 8B-parameter vision–language model for long-horizon, multi-step tasks in real software environments (e.g., browsers and desktop apps). It combines language reasoning with visual perception to follow complex instructions, maintain intermediate state, and recover from minor execution errors.

We recognize the importance of **long-horizon execution** for real-world tasks, where small per-step gains compound into much longer successful chains—so **Jan-v2-VL** is built for stable, many-step execution. For evaluation, we use **[The Illusion of Diminishing Returns: Measuring Long-Horizon Execution in LLMs](https://arxiv.org/pdf/2509.09677)**, which measures execution length. This benchmark aligns with public consensus on what makes a strong coding model—steady, low-drift step execution—suggesting that robust long-horizon ability closely tracks better user experience.

**Variants**

* **Jan-v2-VL-low** — efficiency-oriented, lower latency
* **Jan-v2-VL-med** — balanced latency/quality
* **Jan-v2-VL-high** — deeper reasoning; higher think time

### Intended Use
Tasks where the plan and/or knowledge can be provided up front, and success hinges on stable, many-step execution with minimal drift:

* **Agentic automation & UI control:** Stepwise operation in browsers/desktop apps with screenshot grounding and tool calls (e.g., BrowserMCP).

## Model Performance

![image](https://cdn-uploads.huggingface.co/production/uploads/655e3b59d5c0d3db5359ca3c/bruqlcVK87KMQE99JsS0c.png)

Compared with its base (**[Qwen-3-VL-8B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking)**), **Jan-v2-VL** shows **no degradation** on standard text-only and vision tasks—and is **slightly better on several**—while delivering stronger long-horizon execution on the *Illusion of Diminishing Returns* benchmark.

![image](https://cdn-uploads.huggingface.co/production/uploads/655e3b59d5c0d3db5359ca3c/q4DzuOjmcZOik2c8ZQSCN.png)

![image](https://cdn-uploads.huggingface.co/production/uploads/655e3b59d5c0d3db5359ca3c/JdA1kFh2IEJesQsOAOTrh.png)

![image](https://cdn-uploads.huggingface.co/production/uploads/655e3b59d5c0d3db5359ca3c/fuuZ5pMOGsbbEpKCM5xy8.png)

## Local Deployment

### Integration with Jan App

Jan-v2-VL is optimized for direct integration with the [Jan App](https://jan.ai/). Simply select the model from the Jan App interface for immediate access to its full capabilities.

### Local Deployment

**Using vLLM:**
```bash
vllm serve Menlo/Jan-v2-VL-high \
    --host 0.0.0.0 \
    --port 1234 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 
    
```

**Using llama.cpp:**
```bash
llama-server --model Jan-v2-VL-high-Q8_0.gguf \
    --vision-model-path mmproj-Jan-v2-VL-high.gguf \
    --host 0.0.0.0 \
    --port 1234 \
    --jinja \
    --no-context-shift
```

### Recommended Parameters
For optimal performance in agentic and general tasks, we recommend the following inference parameters:
```yaml
temperature: 1.0
top_p: 0.95
top_k: 20
repetition_penalty: 1.0
presence_penalty: 1.5
```

## 🤝 Community & Support

- **Discussions**: [Hugging Face Community](https://huggingface.co/janhq/Jan-v2-VL-8B/discussions) 
- **Jan App**: Learn more about the Jan App at [jan.ai](https://jan.ai/)

## 📄 Citation
```bibtex
Updated Soon
```