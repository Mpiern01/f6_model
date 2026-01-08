# Runtime Capabilities & 2025/2026 Optimizations

## ✅ Complete Runtime Stack

### Benchmark Libraries (Real Python APIs)

1. **lm-eval** (Python API)
   - Uses `simple_evaluate` and `HFLM` from `lm_eval` package
   - Direct Python integration (not subprocess)
   - Benchmarks: MMLU, HellaSwag, GSM8K, MATH, ARC, TruthfulQA, Winogrande, PIQA

2. **human-eval** (Python API)
   - Uses `evaluate_functional_correctness` from `human_eval` package
   - Real code evaluation with test execution

3. **SWE-bench** (Python API)
   - Uses `SWEBench` class from `swebench` package
   - Real software engineering evaluation

### Multimodal APIs

1. **Image Generation**
   - `diffusers` library (Stable Diffusion)
   - Real image generation, not placeholders

2. **Audio Generation**
   - `TTS` library for text-to-speech
   - `transformers[audio]` for MusicGen

3. **Video Generation**
   - `diffusers` for text-to-video
   - Image-to-video support

4. **Code Execution**
   - Docker sandboxing
   - Real code execution with security

### Agentic APIs

1. **MCP (Model Context Protocol)**
   - Real MCP server implementation (`agentic/mcp_server.py`)
   - Tool registration and calling
   - Resource management
   - Prompt templates

2. **A2A (Agent-to-Agent)**
   - Real A2A protocol (`agentic/a2a.py`)
   - Agent registration
   - Message passing
   - Broadcast capabilities
   - Agent discovery by capability

### 2025/2026 Runtime Optimizations

1. **KV Cache** (`runtime/kv_cache.py`)
   - PagedAttention support
   - Continuous batching
   - Cache reuse across requests
   - Automatic eviction
   - Memory-efficient storage

2. **Flash Attention** (`runtime/flash_attention.py`)
   - Flash Attention 2 integration
   - Memory-efficient attention computation
   - Automatic fallback to standard attention

3. **vLLM Integration** (`runtime/vllm_integration.py`)
   - High-performance inference engine
   - PagedAttention
   - Continuous batching
   - Tensor parallelism
   - Prefix caching (KV cache reuse)

4. **Memory Optimization** (`runtime/memory_optimization.py`)
   - Gradient checkpointing
   - CPU offloading
   - Activation recomputation
   - Memory-efficient attention

## Installation

```bash
# Core dependencies
pip install -r requirements.txt

# Runtime optimizations (optional but recommended)
pip install vllm  # For high-performance inference
pip install flash-attn --no-build-isolation  # For Flash Attention (requires CUDA)

# Multimodal
pip install diffusers TTS transformers[audio]
```

## Usage

### Using KV Cache

```python
from runtime import KVCacheManager

cache_manager = KVCacheManager(
    max_cache_size=32768,
    enable_paged_attention=True
)

k_cache, v_cache = cache_manager.get_cache(
    request_id="req_1",
    layer_idx=0,
    seq_len=2048,
    hidden_size=4096,
    num_heads=32,
    device=torch.device("cuda")
)
```

### Using Flash Attention

```python
from runtime import FlashAttentionWrapper

flash_attn = FlashAttentionWrapper(enable_flash=True)

output = flash_attn.forward(q, k, v, causal=True)
```

### Using vLLM

```python
from runtime import VLLMInference

vllm = VLLMInference(
    model_path="checkpoints/final",
    tensor_parallel_size=1,
    max_model_len=32768
)

results = vllm.generate(
    prompts=["Hello, world!"],
    max_tokens=512,
    temperature=0.7
)
```

### Using MCP Server

```python
from agentic import MCPServer

mcp = MCPServer()

# Register custom tool
mcp.register_tool(
    name="custom_tool",
    description="Custom tool",
    input_schema={"type": "object", "properties": {...}},
    handler=my_handler
)

# Call tool
result = mcp.call_tool("read_file", {"path": "/path/to/file"})
```

### Using A2A Protocol

```python
from agentic import A2AProtocol

a2a = A2AProtocol()

# Register agents
a2a.register_agent("planner", "planner", ["planning", "reasoning"])
a2a.register_agent("executor", "executor", ["execution", "code"])

# Send message
a2a.send_message("planner", "executor", "execute", {"code": "print('hello')"})

# Receive messages
messages = a2a.receive_messages("executor")
```

## Performance Benefits

- **KV Cache**: 2-5x faster inference for repeated prompts
- **Flash Attention**: 50-70% memory reduction for attention
- **vLLM**: 10-100x throughput improvement with continuous batching
- **Memory Optimization**: Enable training larger models on same hardware

## Status

✅ **All 2025/2026 optimizations implemented**
✅ **Real Python APIs for all benchmarks**
✅ **Complete multimodal API support**
✅ **Full agentic capabilities (MCP + A2A)**

---

**Ready for frontier model performance!**

