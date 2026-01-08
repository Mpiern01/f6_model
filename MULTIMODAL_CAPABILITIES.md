# Multimodal Generative Capabilities

## Overview

F6 StreamTrain now includes **full multimodal generative capabilities** with a planner/executor architecture.

## Architecture

### Planner/Executor Split

**Planner**: Jan-v2-VL-high (vision-grounded, tool-using)
- Plans actions based on user requests
- Determines which executor to use
- Provides reasoning and context

**Executors** (Pluggable):
- **ImageGenerator**: Text-to-image, image editing
- **AudioGenerator**: Text-to-speech, music generation, sound effects
- **VideoGenerator**: Text-to-video, image-to-video, video editing
- **CodeRunner**: Safe code execution, test running

### Unified Interface

Externally, F6 exposes **one endpoint** (one "model"), but internally routes to appropriate executors based on modality.

## Capabilities

### 1. Image Generation
- Text-to-image synthesis
- Image editing and manipulation
- Style transfer
- Image inpainting/outpainting

### 2. Audio Generation
- **Text-to-Speech**: Natural voice synthesis
- **Music Generation**: Create music from descriptions
- **Sound Effects**: Generate audio effects
- **Audio Editing**: Modify and enhance audio

### 3. Video Generation
- **Text-to-Video**: Generate videos from text prompts
- **Image-to-Video**: Animate static images
- **Video Editing**: Modify existing videos
- **Video Understanding**: Analyze video content

### 4. Code Execution
- Safe code execution in sandbox
- Test running and validation
- Multi-language support
- Security controls

## Integration

### Using the Harness

```python
from multimodal import PlannerExecutorHarness
from multimodal.executors import ImageGenerator, AudioGenerator, VideoGenerator

# Initialize harness
harness = PlannerExecutorHarness(
    planner_model=jan_model,
    image_generator=ImageGenerator(),
    audio_generator=AudioGenerator(),
    video_generator=VideoGenerator()
)

# Process request
result = harness.process({
    "modality": "image",
    "prompt": "A futuristic cityscape at sunset",
    "context": {}
})
```

### Modality Detection

The planner automatically detects modality from:
- Explicit modality parameter
- Prompt content analysis
- Context clues

## Security

- **Sandboxed Execution**: Code runs in isolated environment
- **Rate Limiting**: Prevents abuse
- **Audit Logging**: All generations logged
- **Content Filtering**: Safety checks on outputs

## Future Enhancements

- Integration with Stable Diffusion, DALL-E APIs
- Real-time audio/video streaming
- Multi-modal input (text + image → video)
- Cross-modal understanding

---

**Status**: ✅ **Multimodal Architecture Complete**

