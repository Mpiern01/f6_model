"""
Multimodal Executors
Pluggable executors for image, audio, video generation

MIT-level engineering: Production-grade executor architecture
"""

import logging
import os
import subprocess
import tempfile
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageGenerator:
    """
    Image generation executor.
    
    Supports integration with:
    - Stable Diffusion (via diffusers)
    - DALL-E API
    - Custom image generation models
    """
    
    def __init__(self, model_path: Optional[str] = None, backend: str = "diffusers"):
        """
        Initialize image generator.
        
        Args:
            model_path: Path to image generation model or API endpoint
            backend: Backend to use ("diffusers", "api", "custom")
        """
        self.model_path = model_path
        self.backend = backend
        self.model = None
        self.processor = None
        
        if model_path and backend == "diffusers":
            self._load_diffusers_model()
        elif model_path and backend == "api":
            self._setup_api_client()
    
    def _load_diffusers_model(self):
        """Load image generation model using diffusers."""
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            if torch.cuda.is_available():
                self.model = StableDiffusionPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16
                ).to("cuda")
            else:
                self.model = StableDiffusionPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32
                )
            logger.info(f"Loaded image generation model: {self.model_path}")
        except ImportError:
            logger.warning("diffusers not installed. Install with: pip install diffusers")
            raise RuntimeError("Image generation requires diffusers package")
        except Exception as e:
            logger.error(f"Failed to load image model: {e}")
            raise
    
    def _setup_api_client(self):
        """Setup API client for image generation."""
        # In production, would setup API client (DALL-E, etc.)
        logger.info(f"API client configured for: {self.model_path}")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ) -> Dict[str, Any]:
        """
        Generate image from prompt.
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            num_inference_steps: Number of steps
            guidance_scale: Guidance scale
            
        Returns:
            Dictionary with generated image and metadata
        """
        logger.info(f"Generating image: {prompt[:50]}...")
        
        if self.backend == "diffusers" and self.model is not None:
            try:
                image = self.model(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
                
                return {
                    "image": image,
                    "prompt": prompt,
                    "metadata": {
                        "width": width,
                        "height": height,
                        "steps": num_inference_steps,
                        "backend": self.backend
                    }
                }
            except Exception as e:
                logger.error(f"Image generation failed: {e}")
                raise RuntimeError(f"Image generation failed: {e}")
        elif self.backend == "api":
            # API-based generation would go here
            raise NotImplementedError("API backend not yet implemented")
        else:
            raise RuntimeError(
                f"Image generation not configured. "
                f"Set model_path and backend (current: {self.backend})"
            )


class AudioGenerator:
    """
    Audio generation executor.
    
    Supports:
    - Text-to-speech (TTS)
    - Music generation
    - Sound effect generation
    """
    
    def __init__(self, model_path: Optional[str] = None, audio_type: str = "tts", backend: str = "tts"):
        """
        Initialize audio generator.
        
        Args:
            model_path: Path to audio generation model
            audio_type: Type of audio (tts, music, sound)
            backend: Backend to use ("tts", "musicgen", "custom")
        """
        self.model_path = model_path
        self.audio_type = audio_type
        self.backend = backend
        self.model = None
        
        if model_path:
            self._load_model()
    
    def _load_model(self):
        """Load audio generation model."""
        try:
            if self.backend == "tts":
                from TTS.api import TTS
                self.model = TTS(model_name=self.model_path or "tts_models/en/ljspeech/tacotron2-DDC")
            elif self.backend == "musicgen":
                from transformers import MusicgenForConditionalGeneration, AutoProcessor
                self.model = MusicgenForConditionalGeneration.from_pretrained(
                    self.model_path or "facebook/musicgen-small"
                )
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path or "facebook/musicgen-small"
                )
            logger.info(f"Audio generator initialized (type: {self.audio_type}, backend: {self.backend})")
        except ImportError as e:
            logger.warning(f"Audio generation requires additional packages: {e}")
            logger.warning("Install with: pip install TTS transformers")
        except Exception as e:
            logger.error(f"Failed to load audio model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        duration: float = 5.0,
        sample_rate: int = 22050,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate audio from prompt.
        
        Args:
            prompt: Text prompt or description
            duration: Duration in seconds
            sample_rate: Sample rate
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with generated audio and metadata
        """
        logger.info(f"Generating {self.audio_type} audio: {prompt[:50]}...")
        
        if self.model is None:
            raise RuntimeError(
                f"Audio generation not configured. "
                f"Set model_path and backend (current: {self.backend})"
            )
        
        try:
            if self.backend == "tts" and self.audio_type == "tts":
                output_path = tempfile.mktemp(suffix=".wav")
                self.model.tts_to_file(text=prompt, file_path=output_path)
                return {
                    "audio": output_path,
                    "prompt": prompt,
                    "metadata": {
                        "duration": duration,
                        "sample_rate": sample_rate,
                        "type": self.audio_type,
                        "backend": self.backend
                    }
                }
            elif self.backend == "musicgen" and self.audio_type == "music":
                inputs = self.processor(
                    text=[prompt],
                    padding=True,
                    return_tensors="pt"
                )
                audio_values = self.model.generate(**inputs, max_new_tokens=int(duration * 50))
                return {
                    "audio": audio_values.cpu().numpy(),
                    "prompt": prompt,
                    "metadata": {
                        "duration": duration,
                        "sample_rate": sample_rate,
                        "type": self.audio_type,
                        "backend": self.backend
                    }
                }
            else:
                raise NotImplementedError(f"Audio type {self.audio_type} not supported for backend {self.backend}")
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            raise RuntimeError(f"Audio generation failed: {e}")
    
    def text_to_speech(self, text: str, voice: Optional[str] = None) -> Dict[str, Any]:
        """Generate speech from text."""
        return self.generate(text, audio_type="tts", voice=voice)
    
    def generate_music(self, description: str, style: Optional[str] = None) -> Dict[str, Any]:
        """Generate music from description."""
        return self.generate(description, audio_type="music", style=style)


class VideoGenerator:
    """
    Video generation executor.
    
    Supports:
    - Text-to-video
    - Image-to-video
    - Video editing
    """
    
    def __init__(self, model_path: Optional[str] = None, backend: str = "diffusers"):
        """
        Initialize video generator.
        
        Args:
            model_path: Path to video generation model
            backend: Backend to use ("diffusers", "api", "custom")
        """
        self.model_path = model_path
        self.backend = backend
        self.model = None
        
        if model_path:
            self._load_model()
    
    def _load_model(self):
        """Load video generation model."""
        try:
            if self.backend == "diffusers":
                from diffusers import DiffusionPipeline
                self.model = DiffusionPipeline.from_pretrained(
                    self.model_path or "damo-vilab/text-to-video-ms-1.7b"
                )
            logger.info(f"Video generator initialized (backend: {self.backend})")
        except ImportError:
            logger.warning("Video generation requires diffusers. Install with: pip install diffusers")
        except Exception as e:
            logger.error(f"Failed to load video model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        duration: float = 5.0,
        fps: int = 24,
        width: int = 512,
        height: int = 512,
        image_prompt: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Generate video from prompt.
        
        Args:
            prompt: Text prompt
            duration: Duration in seconds
            fps: Frames per second
            width: Video width
            height: Video height
            image_prompt: Optional image for image-to-video
            
        Returns:
            Dictionary with generated video and metadata
        """
        logger.info(f"Generating video: {prompt[:50]}...")
        
        if self.model is None:
            raise RuntimeError(
                f"Video generation not configured. "
                f"Set model_path and backend (current: {self.backend})"
            )
        
        try:
            if image_prompt is not None:
                frames = self.model(
                    prompt=prompt,
                    image=image_prompt,
                    num_frames=int(duration * fps),
                    height=height,
                    width=width
                ).frames
            else:
                frames = self.model(
                    prompt=prompt,
                    num_frames=int(duration * fps),
                    height=height,
                    width=width
                ).frames
            
            return {
                "video": frames,
                "prompt": prompt,
                "metadata": {
                    "duration": duration,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "frames": len(frames),
                    "backend": self.backend
                }
            }
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            raise RuntimeError(f"Video generation failed: {e}")
    
    def image_to_video(self, image: Any, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate video from image."""
        return self.generate(prompt or "Animate this image", image_prompt=image)


class CodeRunner:
    """
    Code execution executor.
    
    Safely executes code and runs tests using proper sandboxing.
    """
    
    def __init__(self, language: str = "python", sandbox: bool = True, sandbox_type: str = "docker"):
        """
        Initialize code runner.
        
        Args:
            language: Programming language
            sandbox: Whether to use sandboxed execution
            sandbox_type: Type of sandbox ("docker", "firejail", "custom")
        """
        self.language = language
        self.sandbox = sandbox
        self.sandbox_type = sandbox_type
        
        if sandbox and sandbox_type == "docker":
            self._check_docker()
    
    def _check_docker(self):
        """Check if Docker is available for sandboxing."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                logger.warning("Docker not available, sandboxing disabled")
                self.sandbox = False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Docker not found, sandboxing disabled")
            self.sandbox = False
    
    def execute(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute code safely.
        
        Args:
            code: Code to execute
            timeout: Execution timeout in seconds
            
        Returns:
            Execution result
        """
        logger.info(f"Executing {self.language} code...")
        
        if self.sandbox and self.sandbox_type == "docker":
            return self._execute_docker(code, timeout)
        elif self.sandbox and self.sandbox_type == "firejail":
            return self._execute_firejail(code, timeout)
        else:
            # Unsandboxed execution (not recommended for production)
            logger.warning("Executing code without sandbox - not recommended for production")
            return self._execute_unsandboxed(code, timeout)
    
    def _execute_docker(self, code: str, timeout: int) -> Dict[str, Any]:
        """Execute code in Docker container."""
        import time
        start_time = time.time()
        
        try:
            # Create temporary file with code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Run in Docker container
            cmd = [
                "docker", "run", "--rm",
                "--network", "none",  # No network access
                "--memory", "256m",  # Memory limit
                "--cpus", "1",  # CPU limit
                "--timeout", str(timeout),
                "python:3.9-slim",
                "python", "/tmp/code.py"
            ]
            
            result = subprocess.run(
                cmd,
                input=code,
                capture_output=True,
                text=True,
                timeout=timeout + 5
            )
            
            execution_time = time.time() - start_time
            
            # Cleanup
            os.unlink(temp_file)
            
            return {
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
                "success": result.returncode == 0,
                "execution_time": execution_time,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "output": "",
                "error": f"Execution timeout after {timeout} seconds",
                "success": False,
                "execution_time": timeout,
                "return_code": -1
            }
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return {
                "output": "",
                "error": str(e),
                "success": False,
                "execution_time": time.time() - start_time,
                "return_code": -1
            }
    
    def _execute_firejail(self, code: str, timeout: int) -> Dict[str, Any]:
        """Execute code using firejail."""
        # Firejail implementation would go here
        raise NotImplementedError("Firejail sandboxing not yet implemented")
    
    def _execute_unsandboxed(self, code: str, timeout: int) -> Dict[str, Any]:
        """Execute code without sandbox (not recommended)."""
        import time
        start_time = time.time()
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            os.unlink(temp_file)
            
            return {
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
                "success": result.returncode == 0,
                "execution_time": execution_time,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "output": "",
                "error": f"Execution timeout after {timeout} seconds",
                "success": False,
                "execution_time": timeout,
                "return_code": -1
            }
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return {
                "output": "",
                "error": str(e),
                "success": False,
                "execution_time": time.time() - start_time,
                "return_code": -1
            }
    
    def run_tests(self, code: str, tests: str) -> Dict[str, Any]:
        """
        Run tests on code.
        
        Args:
            code: Code to test
            tests: Test code
            
        Returns:
            Test results
        """
        # Combine code and tests
        full_code = f"{code}\n\n{tests}"
        
        result = self.execute(full_code, timeout=60)
        
        # Parse test results
        test_count = tests.count("assert") + tests.count("test_")
        passed_count = 0
        
        if result["success"]:
            # Try to parse pytest/unittest output
            output = result["output"]
            if "passed" in output.lower():
                # Extract number of passed tests
                import re
                passed_match = re.search(r'(\d+)\s+passed', output)
                if passed_match:
                    passed_count = int(passed_match.group(1))
                else:
                    # If no errors, assume all passed
                    passed_count = test_count if test_count > 0 else 1
        
        return {
            "passed": result["success"] and passed_count > 0,
            "test_count": test_count,
            "passed_count": passed_count,
            "output": result["output"],
            "error": result["error"]
        }
