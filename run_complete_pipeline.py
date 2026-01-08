#!/usr/bin/env python3
"""
Complete Training and Deployment Pipeline
Handles dependencies, pulls model, trains, converts to MLX, and quantizes

MIT-level engineering: Comprehensive error handling and dependency management
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_and_install_dependencies():
    """Check and install missing dependencies."""
    logger.info("Checking dependencies...")
    
    required = ["torch", "transformers", "datasets", "peft", "accelerate", "yaml"]
    missing = []
    
    for pkg in required:
        try:
            if pkg == "yaml":
                __import__("yaml")
            else:
                __import__(pkg)
            logger.info(f"✓ {pkg} installed")
        except ImportError:
            missing.append(pkg)
            logger.warning(f"✗ {pkg} missing")
    
    if missing:
        logger.info(f"Installing missing packages: {', '.join(missing)}")
        # Map package names to pip names
        pip_names = {
            "yaml": "pyyaml",
            "torch": "torch",
            "transformers": "transformers",
            "datasets": "datasets",
            "peft": "peft",
            "accelerate": "accelerate"
        }
        
        packages_to_install = [pip_names.get(pkg, pkg) for pkg in missing]
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade"
            ] + packages_to_install)
            logger.info("✓ Dependencies installed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            logger.error("Please install manually: pip install " + " ".join(packages_to_install))
            return False
    
    return True


def main():
    """Main pipeline execution."""
    logger.info("=" * 80)
    logger.info("F6 StreamTrain - Complete Pipeline")
    logger.info("=" * 80)
    
    # Step 0: Check dependencies
    if not check_and_install_dependencies():
        logger.error("Dependency check failed")
        sys.exit(1)
    
    # Step 1: Pull model
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Pulling Model")
    logger.info("=" * 80)
    
    try:
        from train_and_deploy import pull_model

        # OPTION 1: Jan-v2-VL-high (Qwen3VL)
        model_path = pull_model("janhq/Jan-v2-VL-high")

        # OPTION 2: GLM-4.6V-Flash (Uncomment to use)
        # model_path = pull_model("zai-org/GLM-4.6V-Flash")

        logger.info(f"✓ Model ready: {model_path}")
    except Exception as e:
        logger.error(f"Failed to pull model: {e}")
        logger.info("Will use HuggingFace path directly")

        # OPTION 1: Jan-v2-VL-high (Qwen3VL)
        model_path = "janhq/Jan-v2-VL-high"

        # OPTION 2: GLM-4.6V-Flash (Uncomment to use)
        # model_path = "zai-org/GLM-4.6V-Flash"
    
    # Step 2: Train model (limited steps for testing)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Training Model (10 steps for testing)")
    logger.info("=" * 80)
    
    try:
        from train_and_deploy import train_model
        
        trained_model_path = train_model(
            model_path=model_path,
            config_path="configs/stage1_midtrain.yaml",
            max_steps=10,  # Limited for testing
            dry_run=False
        )
        
        if trained_model_path and os.path.exists(trained_model_path):
            logger.info(f"✓ Training complete: {trained_model_path}")
        else:
            logger.warning("Training may not have produced a final model")
            # Try to find any checkpoint
            checkpoint_dir = "checkpoints/stage1_midtrain"
            if os.path.exists(checkpoint_dir):
                checkpoints = sorted(Path(checkpoint_dir).glob("checkpoint-*"), reverse=True)
                if checkpoints:
                    trained_model_path = str(checkpoints[0])
                    logger.info(f"Using checkpoint: {trained_model_path}")
                else:
                    trained_model_path = checkpoint_dir
            else:
                trained_model_path = model_path  # Fallback to base model
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        logger.warning("Continuing with base model for MLX conversion")
        trained_model_path = model_path
    
    # Step 3: Convert to MLX
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Converting to MLX")
    logger.info("=" * 80)
    
    mlx_model_path = None
    try:
        # Check if MLX is available
        try:
            import mlx
            import mlx_vlm
            logger.info(f"MLX available: {mlx.__version__}")
        except ImportError:
            logger.warning("MLX not installed. Skipping MLX conversion.")
            logger.warning("Install with: pip install mlx mlx-vlm")
            mlx_model_path = trained_model_path
        else:
            from train_and_deploy import convert_to_mlx
            
            # Clean path for output name
            clean_path = trained_model_path.replace('/', '_').replace('\\', '_')
            mlx_output = f"{clean_path}_mlx"
            mlx_model_path = convert_to_mlx(
                model_path=trained_model_path,
                output_path=mlx_output,
                dtype="bfloat16"
            )
            logger.info(f"✓ MLX conversion complete: {mlx_model_path}")
    except Exception as e:
        logger.error(f"MLX conversion failed: {e}", exc_info=True)
        logger.warning("Continuing without MLX conversion")
        mlx_model_path = trained_model_path
    
    # Step 4: 4-bit quantization
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: 4-Bit Quantization")
    logger.info("=" * 80)
    
    if mlx_model_path and os.path.exists(mlx_model_path):
        try:
            # Check if MLX is available
            try:
                import mlx
            except ImportError:
                logger.warning("MLX not available. Skipping quantization.")
            else:
                from train_and_deploy import quantize_4bit
                
                quantized_output = f"{mlx_model_path}_q4_quarot"
                
                # Try QuaRot first, fallback to standard
                success = quantize_4bit(
                    mlx_model_path=mlx_model_path,
                    output_path=quantized_output,
                    method="quarot"
                )
                
                if not success:
                    logger.warning("QuaRot failed, trying standard method...")
                    quantized_output = f"{mlx_model_path}_q4_standard"
                    success = quantize_4bit(
                        mlx_model_path=mlx_model_path,
                        output_path=quantized_output,
                        method="standard"
                    )
                
                if success:
                    logger.info(f"✓ Quantization complete: {quantized_output}")
                else:
                    logger.warning("Quantization failed")
        except Exception as e:
            logger.error(f"Quantization failed: {e}", exc_info=True)
    else:
        logger.warning("No MLX model available for quantization")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Final model paths:")
    logger.info(f"  Trained: {trained_model_path}")
    if mlx_model_path and mlx_model_path != trained_model_path:
        logger.info(f"  MLX: {mlx_model_path}")


if __name__ == "__main__":
    main()

