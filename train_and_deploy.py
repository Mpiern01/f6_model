#!/usr/bin/env python3
"""
Complete Training and Deployment Pipeline
Pull model → Train → MLX Conversion → 4-bit Quantization

MIT-level engineering: Production-grade pipeline with comprehensive error handling
"""

import os
import sys
import logging
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check all required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    required = {
        "torch": "torch",
        "transformers": "transformers",
        "datasets": "datasets",
        "peft": "peft",
        "accelerate": "accelerate",
    }
    
    missing = []
    for module_name, package_name in required.items():
        try:
            __import__(module_name)
            logger.info(f"✓ {package_name} installed")
        except ImportError:
            missing.append(package_name)
            logger.error(f"✗ {package_name} missing")
    
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.error(f"Install with: pip install {' '.join(missing)}")
        return False
    
    # Check MLX (optional but recommended)
    try:
        import mlx
        logger.info(f"✓ MLX installed (version: {mlx.__version__})")
    except ImportError:
        logger.warning("MLX not installed - MLX conversion will fail")
        logger.warning("Install with: pip install mlx mlx-vlm mlx-lm")
    
    return True


def pull_model(model_path: str = "janhq/Jan-v2-VL-high", local_dir: Optional[str] = None):
    """
    Pull model from HuggingFace.

    Supported models:
    - OPTION 1: janhq/Jan-v2-VL-high (Qwen3-VL-8B-Thinking based)
    - OPTION 2: zai-org/GLM-4.6V-Flash (GLM-4V based)

    Args:
        model_path: HuggingFace model path
        local_dir: Optional local directory to save model
    """
    logger.info("=" * 80)
    logger.info("Pulling Model from HuggingFace")
    logger.info("=" * 80)
    logger.info(f"Model: {model_path}")
    
    try:
        from huggingface_hub import snapshot_download
        
        # Set ephemeral cache
        os.environ["HF_HOME"] = "/tmp/f6_hf_home"
        os.environ["HF_DATASETS_CACHE"] = "/tmp/f6_datasets_cache"
        
        if local_dir is None:
            # Standardize model directory name: use just the model name, not org/model
            model_name = model_path.split('/')[-1] if '/' in model_path else model_path
            local_dir = f"models/{model_name}"
        
        logger.info(f"Downloading to: {local_dir}")
        
        # Download model
        snapshot_download(
            repo_id=model_path,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        logger.info(f"✓ Model downloaded to: {local_dir}")
        return local_dir
        
    except Exception as e:
        logger.error(f"Failed to pull model: {e}")
        logger.error("Trying alternative method...")
        
        # Alternative: Use transformers to load (will cache automatically)
        try:
            from transformers import AutoTokenizer, AutoConfig
            
            logger.info("Testing model access...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            config = AutoConfig.from_pretrained(model_path)
            
            logger.info(f"✓ Model accessible: {model_path}")
            logger.info(f"  Vocab size: {tokenizer.vocab_size}")
            logger.info(f"  Model type: {config.model_type}")
            
            return model_path  # Use HuggingFace path directly
            
        except Exception as e2:
            logger.error(f"Alternative method also failed: {e2}")
            raise RuntimeError(f"Could not pull model: {e}")


def train_model(
    model_path: str,
    config_path: str = "configs/stage1_midtrain.yaml",
    max_steps: Optional[int] = None,
    dry_run: bool = False
):
    """
    Train the model.
    
    Args:
        model_path: Path to base model
        config_path: Path to training config
        max_steps: Maximum training steps (None for full training)
        dry_run: If True, validate without training
    """
    logger.info("=" * 80)
    logger.info("Training Model")
    logger.info("=" * 80)
    
    try:
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Update model path in config
        config["model"]["base_model"] = model_path
        
        # Override max_steps if provided
        if max_steps:
            config["training"]["max_steps"] = max_steps
            logger.info(f"Training for {max_steps} steps (limited)")
        
        # Import and run training
        from stages.s1_midtrain import run_stage1
        
        logger.info(f"Starting training with config: {config_path}")
        logger.info(f"Base model: {model_path}")
        
        run_stage1(config, resume=None, dry_run=dry_run)
        
        # Determine output path
        output_dir = config.get("output", {}).get("checkpoint_dir", "checkpoints/stage1_midtrain")
        final_model_path = os.path.join(output_dir, "final")
        
        # If final doesn't exist, use latest checkpoint or save final
        if not os.path.exists(final_model_path):
            checkpoints = sorted(Path(output_dir).glob("checkpoint-*"), reverse=True)
            if checkpoints:
                # Use latest checkpoint as final
                latest_checkpoint = str(checkpoints[0])
                logger.info(f"Using latest checkpoint as final: {latest_checkpoint}")
                
                # Copy to final directory
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                logger.info("Loading model from checkpoint...")
                model = AutoModelForCausalLM.from_pretrained(latest_checkpoint)
                tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint)
                
                logger.info("Saving final model...")
                os.makedirs(final_model_path, exist_ok=True)
                model.save_pretrained(final_model_path)
                tokenizer.save_pretrained(final_model_path)
                
                logger.info(f"Final model saved to: {final_model_path}")
            else:
                # No checkpoints, use output_dir as final
                logger.warning(f"No checkpoints found in {output_dir}, using output_dir as final")
                final_model_path = output_dir
        
        logger.info(f"✓ Training complete. Model saved to: {final_model_path}")
        return final_model_path
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise RuntimeError(f"Training failed: {e}")


def convert_to_mlx(model_path: str, output_path: str, dtype: str = "bfloat16"):
    """
    Convert model to MLX format.
    
    Args:
        model_path: Path to trained model
        output_path: Output path for MLX model
        dtype: Data type (bfloat16, float16, float32)
    """
    logger.info("=" * 80)
    logger.info("Converting to MLX Format")
    logger.info("=" * 80)
    logger.info(f"Model: {model_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Data type: {dtype}")
    
    try:
        # Check MLX installation
        try:
            import mlx
            import mlx_vlm
            logger.info(f"MLX version: {mlx.__version__}")
        except ImportError:
            logger.error("MLX not installed")
            logger.error("Install with: pip install mlx mlx-vlm")
            raise RuntimeError("MLX not available")
        
        # Use mlx-vlm conversion
        from mlx_vlm import convert
        
        logger.info("Starting MLX conversion...")
        
        convert(
            hf_path=model_path,
            mlx_path=output_path,
            dtype=dtype
        )
        
        # Verify output
        if os.path.exists(output_path):
            logger.info(f"✓ MLX conversion complete: {output_path}")
            
            # Check model size
            total_size = sum(
                f.stat().st_size
                for f in Path(output_path).rglob('*')
                if f.is_file()
            )
            logger.info(f"  Model size: {total_size / (1024**3):.2f} GB")
        else:
            raise RuntimeError(f"MLX conversion failed: output path does not exist")
        
        return output_path
        
    except Exception as e:
        logger.error(f"MLX conversion failed: {e}", exc_info=True)
        raise RuntimeError(f"MLX conversion failed: {e}")


def quantize_4bit(mlx_model_path: str, output_path: str, method: str = "quarot"):
    """
    Quantize MLX model to 4-bit.
    
    Args:
        mlx_model_path: Path to MLX model
        output_path: Output path for quantized model
        method: Quantization method (quarot, bitnet_v2, amxfp4, standard)
    """
    logger.info("=" * 80)
    logger.info("4-Bit Quantization")
    logger.info("=" * 80)
    logger.info(f"Model: {mlx_model_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Method: {method}")
    
    try:
        # Check MLX installation
        try:
            import mlx
            import mlx.core as mx
            logger.info(f"MLX version: {mlx.__version__}")
        except ImportError:
            logger.error("MLX not installed")
            raise RuntimeError("MLX not available")
        
        # Import quantization function
        sys.path.insert(0, str(Path(__file__).parent))
        from mlx.quantize_4bit import quantize_model
        
        logger.info(f"Quantizing with {method} method...")
        
        # Quantize
        quantize_model(
            model_path=mlx_model_path,
            output_path=output_path,
            method=method,
            bits=4
        )
        
        # Verify output
        if os.path.exists(output_path):
            logger.info(f"✓ 4-bit quantization complete: {output_path}")
            
            # Check model size
            total_size = sum(
                f.stat().st_size
                for f in Path(output_path).rglob('*')
                if f.is_file()
            )
            logger.info(f"  Quantized model size: {total_size / (1024**3):.2f} GB")
            
            # Compare to original
            original_size = sum(
                f.stat().st_size
                for f in Path(mlx_model_path).rglob('*')
                if f.is_file()
            )
            reduction = (1 - total_size / original_size) * 100 if original_size > 0 else 0
            logger.info(f"  Size reduction: {reduction:.1f}%")
        else:
            raise RuntimeError(f"Quantization failed: output path does not exist")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}", exc_info=True)
        raise RuntimeError(f"Quantization failed: {e}")


def main():
    """Main pipeline execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Complete training and deployment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="janhq/Jan-v2-VL-high",
        help="HuggingFace model path or local path (Options: janhq/Jan-v2-VL-high, zai-org/GLM-4.6V-Flash)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage1_midtrain.yaml",
        help="Training config path"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (for testing)"
    )
    
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (use existing model)"
    )
    
    parser.add_argument(
        "--skip-mlx",
        action="store_true",
        help="Skip MLX conversion"
    )
    
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Skip quantization"
    )
    
    parser.add_argument(
        "--quantization-method",
        type=str,
        default="quarot",
        choices=["quarot", "bitnet_v2", "amxfp4", "standard"],
        help="Quantization method"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run (validate without executing)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("F6 StreamTrain - Complete Pipeline")
    logger.info("=" * 80)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed")
        sys.exit(1)
    
    try:
        # Step 1: Pull model
        if not args.skip_training:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 1: Pulling Model")
            logger.info("=" * 80)
            
            model_path = pull_model(args.model_path)
        else:
            logger.info("Skipping model pull (using existing)")
            model_path = args.model_path
        
        # Step 2: Train model
        trained_model_path = None
        if not args.skip_training:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 2: Training Model")
            logger.info("=" * 80)
            
            trained_model_path = train_model(
                model_path=model_path,
                config_path=args.config,
                max_steps=args.max_steps,
                dry_run=args.dry_run
            )
        else:
            logger.info("Skipping training")
            trained_model_path = model_path
        
        # Step 3: Convert to MLX
        mlx_model_path = None
        if not args.skip_mlx:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3: Converting to MLX")
            logger.info("=" * 80)
            
            mlx_output = f"{trained_model_path.replace('/', '_')}_mlx"
            mlx_model_path = convert_to_mlx(
                model_path=trained_model_path,
                output_path=mlx_output,
                dtype="bfloat16"
            )
        else:
            logger.info("Skipping MLX conversion")
            mlx_model_path = trained_model_path
        
        # Step 4: 4-bit quantization
        if not args.skip_quantization:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 4: 4-Bit Quantization")
            logger.info("=" * 80)
            
            quantized_output = f"{mlx_model_path}_q4_{args.quantization_method}"
            quantized_path = quantize_4bit(
                mlx_model_path=mlx_model_path,
                output_path=quantized_output,
                method=args.quantization_method
            )
            
            logger.info("\n" + "=" * 80)
            logger.info("✓ PIPELINE COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Final quantized model: {quantized_path}")
        else:
            logger.info("Skipping quantization")
            logger.info("\n" + "=" * 80)
            logger.info("✓ PIPELINE COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Final MLX model: {mlx_model_path}")
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("PIPELINE FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

