"""
MLX VLM Export
Convert F6 StreamTrain model to MLX format for Apple Silicon

Supports: Jan-v2-VL-high and derived models
Uses: mlx-vlm for VLM conversion

MIT-level engineering: Production-grade MLX conversion
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_mlx_installation():
    """Check if MLX and mlx-vlm are installed."""
    try:
        import mlx
        import mlx_vlm
        logger.info(f"MLX version: {mlx.__version__}")
        return True
    except ImportError as e:
        logger.error(f"MLX not installed: {e}")
        logger.error("Install with: pip install mlx mlx-vlm")
        return False


def convert_to_mlx(
    model_path: str,
    output_path: str,
    dtype: str = "bfloat16",
    hf_path: Optional[str] = None
):
    """
    Convert model to MLX format.
    
    Args:
        model_path: Path to model (local or HuggingFace)
        output_path: Output path for MLX model
        dtype: Data type (bfloat16, float16, etc.)
        hf_path: Optional HuggingFace path if model_path is local
    """
    logger.info("=" * 80)
    logger.info("MLX VLM Conversion")
    logger.info("=" * 80)
    
    if not check_mlx_installation():
        sys.exit(1)
    
    # Determine if model_path is local or HuggingFace
    if os.path.exists(model_path):
        # Local path
        hf_path = model_path
        logger.info(f"Converting local model: {model_path}")
    else:
        # HuggingFace path
        hf_path = model_path
        logger.info(f"Converting HuggingFace model: {hf_path}")
    
    # Use mlx-vlm conversion
    try:
        from mlx_vlm import convert
        
        logger.info(f"Converting to MLX format (dtype={dtype})...")
        logger.info(f"Output path: {output_path}")
        
        # Convert using mlx-vlm
        convert(
            hf_path=hf_path,
            mlx_path=output_path,
            dtype=dtype
        )
        
        logger.info(f"✓ Conversion complete: {output_path}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        logger.info("Trying alternative conversion method...")
        
        # Alternative: Use command-line mlx_vlm.convert
        try:
            cmd = [
                sys.executable, "-m", "mlx_vlm.convert",
                "--hf-path", hf_path,
                "--mlx-path", output_path,
                "--dtype", dtype
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✓ Conversion complete: {output_path}")
            else:
                logger.error(f"Conversion failed: {result.stderr}")
                sys.exit(1)
                
        except Exception as e2:
            logger.error(f"Alternative conversion also failed: {e2}")
            sys.exit(1)
    
    # Verify output
    if os.path.exists(output_path):
        logger.info(f"✓ MLX model saved to: {output_path}")
        logger.info(f"  Model size: {get_directory_size(output_path) / (1024**3):.2f} GB")
    else:
        logger.error(f"Output path does not exist: {output_path}")
        sys.exit(1)
    
    logger.info("=" * 80)


def get_directory_size(path: str) -> int:
    """Get total size of directory in bytes."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total += os.path.getsize(filepath)
    return total


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert F6 StreamTrain model to MLX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Jan-v2-VL-high (Qwen3VL)
  python mlx/export_mlx_vlm.py --model-path janhq/Jan-v2-VL-high --output-path Jan-v2-VL-high-mlx

  # Convert GLM-4.6V-Flash
  python mlx/export_mlx_vlm.py --model-path zai-org/GLM-4.6V-Flash --output-path GLM-4.6V-Flash-mlx

  # Convert local checkpoint
  python mlx/export_mlx_vlm.py --model-path checkpoints/stage1_midtrain/final --output-path f6-streamtrain-mlx

  # Convert with specific dtype
  python mlx/export_mlx_vlm.py --model-path janhq/Jan-v2-VL-high --output-path Jan-v2-VL-high-mlx --dtype float16
        """
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model (local or HuggingFace identifier)"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path for MLX model"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for MLX model (default: bfloat16)"
    )
    
    args = parser.parse_args()
    
    convert_to_mlx(
        model_path=args.model_path,
        output_path=args.output_path,
        dtype=args.dtype
    )


if __name__ == "__main__":
    main()

