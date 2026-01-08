#!/usr/bin/env python3
"""
Quick test of streaming functionality without loading the full model
"""

import sys
import logging
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

# Bootstrap environment first
from stages.s0_env_bootstrap import bootstrap_environment
bootstrap_environment()

from streaming.hf_stream import MultiDatasetStreamer
from configs import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_streaming():
    """Test streaming with Dolphin Coder dataset"""
    logger.info("="*80)
    logger.info("Testing Streaming Functionality")
    logger.info("="*80)
    
    # Load config
    config = load_config("configs/quick_test.yaml")
    logger.info(f"Loaded config: {config['data']}")
    
    # Create streamer
    logger.info("Creating multi-dataset streamer...")
    streamer = MultiDatasetStreamer(
        dataset_configs=config["data"]["datasets"]
    )
    
    # Test fetching a few samples
    logger.info("Fetching 3 samples from stream...")
    stream_iter = streamer.stream_mixed(max_samples=3)
    for i in range(3):
        try:
            sample = next(stream_iter)
            logger.info(f"\nSample {i+1}:")
            logger.info(f"  Keys: {list(sample.keys())}")
            if 'text' in sample:
                logger.info(f"  Text preview: {sample['text'][:100]}...")
            elif 'content' in sample:
                logger.info(f"  Content preview: {sample['content'][:100]}...")
            else:
                logger.info(f"  Sample: {str(sample)[:200]}...")
        except Exception as e:
            logger.error(f"Error fetching sample {i+1}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    logger.info("\n" + "="*80)
    logger.info("✓ Streaming test complete!")
    logger.info("="*80)

if __name__ == "__main__":
    test_streaming()

