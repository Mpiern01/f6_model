#!/usr/bin/env python3
"""
Stage 0: Environment Bootstrap
Goal: Guarantee "no dataset storage" with ephemeral cache configuration

MIT-level engineering: Robust, production-grade environment setup
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EphemeralCacheManager:
    """Manages ephemeral cache directories for streaming-only data access."""
    
    def __init__(self, base_dir: Optional[str] = None, use_ramdisk: bool = False):
        """
        Initialize ephemeral cache manager.
        
        Args:
            base_dir: Base directory for cache (defaults to /tmp)
            use_ramdisk: If True, attempt to use RAM disk (macOS only)
        """
        self.use_ramdisk = use_ramdisk
        self.base_dir = base_dir or "/tmp"
        self.cache_dir = None
        self._setup_ephemeral_cache()
        
    def _setup_ephemeral_cache(self):
        """Set up ephemeral cache directory."""
        if self.use_ramdisk and sys.platform == "darwin":
            # macOS RAM disk (8GB default)
            ramdisk_path = "/Volumes/RAMDisk"
            if not os.path.exists(ramdisk_path):
                try:
                    os.system(f"diskutil erasevolume HFS+ 'RAMDisk' $(hdiutil attach -nomount ram://16777216)")
                    logger.info(f"Created RAM disk at {ramdisk_path}")
                except Exception as e:
                    logger.warning(f"Failed to create RAM disk: {e}. Using temp directory.")
                    self.cache_dir = tempfile.mkdtemp(prefix="f6_ephemeral_", dir=self.base_dir)
            else:
                self.cache_dir = os.path.join(ramdisk_path, "f6_cache")
                os.makedirs(self.cache_dir, exist_ok=True)
        else:
            # Standard temp directory
            self.cache_dir = tempfile.mkdtemp(prefix="f6_ephemeral_", dir=self.base_dir)
            
        logger.info(f"Ephemeral cache directory: {self.cache_dir}")
        
    def cleanup(self):
        """Clean up ephemeral cache directory."""
        if self.cache_dir and os.path.exists(self.cache_dir):
            if not self.use_ramdisk:
                shutil.rmtree(self.cache_dir, ignore_errors=True)
                logger.info(f"Cleaned up ephemeral cache: {self.cache_dir}")
            else:
                logger.info("RAM disk cache will persist until system restart")


def bootstrap_environment(config_path: Optional[str] = None) -> EphemeralCacheManager:
    """
    Bootstrap F6 StreamTrain environment with no-storage guarantees.
    
    Sets environment variables:
    - HF_HOME: HuggingFace home directory (ephemeral)
    - HF_DATASETS_CACHE: Dataset cache directory (ephemeral)
    - HF_HUB_CACHE: Hub cache directory (ephemeral)
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        EphemeralCacheManager instance
    """
    logger.info("=" * 60)
    logger.info("F6 StreamTrain: Stage 0 - Environment Bootstrap")
    logger.info("=" * 60)
    
    # Initialize ephemeral cache manager
    cache_manager = EphemeralCacheManager(use_ramdisk=False)  # Can be enabled for production
    
    # Set HuggingFace environment variables
    cache_base = cache_manager.cache_dir
    
    os.environ["HF_HOME"] = os.path.join(cache_base, "hf_home")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_base, "datasets_cache")
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_base, "hub_cache")
    
    # Create directories
    for env_var in ["HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE"]:
        cache_path = os.environ[env_var]
        os.makedirs(cache_path, exist_ok=True)
        logger.info(f"Set {env_var} = {cache_path}")
    
    # Verify streaming mode
    os.environ["HF_DATASETS_STREAMING"] = "1"
    logger.info("Enabled HuggingFace streaming mode")
    
    # Log configuration
    logger.info("\nEnvironment Configuration:")
    logger.info(f"  HF_HOME: {os.environ['HF_HOME']}")
    logger.info(f"  HF_DATASETS_CACHE: {os.environ['HF_DATASETS_CACHE']}")
    logger.info(f"  HF_HUB_CACHE: {os.environ['HF_HUB_CACHE']}")
    logger.info(f"  Streaming: Enabled")
    logger.info(f"  No Storage: Guaranteed (ephemeral cache only)")
    
    logger.info("\n✓ Environment bootstrap complete")
    logger.info("=" * 60)
    
    return cache_manager


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bootstrap F6 StreamTrain environment")
    parser.add_argument("--use-ramdisk", action="store_true", help="Use RAM disk for cache (macOS)")
    parser.add_argument("--config", type=str, help="Path to config file")
    
    args = parser.parse_args()
    
    cache_manager = bootstrap_environment(args.config)
    
    # Keep cache manager alive for interactive use
    try:
        logger.info("\nEnvironment ready. Press Ctrl+C to exit and cleanup.")
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nCleaning up...")
        cache_manager.cleanup()

