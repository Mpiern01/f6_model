#!/usr/bin/env python3
"""
F6 StreamTrain: Main Entry Point
Frontier Model-Fusion Training Pipeline

MIT-level engineering: Production-grade training orchestration
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from stages.s0_env_bootstrap import bootstrap_environment
from configs import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for F6 StreamTrain."""
    parser = argparse.ArgumentParser(
        description="F6 StreamTrain: Frontier Model-Fusion Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Bootstrap environment
  python main.py --stage 0

  # Run mid-training (Stage 1)
  python main.py --stage 1 --config configs/stage1_midtrain.yaml

  # Run SFT (Stage 2)
  python main.py --stage 2 --config configs/stage2_sft.yaml

  # Run DPO (Stage 3)
  python main.py --stage 3 --config configs/stage3_dpo.yaml

  # Run RLVR (Stage 4)
  python main.py --stage 4 --config configs/stage4_rlvr.yaml

  # Run InfTool loop (Stage 5)
  python main.py --stage 5 --config configs/stage5_inftool.yaml
        """
    )
    
    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=[0, 1, 2, 3, 4, 5],
        help="Training stage to run"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: auto-detect based on stage)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    
    parser.add_argument(
        "--use-ramdisk",
        action="store_true",
        help="Use RAM disk for ephemeral cache (macOS)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run (validate config without training)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("F6 StreamTrain: Frontier Model-Fusion Training Pipeline")
    logger.info("=" * 80)
    logger.info(f"Stage: {args.stage}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Resume: {args.resume}")
    logger.info("=" * 80)
    
    # Stage 0: Environment bootstrap
    if args.stage == 0:
        cache_manager = bootstrap_environment()
        logger.info("\n✓ Environment bootstrap complete")
        logger.info("You can now run training stages 1-5")
        return
    
    # Bootstrap environment first (if not already done)
    try:
        cache_manager = bootstrap_environment()
    except Exception as e:
        logger.warning(f"Environment bootstrap failed: {e}. Continuing anyway...")
        cache_manager = None
    
    # Load config
    if args.config is None:
        # Auto-detect config based on stage
        config_map = {
            1: "configs/stage1_midtrain.yaml",
            2: "configs/stage2_sft.yaml",
            3: "configs/stage3_dpo.yaml",
            4: "configs/stage4_rlvr.yaml",
            5: "configs/stage5_inftool.yaml"
        }
        args.config = config_map.get(args.stage, "configs/base.yaml")
    
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Import and run stage
    if args.stage == 1:
        from stages.s1_midtrain import run_stage1
        run_stage1(config, resume=args.resume, dry_run=args.dry_run)
    elif args.stage == 2:
        from stages.s2_sft import run_stage2
        run_stage2(config, resume=args.resume, dry_run=args.dry_run)
    elif args.stage == 3:
        from stages.s3_rollout_dpo import run_stage3
        run_stage3(config, resume=args.resume, dry_run=args.dry_run)
    elif args.stage == 4:
        from stages.s4_rlvr_grpo import run_stage4
        run_stage4(config, resume=args.resume, dry_run=args.dry_run)
    elif args.stage == 5:
        from stages.s5_inftool_loop import run_stage5
        run_stage5(config, resume=args.resume, dry_run=args.dry_run)
    
    logger.info("=" * 80)
    logger.info("✓ Training stage complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

