#!/usr/bin/env python3
"""
Production Training Script with Full Tracking
No mocks, no placeholders - real implementation

MIT-level engineering: Production-grade training
"""

import torch
import logging
import sys
from pathlib import Path
import yaml
from transformers import TrainingArguments

sys.path.insert(0, str(Path(__file__).parent))

from stages.s1_midtrain import run_stage1
from stages.s2_sft import run_stage2
from stages.s3_rollout_dpo import run_stage3
from stages.s4_rlvr_grpo import run_stage4
from stages.s5_inftool_loop import run_stage5
from streaming.dataset_groups import DatasetGroupManager
from benchmarks.run_benchmarks import main as run_benchmarks

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_full_pipeline(config_path: str = "configs/base.yaml"):
    """
    Train full pipeline with tracking and safeguards.
    
    Args:
        config_path: Path to base config
    """
    logger.info("=" * 80)
    logger.info("F6 StreamTrain - Full Pipeline Training")
    logger.info("=" * 80)
    
    # Load config
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)
    
    # Initialize dataset group manager
    group_manager = DatasetGroupManager()
    
    try:
        # Stage 1: Mid-training
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: Mid-Training")
        logger.info("=" * 80)
        stage1_config = yaml.safe_load(open("configs/stage1_midtrain.yaml"))
        run_stage1(stage1_config, dry_run=False)
        group_manager.save_progress("tracking/stage1_progress.json")
        
        # Stage 2: SFT
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: Supervised Fine-Tuning")
        logger.info("=" * 80)
        stage2_config = yaml.safe_load(open("configs/stage2_sft.yaml"))
        run_stage2(stage2_config, dry_run=False)
        group_manager.save_progress("tracking/stage2_progress.json")
        
        # Stage 3: DPO
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 3: Rollout-DPO")
        logger.info("=" * 80)
        stage3_config = yaml.safe_load(open("configs/stage3_dpo.yaml"))
        run_stage3(stage3_config, dry_run=False)
        group_manager.save_progress("tracking/stage3_progress.json")
        
        # Stage 4: RLVR
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 4: RLVR with GRPO")
        logger.info("=" * 80)
        stage4_config = yaml.safe_load(open("configs/stage4_rlvr.yaml"))
        run_stage4(stage4_config, dry_run=False)
        group_manager.save_progress("tracking/stage4_progress.json")
        
        # Stage 5: InfTool Loop
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 5: InfTool Closed Loop")
        logger.info("=" * 80)
        stage5_config = yaml.safe_load(open("configs/stage5_inftool.yaml"))
        run_stage5(stage5_config, dry_run=False)
        group_manager.save_progress("tracking/stage5_progress.json")
        
        # Final model path
        final_model_path = "checkpoints/stage5_inftool/final"
        
        # Run benchmarks
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING BENCHMARKS")
        logger.info("=" * 80)
        
        sys.argv = ["run_benchmarks.py", "--model-path", final_model_path]
        run_benchmarks()
        
        logger.info("=" * 80)
        logger.info("✓ Full pipeline training complete")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        group_manager.save_progress("tracking/training_failed.json")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train F6 StreamTrain pipeline")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--stage", type=int, help="Train specific stage only")
    
    args = parser.parse_args()
    
    if args.stage:
        # Train single stage
        configs = {
            1: "configs/stage1_midtrain.yaml",
            2: "configs/stage2_sft.yaml",
            3: "configs/stage3_dpo.yaml",
            4: "configs/stage4_rlvr.yaml",
            5: "configs/stage5_inftool.yaml"
        }
        
        config_path = configs.get(args.stage)
        if not config_path:
            logger.error(f"Invalid stage: {args.stage}")
            sys.exit(1)
        
        config = yaml.safe_load(open(config_path))
        
        if args.stage == 1:
            run_stage1(config)
        elif args.stage == 2:
            run_stage2(config)
        elif args.stage == 3:
            run_stage3(config)
        elif args.stage == 4:
            run_stage4(config)
        elif args.stage == 5:
            run_stage5(config)
    else:
        # Train full pipeline
        train_full_pipeline(args.config)

