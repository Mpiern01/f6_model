#!/usr/bin/env python3
"""
Run All Benchmarks
Automated benchmark testing after training

MIT-level engineering: Production-grade benchmark execution
"""

import argparse
import logging
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.frontier_benchmarks import FrontierBenchmarkSuite
from benchmarks.long_horizon_benchmark import LongHorizonBenchmark
from benchmarks.coding_benchmarks import CodingBenchmarkSuite
from benchmarks.automated_report import BenchmarkReportGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run all benchmarks and generate report."""
    parser = argparse.ArgumentParser(description="Run comprehensive benchmark suite")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output-dir", type=str, default="benchmark_reports", help="Output directory")
    parser.add_argument("--skip-long-horizon", action="store_true", help="Skip long-horizon benchmark")
    parser.add_argument("--skip-multimodal", action="store_true", help="Skip multimodal benchmarks")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("F6 StreamTrain - Comprehensive Benchmark Suite")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info("=" * 80)
    
    # Run benchmarks (using real evaluation frameworks)
    all_results = {}
    
    # Frontier benchmarks (using lm-eval-harness)
    logger.info("\n" + "=" * 80)
    logger.info("Running Frontier Benchmarks (lm-eval-harness)")
    logger.info("=" * 80)
    frontier_suite = FrontierBenchmarkSuite(args.model_path, model_type="hf")
    frontier_results = frontier_suite.run_all_benchmarks()
    all_results.update(frontier_results)
    
    # Coding benchmarks (using official frameworks)
    logger.info("\n" + "=" * 80)
    logger.info("Running Coding Benchmarks (HumanEval, MBPP, SWE-bench)")
    logger.info("=" * 80)
    coding_suite = CodingBenchmarkSuite(args.model_path)
    coding_results = coding_suite.run_all_benchmarks()
    all_results.update(coding_results)
    
    # Long-horizon benchmark
    if not args.skip_long_horizon:
        logger.info("\n" + "=" * 80)
        logger.info("Running Long-Horizon Benchmark")
        logger.info("=" * 80)
        long_horizon = LongHorizonBenchmark(args.model_path)
        long_horizon_results = long_horizon.evaluate()
        all_results["long_horizon"] = long_horizon_results
    
    # Generate report
    logger.info("\n" + "=" * 80)
    logger.info("Generating Benchmark Report")
    logger.info("=" * 80)
    
    model_info = {
        "name": "F6-StreamTrain",
        # OPTION 1: Jan-v2-VL-high (Qwen3VL)
        "base_model": "janhq/Jan-v2-VL-high",

        # OPTION 2: GLM-4.6V-Flash (Uncomment to use)
        # "base_model": "zai-org/GLM-4.6V-Flash",
        "path": args.model_path
    }
    
    report_generator = BenchmarkReportGenerator(output_dir=args.output_dir)
    report_path = report_generator.generate_report(
        benchmark_results=all_results,
        model_info=model_info
    )
    
    logger.info("=" * 80)
    logger.info(f"✓ Benchmark suite complete")
    logger.info(f"Report saved to: {report_path}")
    logger.info("=" * 80)
    
    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    for name, result in all_results.items():
        if isinstance(result, dict):
            score = result.get("score", result.get("pass@1", 0.0))
            status = result.get("status", "unknown")
            print(f"{name.upper()}: {score:.4f} ({status})")
    print("=" * 80)


if __name__ == "__main__":
    main()

