"""
Reasoning Benchmark Suite
Uses real evaluation frameworks: lm-eval-harness

Note: GSM8K, MATH, ARC are already covered in frontier_benchmarks.py
This module provides additional reasoning benchmarks.

MIT-level engineering: Production-grade reasoning evaluation using standard tools
"""

import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReasoningBenchmarkSuite:
    """
    Additional reasoning benchmarks using lm-eval-harness.
    
    Note: Main reasoning benchmarks (GSM8K, MATH, ARC) are in frontier_benchmarks.py
    """
    
    def __init__(self, model_path: str, model_type: str = "hf"):
        """
        Initialize reasoning benchmark suite.
        
        Args:
            model_path: Path to model
            model_type: Model type ("hf" for HuggingFace)
        """
        self.model_path = model_path
        self.model_type = model_type
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all reasoning benchmarks."""
        logger.info("Running reasoning benchmark suite...")
        
        return {
            "logiqa": self.run_logiqa(),
        }
    
    def run_logiqa(self) -> Dict[str, Any]:
        """Run LogiQA logical reasoning benchmark using lm-eval-harness."""
        logger.info("Running LogiQA...")
        
        try:
            # Use lm-eval-harness for LogiQA
            cmd = [
                "lm_eval",
                "--model", self.model_type,
                "--model_args", f"pretrained={self.model_path}",
                "--tasks", "logiqa",
                "--num_fewshot", "0",
                "--batch_size", "auto",
                "--output_path", "benchmark_results/logiqa_results.json"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            if result.returncode != 0:
                logger.error(f"lm-eval failed: {result.stderr}")
                return {"error": result.stderr, "status": "failed"}
            
            # Parse results
            try:
                results_file = Path("benchmark_results/logiqa_results.json")
                if results_file.exists():
                    with open(results_file, "r") as f:
                        data = json.load(f)
                        if "results" in data and "logiqa" in data["results"]:
                            task_results = data["results"]["logiqa"]
                            score = task_results.get("acc", task_results.get("acc_norm", 0.0))
                            return {
                                "score": score,
                                "details": task_results,
                                "status": "complete"
                            }
            except Exception as e:
                logger.error(f"Failed to parse results: {e}")
            
            return {"error": "Could not parse results", "status": "failed"}
        except Exception as e:
            logger.error(f"LogiQA evaluation failed: {e}")
            return {"error": str(e), "status": "failed"}

