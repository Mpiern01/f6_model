"""
Frontier Benchmark Suite
Uses real evaluation frameworks: lm-eval-harness

MIT-level engineering: Production-grade benchmark evaluation using standard tools
"""

import logging
import subprocess
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrontierBenchmarkSuite:
    """
    Comprehensive frontier benchmark suite using lm-eval-harness.
    
    Uses the official lm-eval-harness framework which is the standard
    for evaluating frontier models.
    
    Benchmarks:
    - MMLU (Massive Multitask Language Understanding)
    - HellaSwag (Commonsense reasoning)
    - GSM8K (Math word problems)
    - MATH (Competition math)
    - ARC (AI2 Reasoning Challenge)
    - TruthfulQA (Truthfulness)
    - Winogrande (Commonsense)
    - PIQA (Physical reasoning)
    """
    
    def __init__(self, model_path: str, model_type: str = "hf"):
        """
        Initialize benchmark suite.
        
        Args:
            model_path: Path to model (HuggingFace path or local path)
            model_type: Model type ("hf" for HuggingFace, "vllm" for vLLM, etc.)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.results = {}
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """
        Run all frontier benchmarks using lm-eval-harness.
        
        Returns:
            Dictionary with all benchmark results
        """
        logger.info("Running frontier benchmark suite with lm-eval-harness...")
        
        # Standard benchmarks used by frontier models
        benchmarks = [
            "mmlu",
            "hellaswag",
            "gsm8k",
            "math",
            "arc",
            "truthfulqa",
            "winogrande",
            "piqa"
        ]
        
        results = {}
        for benchmark in benchmarks:
            try:
                logger.info(f"Running {benchmark}...")
                result = self._run_lm_eval(benchmark)
                results[benchmark] = result
            except Exception as e:
                logger.error(f"Failed to run {benchmark}: {e}")
                results[benchmark] = {"error": str(e), "status": "failed"}
        
        self.results = results
        return results
    
    def _run_lm_eval(self, task: str, num_fewshot: Optional[int] = None) -> Dict[str, Any]:
        """
        Run benchmark using lm-eval-harness Python API (not subprocess).
        
        Uses the actual Python library for better integration.
        
        Args:
            task: Benchmark task name
            num_fewshot: Number of few-shot examples (uses default if None)
            
        Returns:
            Benchmark results
        """
        # Try to use Python API first
        try:
            from lm_eval import simple_evaluate
            from lm_eval.models.huggingface import HFLM
            
            # Default few-shot settings
            fewshot_defaults = {
                "mmlu": 5,
                "hellaswag": 10,
                "gsm8k": 5,
                "math": 4,
                "arc": 25,
                "truthfulqa": 0,
                "winogrande": 0,
                "piqa": 0
            }
            
            num_fewshot = num_fewshot or fewshot_defaults.get(task, 0)
            
            # Initialize model
            model = HFLM(pretrained=self.model_path)
            
            # Run evaluation
            results = simple_evaluate(
                model=model,
                tasks=[task],
                num_fewshot=num_fewshot,
                batch_size="auto"
            )
            
            # Extract results
            if "results" in results and task in results["results"]:
                task_results = results["results"][task]
                score = task_results.get("acc", task_results.get("acc_norm", task_results.get("exact_match", 0.0)))
                return {
                    "score": score,
                    "details": task_results,
                    "status": "complete"
                }
            
            return {"error": "No results found", "status": "failed"}
            
        except ImportError:
            logger.warning("lm-eval Python API not available, using CLI fallback")
            # Fallback to CLI
            return self._run_lm_eval_cli(task, num_fewshot)
        except Exception as e:
            logger.error(f"lm-eval Python API failed: {e}, using CLI fallback")
            return self._run_lm_eval_cli(task, num_fewshot)
    
    def _run_lm_eval_cli(self, task: str, num_fewshot: Optional[int] = None) -> Dict[str, Any]:
        """Fallback: Run lm-eval using CLI."""
        fewshot_defaults = {
            "mmlu": 5, "hellaswag": 10, "gsm8k": 5, "math": 4,
            "arc": 25, "truthfulqa": 0, "winogrande": 0, "piqa": 0
        }
        num_fewshot = num_fewshot or fewshot_defaults.get(task, 0)
        
        cmd = [
            "lm_eval",
            "--model", self.model_type,
            "--model_args", f"pretrained={self.model_path}",
            "--tasks", task,
            "--num_fewshot", str(num_fewshot),
            "--batch_size", "auto",
            "--output_path", f"benchmark_results/{task}_results.json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        if result.returncode != 0:
            return {"error": result.stderr, "status": "failed"}
        
        try:
            results_file = Path(f"benchmark_results/{task}_results.json")
            if results_file.exists():
                with open(results_file, "r") as f:
                    data = json.load(f)
                    if "results" in data and task in data["results"]:
                        task_results = data["results"][task]
                        score = task_results.get("acc", task_results.get("acc_norm", 0.0))
                        return {"score": score, "details": task_results, "status": "complete"}
        except Exception as e:
            logger.error(f"Failed to parse results: {e}")
        
        return {"error": "Could not parse results", "status": "failed"}
    
    def run_mmlu(self) -> Dict[str, Any]:
        """Run MMLU benchmark."""
        return self._run_lm_eval("mmlu", num_fewshot=5)
    
    def run_hellaswag(self) -> Dict[str, Any]:
        """Run HellaSwag benchmark."""
        return self._run_lm_eval("hellaswag", num_fewshot=10)
    
    def run_gsm8k(self) -> Dict[str, Any]:
        """Run GSM8K benchmark."""
        return self._run_lm_eval("gsm8k", num_fewshot=5)
    
    def run_math(self) -> Dict[str, Any]:
        """Run MATH benchmark."""
        return self._run_lm_eval("math", num_fewshot=4)
    
    def run_arc(self) -> Dict[str, Any]:
        """Run ARC benchmark."""
        return self._run_lm_eval("arc", num_fewshot=25)
    
    def run_truthfulqa(self) -> Dict[str, Any]:
        """Run TruthfulQA benchmark."""
        return self._run_lm_eval("truthfulqa", num_fewshot=0)
    
    def run_winogrande(self) -> Dict[str, Any]:
        """Run Winogrande benchmark."""
        return self._run_lm_eval("winogrande", num_fewshot=0)
    
    def run_piqa(self) -> Dict[str, Any]:
        """Run PIQA benchmark."""
        return self._run_lm_eval("piqa", num_fewshot=0)
