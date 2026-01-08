"""
Automated Benchmark Report Generator
Generates comprehensive benchmark reports after training

MIT-level engineering: Production-grade reporting
"""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkReportGenerator:
    """
    Generates comprehensive benchmark reports.
    """
    
    def __init__(self, output_dir: str = "benchmark_reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_report(
        self,
        benchmark_results: Dict[str, Any],
        model_info: Dict[str, Any],
        training_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate comprehensive benchmark report.
        
        Args:
            benchmark_results: Results from all benchmarks
            model_info: Model information
            training_info: Optional training information
            
        Returns:
            Path to generated report
        """
        logger.info("Generating benchmark report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"benchmark_report_{timestamp}.json"
        html_path = self.output_dir / f"benchmark_report_{timestamp}.html"
        
        # Compile report data
        report = {
            "timestamp": timestamp,
            "model_info": model_info,
            "training_info": training_info or {},
            "benchmarks": benchmark_results,
            "summary": self._generate_summary(benchmark_results),
            "comparison": self._compare_to_baseline(benchmark_results)
        }
        
        # Save JSON report
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Generate HTML report
        html_content = self._generate_html_report(report)
        with open(html_path, "w") as f:
            f.write(html_content)
        
        logger.info(f"Report generated: {html_path}")
        
        return str(html_path)
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            "total_benchmarks": 0,
            "completed": 0,
            "failed": 0,
            "average_scores": {},
            "best_performers": [],
            "needs_improvement": []
        }
        
        for benchmark_name, benchmark_result in results.items():
            if isinstance(benchmark_result, dict):
                summary["total_benchmarks"] += 1
                
                if benchmark_result.get("status") == "complete":
                    summary["completed"] += 1
                    score = benchmark_result.get("score", benchmark_result.get("pass@1", 0.0))
                    summary["average_scores"][benchmark_name] = score
                    
                    if score >= 0.8:
                        summary["best_performers"].append(benchmark_name)
                    elif score < 0.5:
                        summary["needs_improvement"].append(benchmark_name)
                else:
                    summary["failed"] += 1
        
        return summary
    
    def _compare_to_baseline(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare to baseline models."""
        # Baseline scores (approximate frontier model scores)
        baselines = {
            "mmlu": 0.90,
            "hellaswag": 0.95,
            "humaneval": 0.75,
            "gsm8k": 0.95,
            "math": 0.60,
            "arc": 0.85,
        }
        
        comparison = {}
        for benchmark_name, baseline_score in baselines.items():
            if benchmark_name in results:
                result = results[benchmark_name]
                if isinstance(result, dict) and "score" in result:
                    current_score = result["score"]
                    comparison[benchmark_name] = {
                        "baseline": baseline_score,
                        "current": current_score,
                        "difference": current_score - baseline_score,
                        "percentage": ((current_score - baseline_score) / baseline_score) * 100
                    }
        
        return comparison
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>F6 StreamTrain Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .score-high {{ color: green; font-weight: bold; }}
        .score-medium {{ color: orange; font-weight: bold; }}
        .score-low {{ color: red; font-weight: bold; }}
        .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>F6 StreamTrain Benchmark Report</h1>
    <p><strong>Generated:</strong> {report['timestamp']}</p>
    
    <h2>Model Information</h2>
    <div class="summary">
        <p><strong>Model:</strong> {report['model_info'].get('name', 'Unknown')}</p>
        <p><strong>Base Model:</strong> {report['model_info'].get('base_model', 'Unknown')}</p>
        <p><strong>Parameters:</strong> {report['model_info'].get('parameters', 'Unknown')}</p>
    </div>
    
    <h2>Benchmark Results</h2>
    <table>
        <tr>
            <th>Benchmark</th>
            <th>Score</th>
            <th>Status</th>
            <th>Details</th>
        </tr>
"""
        
        for benchmark_name, result in report["benchmarks"].items():
            if isinstance(result, dict):
                score = result.get("score", result.get("pass@1", 0.0))
                status = result.get("status", "unknown")
                
                score_class = "score-high" if score >= 0.8 else "score-medium" if score >= 0.5 else "score-low"
                
                html += f"""
        <tr>
            <td>{benchmark_name.upper()}</td>
            <td class="{score_class}">{score:.4f}</td>
            <td>{status}</td>
            <td>{json.dumps(result.get('details', {}), indent=2)[:100]}...</td>
        </tr>
"""
        
        html += """
    </table>
    
    <h2>Summary</h2>
    <div class="summary">
"""
        
        summary = report["summary"]
        html += f"""
        <p><strong>Total Benchmarks:</strong> {summary['total_benchmarks']}</p>
        <p><strong>Completed:</strong> {summary['completed']}</p>
        <p><strong>Failed:</strong> {summary['failed']}</p>
        <p><strong>Best Performers:</strong> {', '.join(summary['best_performers']) if summary['best_performers'] else 'None'}</p>
        <p><strong>Needs Improvement:</strong> {', '.join(summary['needs_improvement']) if summary['needs_improvement'] else 'None'}</p>
"""
        
        html += """
    </div>
    
    <h2>Comparison to Baseline</h2>
    <table>
        <tr>
            <th>Benchmark</th>
            <th>Baseline</th>
            <th>Current</th>
            <th>Difference</th>
            <th>Percentage</th>
        </tr>
"""
        
        for benchmark_name, comp in report.get("comparison", {}).items():
            diff_class = "score-high" if comp["difference"] >= 0 else "score-low"
            html += f"""
        <tr>
            <td>{benchmark_name.upper()}</td>
            <td>{comp['baseline']:.4f}</td>
            <td>{comp['current']:.4f}</td>
            <td class="{diff_class}">{comp['difference']:+.4f}</td>
            <td class="{diff_class}">{comp['percentage']:+.2f}%</td>
        </tr>
"""
        
        html += """
    </table>
</body>
</html>
"""
        
        return html

