#!/usr/bin/env python3
"""
Run Gap Analysis
Analyze dataset coverage and identify gaps

MIT-level engineering: Comprehensive gap analysis
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dataset_registry.gap_analysis import analyze_gaps
from dataset_registry.registry import get_all_datasets, get_datasets_by_category
import json

def main():
    """Run gap analysis."""
    print("=" * 80)
    print("F6 StreamTrain - Dataset Gap Analysis")
    print("=" * 80)
    
    # Analyze gaps
    gaps = analyze_gaps()
    
    print(f"\nCurrent Categories: {len(gaps['current_categories'])}")
    print(f"Missing Categories: {len(gaps['missing_categories'])}")
    print(f"Required Gaps: {len(gaps['required_gaps'])}")
    
    print("\n" + "=" * 80)
    print("Gap Details:")
    print("=" * 80)
    
    for category, info in gaps['gaps'].items():
        status = "⚠️ REQUIRED" if info['required'] else "ℹ️ Optional"
        print(f"\n{status} - {category.upper()}")
        print(f"  Description: {info['description']}")
        print(f"  Suggested Datasets: {', '.join(info['suggested_datasets'][:5])}")
    
    print("\n" + "=" * 80)
    print("Current Dataset Coverage:")
    print("=" * 80)
    
    all_datasets = get_all_datasets()
    categories = {}
    for ds in all_datasets:
        cat = ds.category
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} datasets")
    
    print("\n" + "=" * 80)
    print("✅ Gap Analysis Complete")
    print("=" * 80)
    
    # Save results
    with open("gap_analysis_results.json", "w") as f:
        json.dump(gaps, f, indent=2)
    
    print("\nResults saved to: gap_analysis_results.json")

if __name__ == "__main__":
    main()

