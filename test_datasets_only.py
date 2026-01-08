#!/usr/bin/env python3
"""
Test ONLY dataset streaming - 3 records each
No model loading - just verify datasets work
"""

from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Your datasets from REAL_BENCHMARKS.md
DATASETS = [
    ("QuixiAI/dolphin-coder", "Dolphin Coder", None),
    ("Locutusque/hercules-v5.5", "Hercules v5.5", None),
    ("Locutusque/hercules-v6.0", "Hercules v6.0", "cleaned"),  # Needs config
    ("Locutusque/hercules-v7.0", "Hercules v7.0", None),
    ("Locutusque/function-calling-sharegpt-v1", "Function Calling", None),
]

def test_dataset(dataset_name, dataset_label, config_name=None):
    """Test if a dataset streams properly"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Testing: {dataset_label}")
    logger.info(f"Dataset: {dataset_name}")
    if config_name:
        logger.info(f"Config: {config_name}")
    logger.info(f"{'='*70}")
    
    try:
        # Load streaming
        logger.info("  Loading in streaming mode...")
        if config_name:
            dataset = load_dataset(dataset_name, config_name, split="train", streaming=True)
        else:
            dataset = load_dataset(dataset_name, split="train", streaming=True)
        
        # Get 3 samples
        samples = []
        logger.info("  Fetching 3 samples...")
        for i, sample in enumerate(dataset):
            if i >= 3:
                break
            samples.append(sample)
            logger.info(f"    Sample {i+1}/3 fetched")
        
        if len(samples) == 0:
            logger.error(f"  ✗ No samples retrieved")
            return False, None
        
        # Find text field
        text_field = None
        for field in ['text', 'content', 'code', 'instruction', 'prompt', 'conversations', 'system_prompt']:
            if field in samples[0]:
                text_field = field
                break
        
        if text_field is None:
            # Use first string field
            for k, v in samples[0].items():
                if isinstance(v, str):
                    text_field = k
                    break
        
        if text_field is None:
            logger.error(f"  ✗ No text field found. Fields: {list(samples[0].keys())}")
            return False, None
        
        logger.info(f"  ✓ Streaming works!")
        logger.info(f"  ✓ Text field: '{text_field}'")
        logger.info(f"  ✓ Retrieved {len(samples)} samples")
        
        # Show sample preview
        sample_text = str(samples[0][text_field])[:80]
        logger.info(f"  ✓ Sample preview: {sample_text}...")
        
        return True, text_field
        
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        return False, None

def main():
    logger.info("\n" + "="*80)
    logger.info("DATASET STREAMING TEST - 3 records per dataset")
    logger.info("="*80)
    
    # Test all datasets
    results = {}
    for dataset_name, dataset_label, config_name in DATASETS:
        success, text_field = test_dataset(dataset_name, dataset_label, config_name)
        results[dataset_label] = {
            'success': success,
            'text_field': text_field,
            'dataset_name': dataset_name,
            'config_name': config_name
        }
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    
    success_count = sum(1 for r in results.values() if r['success'])
    total_count = len(results)
    
    logger.info(f"\nDatasets tested: {total_count}")
    logger.info(f"Streaming successfully: {success_count}")
    logger.info(f"Failed: {total_count - success_count}\n")
    
    for label, result in results.items():
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        field = f" (field: {result.get('text_field', 'N/A')})" if result['success'] else ""
        logger.info(f"{status}: {label}{field}")
    
    if success_count == 0:
        logger.error("\n❌ NO DATASETS WORK")
        return False
    elif success_count == total_count:
        logger.info(f"\n✅ ALL {total_count} DATASETS STREAM SUCCESSFULLY!")
        return True
    else:
        logger.warning(f"\n⚠️  PARTIAL SUCCESS: {success_count}/{total_count} datasets work")
        logger.info("\nWorking datasets:")
        for label, result in results.items():
            if result['success']:
                config_str = f", config='{result['config_name']}'" if result['config_name'] else ""
                logger.info(f"  - {label}: {result['dataset_name']}{config_str}")
        return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

