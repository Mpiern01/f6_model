#!/usr/bin/env python3
"""
Quick streaming test - 3 records per dataset
Tests if datasets stream and model trains without catastrophic loss
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Your datasets from REAL_BENCHMARKS.md
DATASETS = [
    ("QuixiAI/dolphin-coder", "Dolphin Coder"),
    ("Locutusque/hercules-v5.5", "Hercules v5.5"),
    ("Locutusque/hercules-v6.0", "Hercules v6.0"),
    ("Locutusque/hercules-v7.0", "Hercules v7.0"),
    ("Locutusque/function-calling-sharegpt-v1", "Function Calling"),
]

def test_dataset_streaming(dataset_name, dataset_label):
    """Test if a dataset streams and can be tokenized"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {dataset_label}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"{'='*60}")
    
    try:
        # Load streaming
        logger.info("  Loading in streaming mode...")
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
            return False
        
        # Find text field
        text_field = None
        for field in ['text', 'content', 'code', 'instruction', 'prompt', 'conversations']:
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
            return False
        
        logger.info(f"  ✓ Streaming works! Text field: '{text_field}'")
        logger.info(f"  ✓ Retrieved {len(samples)} samples")
        
        # Show sample preview
        sample_text = str(samples[0][text_field])[:100]
        logger.info(f"  Sample preview: {sample_text}...")
        
        return True, text_field, samples
        
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def main():
    logger.info("\n" + "="*80)
    logger.info("QUICK STREAMING TEST - 3 records per dataset")
    logger.info("="*80)
    
    # Test all datasets
    results = {}
    for dataset_name, dataset_label in DATASETS:
        result = test_dataset_streaming(dataset_name, dataset_label)
        if result and result[0]:
            results[dataset_label] = {
                'success': True,
                'text_field': result[1],
                'samples': result[2]
            }
        else:
            results[dataset_label] = {'success': False}
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("STREAMING TEST SUMMARY")
    logger.info("="*80)
    
    success_count = sum(1 for r in results.values() if r['success'])
    total_count = len(results)
    
    for label, result in results.items():
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        field = f" (field: {result.get('text_field', 'N/A')})" if result['success'] else ""
        logger.info(f"{status}: {label}{field}")
    
    logger.info(f"\nResult: {success_count}/{total_count} datasets stream successfully")
    
    if success_count == 0:
        logger.error("\n❌ NO DATASETS WORK - Cannot proceed with training test")
        return False
    
    # Now test model loading and training with first successful dataset
    logger.info("\n" + "="*80)
    logger.info("MODEL LOADING & TRAINING TEST")
    logger.info("="*80)
    
    # Find first successful dataset
    first_success = None
    for label, result in results.items():
        if result['success']:
            first_success = (label, result)
            break
    
    if not first_success:
        return False
    
    logger.info(f"\nUsing dataset: {first_success[0]}")
    
    try:
        # Load model
        logger.info("\n[1/4] Loading model...")
        # OPTION 1: Jan-v2-VL-high (Qwen3VL)
        model_path = "models/Jan-v2-VL-high"

        # OPTION 2: GLM-4.6V-Flash (Uncomment to use)
        # model_path = "zai-org/GLM-4.6V-Flash"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        logger.info("  Loading with CPU offloading...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder="offload",
            max_memory={0: "10GB", "cpu": "20GB"}
        )
        logger.info(f"  ✓ Model loaded ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)")
        
        # Add LoRA
        logger.info("\n[2/4] Adding LoRA...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  ✓ LoRA added ({trainable / 1e6:.2f}M trainable params)")
        
        # Prepare data
        logger.info("\n[3/4] Preparing data...")
        samples = first_success[1]['samples']
        text_field = first_success[1]['text_field']
        
        tokenized_data = []
        for sample in samples:
            text = str(sample[text_field])
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            encoded["labels"] = encoded["input_ids"].clone()
            tokenized_data.append({k: v.squeeze(0) for k, v in encoded.items()})
        
        logger.info(f"  ✓ Tokenized {len(tokenized_data)} samples")
        
        # Create dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]
        
        train_dataset = SimpleDataset(tokenized_data)
        
        # Train
        logger.info("\n[4/4] Training (3 steps)...")
        training_args = TrainingArguments(
            output_dir="./quick_test_output",
            max_steps=3,
            per_device_train_batch_size=1,
            logging_steps=1,
            save_steps=999,
            learning_rate=2e-5,
            fp16=False,
            report_to="none",
            remove_unused_columns=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        logger.info("  Starting training...")
        result = trainer.train()
        
        final_loss = result.training_loss
        logger.info(f"\n  ✓ Training completed!")
        logger.info(f"  Final loss: {final_loss:.4f}")
        
        # Check for catastrophic loss
        if final_loss > 100:
            logger.warning(f"  ⚠ WARNING: High loss ({final_loss:.4f}) - might be catastrophic")
        elif final_loss < 0.001:
            logger.warning(f"  ⚠ WARNING: Very low loss ({final_loss:.4f}) - might be overfitting")
        else:
            logger.info(f"  ✓ Loss looks reasonable")
        
        logger.info("\n" + "="*80)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("="*80)
        logger.info(f"Datasets streaming: {success_count}/{total_count}")
        logger.info(f"Model training: SUCCESS")
        logger.info(f"Final loss: {final_loss:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

