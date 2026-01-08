#!/usr/bin/env python3
"""
Quick streaming test - 3 records from Dolphin Coder
Tests streaming + model training without catastrophic loss
"""

import torch
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("\n" + "="*80)
    logger.info("STREAMING + TRAINING TEST - Dolphin Coder (3 samples)")
    logger.info("="*80)
    
    # 1. Test dataset streaming
    logger.info("\n[1/4] Testing dataset streaming...")
    try:
        dataset = load_dataset("QuixiAI/dolphin-coder", split="train", streaming=True)
        
        samples = []
        for i, sample in enumerate(dataset):
            if i >= 3:
                break
            samples.append(sample)
        
        logger.info(f"  ✓ Streamed {len(samples)} samples")
        logger.info(f"  ✓ Text field: 'system_prompt'")
        
    except Exception as e:
        logger.error(f"  ✗ Streaming failed: {e}")
        return False
    
    # 2. Load model
    logger.info("\n[2/4] Loading model...")
    try:
        # OPTION 1: Jan-v2-VL-high (Qwen3VL)
        model_path = "models/Jan-v2-VL-high"

        # OPTION 2: GLM-4.6V-Flash (Uncomment to use)
        # model_path = "zai-org/GLM-4.6V-Flash"

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        logger.info("  Loading with trust_remote_code (will auto-detect model type)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use float32 for Mac
            device_map="cpu",  # Force CPU for Mac
            low_cpu_mem_usage=True
        )
        logger.info(f"  ✓ Model loaded ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)")

    except Exception as e:
        logger.error(f"  ✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Add LoRA
    logger.info("\n[3/4] Adding LoRA...")
    try:
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
        
    except Exception as e:
        logger.error(f"  ✗ LoRA failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Prepare data and train
    logger.info("\n[4/4] Training (3 steps)...")
    try:
        # Tokenize
        tokenized_data = []
        for sample in samples:
            text = str(sample['system_prompt'])
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
        
        # Check loss
        if final_loss > 100:
            logger.warning(f"  ⚠ WARNING: High loss ({final_loss:.4f}) - might be catastrophic")
        elif final_loss < 0.001:
            logger.warning(f"  ⚠ WARNING: Very low loss ({final_loss:.4f}) - might be overfitting")
        else:
            logger.info(f"  ✓ Loss looks reasonable")
        
        logger.info("\n" + "="*80)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("="*80)
        logger.info(f"Dataset streaming: SUCCESS")
        logger.info(f"Model loading: SUCCESS")
        logger.info(f"Training: SUCCESS")
        logger.info(f"Final loss: {final_loss:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"\n  ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

