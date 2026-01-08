#!/usr/bin/env python3
"""
REAL Training Test - Actually train the model for a few steps
No bullshit, no mocks, REAL training
"""

import torch
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoTokenizer, AutoModel, AutoProcessor, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_real_training():
    """Actually train the model for 10 steps."""

    logger.info("=" * 80)
    logger.info("REAL TRAINING TEST - Vision-Language Model")
    logger.info("=" * 80)

    # OPTION 1: Jan-v2-VL-high (Qwen3VL)
    model_path = "models/Jan-v2-VL-high"

    # OPTION 2: GLM-4.6V-Flash (Uncomment to use)
    # model_path = "zai-org/GLM-4.6V-Flash"

    # 1. Load model and tokenizer
    logger.info(f"\n[1/6] Loading model from {model_path}...")
    try:
        # This is a vision-language model (Qwen3VL or GLM-4V)
        # Load with CPU offloading for Mac
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        logger.info("  Loading with CPU offloading (Mac - no CUDA)...")
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder="offload",
            max_memory={0: "10GB", "cpu": "20GB"}  # Use CPU offloading
        )
        logger.info(f"✓ Model loaded: {model.config.model_type}")
        logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        logger.info(f"  Model type: Vision-Language Model (Qwen3VL)")
        logger.info(f"  Device map: CPU offloading enabled")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 2. Add LoRA adapters
    logger.info("\n[2/6] Adding LoRA adapters...")
    try:
        # Find attention modules in the model
        target_modules = []
        for name, module in model.named_modules():
            if any(x in name for x in ["q_proj", "v_proj", "k_proj", "o_proj", "qkv_proj"]):
                target_modules.append(name.split('.')[-1])

        if not target_modules:
            # Fallback to common names
            target_modules = ["q_proj", "v_proj"]

        target_modules = list(set(target_modules))[:4]  # Limit to 4 unique modules

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"✓ LoRA added: {trainable_params / 1e6:.2f}M trainable params")
        logger.info(f"  Target modules: {target_modules}")
    except Exception as e:
        logger.error(f"✗ Failed to add LoRA: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Load streaming dataset from YOUR registry
    logger.info("\n[3/6] Loading streaming dataset from YOUR registry...")
    try:
        # Use actual datasets from your registry
        # Try multiple datasets in case one fails
        datasets_to_try = [
            ("QuixiAI/dolphin-coder", "Dolphin Coder"),
            ("microsoft/rStar-Coder", "rStar Coder"),
            ("Nilaksh404/qwen3-coder-30b", "Qwen3 Coder"),
        ]

        dataset = None
        for hf_path, name in datasets_to_try:
            try:
                logger.info(f"  Trying {name} ({hf_path})...")
                dataset = load_dataset(
                    hf_path,
                    split="train",
                    streaming=True,
                    trust_remote_code=True
                )
                logger.info(f"✓ Dataset loaded: {name}")
                break
            except Exception as e:
                logger.warning(f"  Failed to load {name}: {str(e)[:100]}")
                continue

        if dataset is None:
            raise Exception("All datasets failed to load")

    except Exception as e:
        logger.error(f"✗ Failed to load any dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Prepare data
    logger.info("\n[4/6] Preparing training data...")
    try:
        # Take only 50 samples for quick test
        samples = []
        logger.info("  Collecting 50 samples...")
        for i, sample in enumerate(dataset):
            if i >= 50:
                break
            samples.append(sample)
            if (i + 1) % 10 == 0:
                logger.info(f"    Collected {i + 1}/50 samples...")

        logger.info(f"✓ Collected {len(samples)} samples")

        # Find text field (could be 'text', 'content', 'code', etc.)
        text_field = None
        if samples:
            for field in ['text', 'content', 'code', 'instruction', 'prompt']:
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
            raise Exception(f"Could not find text field in sample: {samples[0].keys() if samples else 'no samples'}")

        logger.info(f"  Using text field: '{text_field}'")

        # Tokenize samples
        texts = [s[text_field] for s in samples if text_field in s and s[text_field]]
        logger.info(f"  Tokenizing {len(texts)} texts...")

        tokenized_data = []
        for text in texts:
            encoded = tokenizer(
                str(text),
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            encoded["labels"] = encoded["input_ids"].clone()
            tokenized_data.append({k: v.squeeze(0) for k, v in encoded.items()})

        logger.info(f"✓ Data prepared ({len(tokenized_data)} samples)")

    except Exception as e:
        logger.error(f"✗ Failed to prepare data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Setup training
    logger.info("\n[5/6] Setting up training...")
    try:
        # Create a simple dataset class
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        train_dataset = SimpleDataset(tokenized_data)

        training_args = TrainingArguments(
            output_dir="./test_training_output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_steps=10,  # Only 10 steps for test
            logging_steps=1,
            save_steps=10,
            learning_rate=2e-5,
            warmup_steps=2,
            fp16=False,  # Disable fp16 for Mac (MPS doesn't support it)
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        logger.info("✓ Trainer configured")
    except Exception as e:
        logger.error(f"✗ Failed to setup trainer: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. ACTUALLY TRAIN
    logger.info("\n[6/6] ACTUALLY TRAINING FOR 10 STEPS...")
    logger.info("-" * 80)
    try:
        trainer.train()
        logger.info("-" * 80)
        logger.info("✓ Training completed successfully!")
        
        # Get final loss
        train_result = trainer.state.log_history
        if train_result:
            final_loss = train_result[-1].get('loss', 'N/A')
            logger.info(f"  Final loss: {final_loss}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("\n" + "=" * 80)
    logger.info("REAL TRAINING TEST")
    logger.info("This will ACTUALLY train your model for 10 steps")
    logger.info("=" * 80)
    
    success = test_real_training()
    
    logger.info("\n" + "=" * 80)
    if success:
        logger.info("✅ REAL TRAINING TEST PASSED")
        logger.info("The model CAN be trained successfully!")
    else:
        logger.info("❌ REAL TRAINING TEST FAILED")
        logger.info("There are issues that need to be fixed")
    logger.info("=" * 80)
    
    sys.exit(0 if success else 1)

