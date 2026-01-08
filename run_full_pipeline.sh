#!/bin/bash
# Complete Training and Deployment Pipeline
# Pull model → Train → MLX → 4-bit Quantize

set -e  # Exit on error

echo "=========================================="
echo "F6 StreamTrain - Complete Pipeline"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Bootstrap environment
echo -e "${GREEN}Step 1: Bootstrapping environment...${NC}"
python3 main.py --stage 0 || {
    echo -e "${YELLOW}Warning: Environment bootstrap had issues, continuing...${NC}"
}

# Step 2: Pull and train model (with limited steps for testing)
echo -e "${GREEN}Step 2: Training model (limited to 10 steps for testing)...${NC}"
python3 train_and_deploy.py \
    --model-path "janhq/Jan-v2-VL-high" \
    --config "configs/stage1_midtrain.yaml" \
    --max-steps 10 \
    --skip-mlx \
    --skip-quantization || {
    echo -e "${RED}Training failed. Check logs above.${NC}"
    exit 1
}

# Step 3: Convert to MLX (if training succeeded)
if [ -d "checkpoints/stage1_midtrain/final" ] || [ -d "checkpoints/stage1_midtrain" ]; then
    echo -e "${GREEN}Step 3: Converting to MLX...${NC}"
    
    MODEL_PATH="checkpoints/stage1_midtrain/final"
    if [ ! -d "$MODEL_PATH" ]; then
        # Use latest checkpoint
        MODEL_PATH=$(ls -td checkpoints/stage1_midtrain/checkpoint-* 2>/dev/null | head -1)
    fi
    
    if [ -n "$MODEL_PATH" ] && [ -d "$MODEL_PATH" ]; then
        python3 mlx/export_mlx_vlm.py \
            --model-path "$MODEL_PATH" \
            --output-path "${MODEL_PATH}_mlx" \
            --dtype bfloat16 || {
            echo -e "${YELLOW}MLX conversion failed. Continuing...${NC}"
        }
    else
        echo -e "${YELLOW}No trained model found, skipping MLX conversion${NC}"
    fi
else
    echo -e "${YELLOW}Skipping MLX conversion (no trained model)${NC}"
fi

# Step 4: 4-bit quantization (if MLX conversion succeeded)
MLX_MODEL=$(ls -td checkpoints/stage1_midtrain*_mlx 2>/dev/null | head -1)
if [ -n "$MLX_MODEL" ] && [ -d "$MLX_MODEL" ]; then
    echo -e "${GREEN}Step 4: 4-bit quantization...${NC}"
    
    python3 mlx/quantize_4bit.py \
        --model-path "$MLX_MODEL" \
        --output-path "${MLX_MODEL}_q4" \
        --method quarot || {
        echo -e "${YELLOW}Quantization failed. Trying standard method...${NC}"
        python3 mlx/quantize_4bit.py \
            --model-path "$MLX_MODEL" \
            --output-path "${MLX_MODEL}_q4" \
            --method standard || {
            echo -e "${RED}Quantization failed${NC}"
        }
    }
else
    echo -e "${YELLOW}Skipping quantization (no MLX model)${NC}"
fi

echo -e "${GREEN}=========================================="
echo "Pipeline Complete!"
echo "==========================================${NC}"

