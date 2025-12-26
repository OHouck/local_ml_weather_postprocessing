#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=transformer_tests
#SBATCH --account=pi-jfranke
#SBATCH --output=transformer_tests-%J.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# Transformer vs UNet vs Attention UNet Comparison
#
# Tests:
# 1. Transformer (baseline - best accuracy but slow)
# 2. Basic UNet (baseline - fast but less accurate)
# 3. Attention UNet with SE blocks (target: transformer accuracy at UNet speed)
# 4. Attention UNet with increased capacity
#
# Output variable: 2m_temperature
# Lead times: 24h, 72h, 144h (1, 3, 6 days)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Configuration
# Uncomment for local Mac development:
# DATA_DIR="/Users/ohouck/globus/forecast_data/raw"
# OUTPUT_DIR="/Users/ohouck/globus/forecast_data/processed/transformer_tests"

# Server configuration:
DATA_DIR="/project/jfranke/ozma/forecast_data/raw"
OUTPUT_DIR="/project/jfranke/ozma/forecast_data/processed/transformer_tests"
REGION=india
SUBREGION=6x6
MODEL=pangu

# Training period
TRAIN_START=2018-01-01
TRAIN_END=2021-12-31

# Test period
TEST_START=2022-01-01
TEST_END=2022-12-31

# Full variable set (best performance)
TRAINING_VARS="2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind temperature_1000hPa specific_humidity_1000hPa geopotential_1000hPa"
OUTPUT_VARS="2m_temperature"
LEAD_TIMES="24 72 144"

# Create output directory
mkdir -p ${OUTPUT_DIR}
LOG_DIR=${OUTPUT_DIR}/logs
mkdir -p ${LOG_DIR}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TRANSFORMER vs ATTENTION UNET TESTS${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Region: ${REGION} (${SUBREGION})"
echo "Model: ${MODEL}"
echo "Training: ${TRAIN_START} to ${TRAIN_END}"
echo "Testing: ${TEST_START} to ${TEST_END}"
echo "Lead times: ${LEAD_TIMES} hours"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Function to run experiment
run_experiment() {
    local exp_name=$1
    local description=$2
    local architecture=$3
    shift 3
    local arch_params="$@"

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Experiment: ${exp_name}${NC}"
    echo -e "${BLUE}${description}${NC}"
    echo -e "${BLUE}========================================${NC}"

    local log_file=${LOG_DIR}/${exp_name}_$(date +%Y%m%d_%H%M%S).log

    python3 -u finetuning/finetune.py \
        --region=${REGION} \
        --subregion=${SUBREGION} \
        --model_name=${MODEL} \
        --nn_architecture=${architecture} \
        --training_vars ${TRAINING_VARS} \
        --output_vars ${OUTPUT_VARS} \
        --lead_time_hours ${LEAD_TIMES} \
        --train_start=${TRAIN_START} \
        --train_end=${TRAIN_END} \
        --test_start=${TEST_START} \
        --test_end=${TEST_END} \
        --data_dir=${DATA_DIR} \
        --output_dir=${OUTPUT_DIR} \
        ${arch_params} \
        2>&1 | tee ${log_file}

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ Experiment ${exp_name} completed successfully${NC}"
        echo "  Log saved to: ${log_file}"

        # Extract and display key results
        echo ""
        echo "Key Results:"
        grep "Lead time" ${log_file} | grep "RMSE" || echo "  (Results not found in log)"
        grep "Training complete" ${log_file} || echo "  (Training time not found)"
        echo ""
    else
        echo -e "${RED}✗ Experiment ${exp_name} FAILED${NC}"
        echo "  Check log: ${log_file}"
        exit 1
    fi
}

echo -e "${YELLOW}Total experiments: 5${NC}"
echo ""

# ============================================================================
# GROUP 1: BASELINE ARCHITECTURES
# ============================================================================
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}GROUP 1: Baseline Architectures${NC}"
echo -e "${YELLOW}Transformer vs Basic UNet${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# ----------------------------------------------------------------------------
# Experiment 1.1: Transformer (Reduced Size for Speed)
# Original best: hidden_dim=256, num_layers=4, num_heads=8
# Reduced: hidden_dim=128, num_layers=3, num_heads=4 (for faster comparison)
# ----------------------------------------------------------------------------
run_experiment \
    "transformer_reduced" \
    "Transformer: 128 dim, 3 layers, 4 heads (reduced for speed)" \
    "transformer" \
    "--transformer_hidden_dim=128 --transformer_num_layers=3 --transformer_num_heads=4 --transformer_mlp_ratio=2.0 --transformer_dropout=0.1"

# ----------------------------------------------------------------------------
# Experiment 1.2: Basic UNet
# Standard UNet without attention mechanisms
# ----------------------------------------------------------------------------
run_experiment \
    "unet_basic" \
    "Basic UNet: 64 channels (no attention)" \
    "unet" \
    "--unet_hidden_dim=64 --unet_dropout=0.1"

# ============================================================================
# GROUP 2: ATTENTION UNET VARIATIONS
# ============================================================================
echo ""
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}GROUP 2: Attention UNet${NC}"
echo -e "${YELLOW}Testing attention mechanisms${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# ----------------------------------------------------------------------------
# Experiment 2.1: Attention UNet (Attention Gates + SE Blocks)
# Same capacity as basic UNet but with attention mechanisms
# ----------------------------------------------------------------------------
run_experiment \
    "unet_attention" \
    "Attention UNet: 64 channels + attention gates + SE blocks" \
    "unet" \
    "--unet_hidden_dim=64 --unet_dropout=0.1 --unet_use_attention --unet_use_residual"

# ----------------------------------------------------------------------------
# Experiment 2.2: Attention UNet (Higher Capacity)
# Increased capacity with attention mechanisms
# ----------------------------------------------------------------------------
run_experiment \
    "unet_attention_128" \
    "Attention UNet: 128 channels + attention gates + SE blocks" \
    "unet" \
    "--unet_hidden_dim=128 --unet_dropout=0.1 --unet_use_attention --unet_use_residual"

# ----------------------------------------------------------------------------
# Experiment 2.3: Attention UNet (Just Residual Connections)
# Test residual connections alone without attention
# ----------------------------------------------------------------------------
run_experiment \
    "unet_residual_only" \
    "UNet: 64 channels + residual connections (no attention)" \
    "unet" \
    "--unet_hidden_dim=64 --unet_dropout=0.1 --unet_use_residual"

# ============================================================================
# All experiments complete!
# ============================================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ALL 5 EXPERIMENTS COMPLETED!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo "Logs saved to: ${LOG_DIR}"
echo ""
echo -e "${YELLOW}Experiment Summary:${NC}"
echo "  Group 1: Baselines (2 experiments)"
echo "    - Transformer (reduced size for speed)"
echo "    - Basic UNet (no attention)"
echo "  Group 2: Attention UNet (3 experiments)"
echo "    - 64ch + attention gates + SE blocks + residual"
echo "    - 128ch + attention gates + SE blocks + residual"
echo "    - 64ch + residual only (no attention)"
echo ""
echo -e "${BLUE}Expected Performance Rankings:${NC}"
echo "  Speed (fastest → slowest):"
echo "    1. Basic UNet"
echo "    2. UNet Residual Only"
echo "    3. Attention UNet 64ch"
echo "    4. Attention UNet 128ch"
echo "    5. Transformer"
echo ""
echo "  Accuracy (target: Attention UNet matches Transformer):"
echo "    - Transformer: Current best (but slow)"
echo "    - Attention UNet 128ch: Target best (fast + accurate)"
echo "    - Attention UNet 64ch: Good balance"
echo "    - Basic UNet: Fastest (baseline accuracy)"
echo ""
echo -e "${YELLOW}To analyze results, compare RMSE values in logs:${NC}"
echo "  grep 'Lead time' ${LOG_DIR}/*_*.log | grep 'RMSE'"
echo ""
