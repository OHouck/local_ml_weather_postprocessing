#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=run_arch_experiments
#SBATCH --account=pi-jfranke
#SBATCH --output=run_arch_experiments-%J.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# Architecture Experiment Runner for India Region (6x6)
# Comprehensive comparison of MLP vs UNet architectures and input variable sets
#
# Tests:
# 1. MLP: Wide & Shallow vs Skinny & Deep vs Moderate
# 2. UNet: Different channel sizes (32, 64, 128, 256)
# 3. Input variables: Full set vs 2m_temperature only
#
# Output variable: 2m_temperature
# Lead times: 24h, 120h, 216h (1, 5, 9 days)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Configuration
# DATA_DIR="/Users/ohouck/globus/forecast_data/raw"
# OUTPUT_DIR="/Users/ohouck/globus/forecast_data/processed/architecture_experiments"

DATA_DIR="/project/jfranke/ozma/forecast_data/raw"
OUTPUT_DIR="/project/jfranke/ozma/forecast_data/processed/architecture_experiments"
REGION=india
SUBREGION=6x6
MODEL=pangu

# Training period
TRAIN_START=2018-01-01
TRAIN_END=2021-12-31

# Test period
TEST_START=2022-01-01
TEST_END=2022-12-31

# Full variable set
TRAINING_VARS_FULL="2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind temperature_1000hPa specific_humidity_1000hPa geopotential_1000hPa"
# Minimal variable set
TRAINING_VARS_MINIMAL="2m_temperature"
OUTPUT_VARS="2m_temperature"
LEAD_TIMES="24 120 216"

# Create output directory
mkdir -p ${OUTPUT_DIR}
LOG_DIR=${OUTPUT_DIR}/logs
mkdir -p ${LOG_DIR}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ARCHITECTURE EXPERIMENTS${NC}"
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
    local training_vars=$4
    shift 4
    local arch_params="$@"

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Experiment: ${exp_name}${NC}"
    echo -e "${BLUE}${description}${NC}"
    echo -e "${BLUE}========================================${NC}"

    local log_file=${LOG_DIR}/${exp_name}_$(date +%Y%m%d_%H%M%S).log

    python3 -u finetuning/post_process.py \
        --region=${REGION} \
        --subregion=${SUBREGION} \
        --model_name=${MODEL} \
        --nn_architecture=${architecture} \
        --training_vars ${training_vars} \
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
        grep "Lead time" ${log_file} | grep "MSE" || echo "  (Results not found in log)"
        grep "Training complete" ${log_file} || echo "  (Training time not found)"
        echo ""
    else
        echo -e "${RED}✗ Experiment ${exp_name} FAILED${NC}"
        echo "  Check log: ${log_file}"
        exit 1
    fi
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ARCHITECTURE EXPERIMENTS${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Region: ${REGION} (${SUBREGION})"
echo "Model: ${MODEL}"
echo "Training: ${TRAIN_START} to ${TRAIN_END}"
echo "Testing: ${TEST_START} to ${TEST_END}"
echo "Lead times: ${LEAD_TIMES} hours"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo -e "${YELLOW}Total experiments: 9${NC}"
echo ""

# ============================================================================
# GROUP 1: MLP ARCHITECTURE VARIATIONS (FULL VARIABLES)
# ============================================================================
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}GROUP 1: MLP Architecture Variations${NC}"
echo -e "${YELLOW}Using full variable set (6 variables)${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# ----------------------------------------------------------------------------
# Experiment 1.1: MLP Wide Shallow
# Wide and shallow: 3 layers × 2048 neurons
# ----------------------------------------------------------------------------
run_experiment \
    "mlp_wide_shallow" \
    "MLP Wide Shallow: 3 layers × 2048 neurons (Full vars)" \
    "mlp" \
    "${TRAINING_VARS_FULL}" \
    "--mlp_hidden_dim=2048 --mlp_num_layers=3 --mlp_dropout=0.3"

# ----------------------------------------------------------------------------
# Experiment 1.2: MLP Moderate (Baseline)
# Moderate: 6 layers × 1024 neurons
# ----------------------------------------------------------------------------
run_experiment \
    "mlp_moderate" \
    "MLP Moderate: 6 layers × 1024 neurons (Full vars)" \
    "mlp" \
    "${TRAINING_VARS_FULL}" \
    "--mlp_hidden_dim=1024 --mlp_num_layers=6 --mlp_dropout=0.25"

# ----------------------------------------------------------------------------
# Experiment 1.3: MLP Skinny Deep
# Skinny and deep: 8 layers × 512 neurons
# ----------------------------------------------------------------------------
run_experiment \
    "mlp_skinny_deep" \
    "MLP Skinny Deep: 8 layers × 512 neurons (Full vars)" \
    "mlp" \
    "${TRAINING_VARS_FULL}" \
    "--mlp_hidden_dim=512 --mlp_num_layers=8 --mlp_dropout=0.2"

# ============================================================================
# GROUP 2: UNET ARCHITECTURE VARIATIONS (FULL VARIABLES)
# ============================================================================
echo ""
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}GROUP 2: UNet Architecture Variations${NC}"
echo -e "${YELLOW}Using full variable set (6 variables)${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# ----------------------------------------------------------------------------
# Experiment 2.1: UNet Light
# 32 base channels
# ----------------------------------------------------------------------------
run_experiment \
    "unet_light" \
    "UNet Light: 32 channels (Full vars)" \
    "unet" \
    "${TRAINING_VARS_FULL}" \
    "--unet_hidden_dim=32 --unet_dropout=0.1"

# ----------------------------------------------------------------------------
# Experiment 2.2: UNet Medium
# 64 base channels
# ----------------------------------------------------------------------------
run_experiment \
    "unet_medium" \
    "UNet Medium: 64 channels (Full vars)" \
    "unet" \
    "${TRAINING_VARS_FULL}" \
    "--unet_hidden_dim=64 --unet_dropout=0.1"

# ----------------------------------------------------------------------------
# Experiment 2.3: UNet Heavy
# 128 base channels
# ----------------------------------------------------------------------------
run_experiment \
    "unet_heavy" \
    "UNet Heavy: 128 channels (Full vars)" \
    "unet" \
    "${TRAINING_VARS_FULL}" \
    "--unet_hidden_dim=128 --unet_dropout=0.15"

# ----------------------------------------------------------------------------
# Experiment 2.4: UNet Very Heavy
# 256 base channels
# ----------------------------------------------------------------------------
run_experiment \
    "unet_very_heavy" \
    "UNet Very Heavy: 256 channels (Full vars)" \
    "unet" \
    "${TRAINING_VARS_FULL}" \
    "--unet_hidden_dim=256 --unet_dropout=0.2"

# ============================================================================
# GROUP 3: INPUT VARIABLE COMPARISON (BEST ARCHITECTURES)
# ============================================================================
echo ""
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}GROUP 3: Input Variable Comparison${NC}"
echo -e "${YELLOW}Using 2m_temperature only${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# ----------------------------------------------------------------------------
# Experiment 3.1: MLP Moderate with Minimal Variables
# Best MLP architecture with only 2m_temperature
# ----------------------------------------------------------------------------
run_experiment \
    "mlp_moderate_minimal" \
    "MLP Moderate: 6 layers × 1024 neurons (2m_temp only)" \
    "mlp" \
    "${TRAINING_VARS_MINIMAL}" \
    "--mlp_hidden_dim=1024 --mlp_num_layers=6 --mlp_dropout=0.25"

# ----------------------------------------------------------------------------
# Experiment 3.2: UNet Medium with Minimal Variables
# Best UNet architecture with only 2m_temperature
# ----------------------------------------------------------------------------
run_experiment \
    "unet_medium_minimal" \
    "UNet Medium: 64 channels (2m_temp only)" \
    "unet" \
    "${TRAINING_VARS_MINIMAL}" \
    "--unet_hidden_dim=64 --unet_dropout=0.1"

# ============================================================================
# All experiments complete!
# ============================================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ALL 9 EXPERIMENTS COMPLETED!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo "Logs saved to: ${LOG_DIR}"
echo ""
echo -e "${YELLOW}Experiment Summary:${NC}"
echo "  Group 1: MLP Variations (3 experiments)"
echo "    - Wide Shallow, Moderate, Skinny Deep"
echo "  Group 2: UNet Variations (4 experiments)"
echo "    - Light, Medium, Heavy, Very Heavy"
echo "  Group 3: Input Variable Comparison (2 experiments)"
echo "    - MLP and UNet with minimal variables"
echo ""
echo "To analyze results, run:"
echo "  python3 analyze_architecture_results.py"
