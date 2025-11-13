#!/bin/bash
# Architecture Experiment Runner for India Region (6x6)
# Compares 4 architectures for improving Pangu forecasts
#
# Input variables:
#   - 2m_temperature
#   - 10m_u_component_of_wind
#   - 10m_v_component_of_wind
#   - temperature_1000hPa
#   - specific_humidity_1000hPa
#   - geopotential_1000hPa
#
# Output variable:
#   - 2m_temperature
#
# Lead times: 24h, 72h, 144h (1, 3, 6 days)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR="/Users/ohouck/globus/forecast_data/raw"
OUTPUT_DIR="/Users/ohouck/globus/forecast_data/processed/architecture_experiments"
REGION=india
SUBREGION=6x6
MODEL=pangu

# Training period 
TRAIN_START=2018-01-01
TRAIN_END=2021-12-31

# Test period 
TEST_START=2022-01-01
TEST_END=2022-12-31

# Variables
TRAINING_VARS="2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind temperature_1000hPa specific_humidity_1000hPa geopotential_1000hPa"
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
        grep "MSE original" ${log_file} || echo "  (Results not found in log)"
        echo ""
    else
        echo -e "${RED}✗ Experiment ${exp_name} FAILED${NC}"
        echo "  Check log: ${log_file}"
        exit 1
    fi
}

# ----------------------------------------------------------------------------
# Experiment 3: UNet_Light
# Lightweight UNet with 64 base channels
# ----------------------------------------------------------------------------
run_experiment \
    "unet_light" \
    "Lightweight UNet: 64 base channels" \
    "unet" \
    "--unet_hidden_dim=64 --unet_dropout=0.1"

# ----------------------------------------------------------------------------
# Experiment 4: UNet_Deep
# Deep UNet with 128 base channels
# ----------------------------------------------------------------------------
run_experiment \
    "unet_deep" \
    "Deep UNet: 128 base channels" \
    "unet" \
    "--unet_hidden_dim=128 --unet_dropout=0.15"

# ----------------------------------------------------------------------------
# Experiment 1: MLP_Deep
# Deep MLP with 6 hidden layers, 1024 neurons each
# ----------------------------------------------------------------------------
run_experiment \
    "mlp_deep" \
    "Deep MLP: 6 layers × 1024 neurons" \
    "mlp" \
    "--mlp_hidden_dim=1024 --mlp_num_layers=6 --mlp_dropout=0.25"
# ----------------------------------------------------------------------------
# Experiment 2: MLP_Wide
# Wide MLP with 3 hidden layers, 2048 neurons each
# ----------------------------------------------------------------------------
run_experiment \
    "mlp_wide" \
    "Wide MLP: 3 layers × 2048 neurons" \
    "mlp" \
    "--mlp_hidden_dim=2048 --mlp_num_layers=3 --mlp_dropout=0.3"


# ----------------------------------------------------------------------------
# All experiments complete!
# ----------------------------------------------------------------------------
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ALL EXPERIMENTS COMPLETED!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo "Logs saved to: ${LOG_DIR}"
echo ""
echo "To analyze results, run:"
echo "  python3 analyze_architecture_results.py"
