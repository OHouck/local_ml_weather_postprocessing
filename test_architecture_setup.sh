#!/bin/bash
# Quick test of architecture experiment setup
# Tests one architecture with minimal data (just 1 week)

set -e

echo "Testing Architecture Experiment Setup"
echo "======================================"
echo ""

# Test with minimal parameters
REGION=india
SUBREGION=6x6
MODEL=pangu
ARCHITECTURE=mlp

# Very short period for quick test (1 week training, 2 days testing)
TRAIN_START=2018-01-01
TRAIN_END=2020-01-07
TEST_START=2020-01-08
TEST_END=2020-01-10

DATA_DIR=~/ai_weather_ag/data/raw
OUTPUT_DIR=~/ai_weather_ag/data/architecture_test

echo "Configuration:"
echo "  Region: ${REGION} (${SUBREGION})"
echo "  Model: ${MODEL}"
echo "  Architecture: ${ARCHITECTURE}"
echo "  Training: ${TRAIN_START} to ${TRAIN_END} (1 week - FAST TEST)"
echo "  Testing: ${TEST_START} to ${TEST_END} (3 days)"
echo ""
echo "This should take 5-15 minutes instead of hours."
echo ""
read -p "Press Enter to start test, or Ctrl+C to cancel..."

python3 finetuning/finetune.py \
    --region=${REGION} \
    --subregion=${SUBREGION} \
    --model_name=${MODEL} \
    --nn_architecture=${ARCHITECTURE} \
    --mlp_hidden_dim=512 \
    --mlp_num_layers=2 \
    --mlp_dropout=0.2 \
    --training_vars 2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind temperature_1000hPa specific_humidity_1000hPa geopotential_1000hPa \
    --output_vars 2m_temperature \
    --lead_time_hours 24 \
    --train_start=${TRAIN_START} \
    --train_end=${TRAIN_END} \
    --test_start=${TEST_START} \
    --test_end=${TEST_END} \
    --data_dir=${DATA_DIR} \
    --output_dir=${OUTPUT_DIR}

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ TEST SUCCESSFUL!"
    echo ""
    echo "Your environment is set up correctly."
    echo "You can now run the full experiments:"
    echo "  ./run_architecture_experiments.sh"
else
    echo ""
    echo "✗ TEST FAILED"
    echo ""
    echo "Please check:"
    echo "  1. All Python packages installed (torch, dask, xarray, pandas, numpy)"
    echo "  2. Network connection (for data download)"
    echo "  3. Sufficient disk space (~2GB for this test)"
fi
