#!/bin/bash
#SBATCH --job-name=run_arch_experiments_pangu
#SBATCH --account=pi-jfranke
#SBATCH --output=run_arch_experiments_pangu%J.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8 
# lat must be between -90 and 90
# lon must be between 0 and 360 (0 is at prime meridian)
#
source .venv/bin/activate

# Auto-detect environment based on hostname
hostname=$(hostname)
if [[ "$hostname" == "oMac.local" ]]; then
    # laptop
    data_dir="/Users/ohouck/globus/forecast_data/raw/"
    output_dir="/Users/ohouck/globus/forecast_data/processed/finetuning_output/"
elif [[ "$hostname" == *"midway3"* ]]; then
    # midway
    data_dir="/project/jfranke/ozma/forecast_data/raw/"
    output_dir="/project/jfranke/ozma/forecast_data/processed/finetuning_output/"
else
    echo "Unknown environment. Hostname: $hostname"
    exit 1
fi

# ============================================================================
# TRAINING/OUTPUT VARIABLE PAIRS
# ============================================================================
Each entry is a paired configuration: "training_vars|output_vars"
Training vars are space-separated variables used as model inputs
Output vars are space-separated variables the model predicts

Examples:
  "2m_temperature|2m_temperature" - Predict temp using only temp
  "2m_temperature 10m_u_component_of_wind|2m_temperature" - Predict temp using temp and wind

training_output_vars=(
    # Minimal: Use only the output variable for training
    "2m_temperature|2m_temperature"

    # partial:use 3 vars
    # "2m_temperature temperature_1000hPa specific_humidity_1000hPa|2m_temperature"

    # "10m_wind_speed|10m_wind_speed"
)

subregions=(6x6)
regions=("india")
all_lead_times=(24 120 216)
nn_architectures=("mlp" "unet")
model_names=("pangu")
loss_functions=("mse")
# Define bootstrap regions
bootstrap_regions=("temperate" "tropical" "arid" "flat" "hilly" "mountainous")

train_start="2018-01-01"
train_end="2021-12-31"
test_start="2022-01-01"
test_end="2022-12-31"
model_name="pangu"
region="india"
subregion="6x6"

# # ============================================================================
# # PLAIN MLP AND UNET EXPERIMENTS  (no ensemble)
# # Produces standard early-stopping MLP and UNet zarrs for the arch comparison.
# # ============================================================================
# echo ""
# echo "========================================================"
# echo "Running plain MLP / UNet experiments"
# echo "========================================================"

# for nn_architecture in "${nn_architectures[@]}"; do
#     for var_pair in "${training_output_vars[@]}"; do
#         IFS='|' read -r training_vars output_vars <<< "$var_pair"

#         cmd="python3 finetuning/finetune.py \
#             --data_dir=\"$data_dir\" \
#             --output_dir=\"$output_dir\" \
#             --training_vars $training_vars \
#             --output_vars $output_vars \
#             --train_start=\"$train_start\" --train_end=\"$train_end\" \
#             --test_start=\"$test_start\" --test_end=\"$test_end\" \
#             --model_name=\"$model_name\" \
#             --region=\"$region\" \
#             --subregion=\"$subregion\" \
#             --lead_time_hours ${all_lead_times[@]} \
#             --nn_architecture=\"$nn_architecture\""

#         eval $cmd
#     done
# done

# # ============================================================================
# # SNAPSHOT ENSEMBLE MLP EXPERIMENTS
# # snapshot_ensemble=3 produces _snapshot3 files matched by SNAPSHOT_RUNS in
# # figures_finetuning.py → plot_arch_experiment_results.
# # ============================================================================
# snapshot_ensemble_runs=3   # must match SNAPSHOT_RUNS in figures_finetuning.py

# echo ""
# echo "========================================================"
# echo "Running MLP Snapshot Ensemble experiments (${snapshot_ensemble_runs} runs, T0=30)"
# echo "========================================================"

# for var_pair in "${training_output_vars[@]}"; do
#     IFS='|' read -r training_vars output_vars <<< "$var_pair"

#     cmd="python3 finetuning/finetune.py \
#         --data_dir=\"$data_dir\" \
#         --output_dir=\"$output_dir\" \
#         --training_vars $training_vars \
#         --output_vars $output_vars \
#         --train_start=\"$train_start\" --train_end=\"$train_end\" \
#         --test_start=\"$test_start\" --test_end=\"$test_end\" \
#         --model_name=\"$model_name\" \
#         --region=\"$region\" \
#         --subregion=\"$subregion\" \
#         --lead_time_hours ${all_lead_times[@]} \
#         --nn_architecture=mlp \
#         --snapshot_ensemble=${snapshot_ensemble_runs} \
#         --snapshot_T0=30 --snapshot_T_mult=1"

#     eval $cmd
# done

# ============================================================================
# BLOCK LTHO (LEAVE-THREE-OUT) ENSEMBLE EXPERIMENTS  ← new best method
# Trains 4 single-year models × 21 snapshots each (T0=10) = 84 total predictions.
# Uses val-loss weighted averaging automatically.  Training: ~0.5 min.
# Only the minimal (single-variable) configuration is run; multi-variable input
# was tested and confirmed to hurt performance for the k=3 block size.
# Produces _blockk3_snapshot1 files matched by BLOCK_LTHO_RUNS in
# figures_finetuning.py → plot_arch_experiment_results.
# ============================================================================
echo ""
echo "========================================================"
echo "Running Block LTHO (k=3) ensemble experiments"
echo "========================================================"
for var_pair in "${training_output_vars[@]}"; do
    IFS='|' read -r training_vars output_vars <<< "$var_pair"

    cmd="python3 finetuning/finetune.py \
        --data_dir=\"$data_dir\" \
        --output_dir=\"$output_dir\" \
        --training_vars $training_vars \
        --output_vars $output_vars \
        --train_start=\"$train_start\" --train_end=\"$train_end\" \
        --test_start=\"$test_start\" --test_end=\"$test_end\" \
        --model_name=\"$model_name\" \
        --region=\"$region\" \
        --subregion=\"$subregion\" \
        --lead_time_hours ${all_lead_times[@]} \
        --nn_architecture=mlp \
        --block_ensemble \
        --block_holdout=3 \
        --snapshot_ensemble=1 \
        --snapshot_T0=10 --snapshot_T_mult=1"

    eval $cmd
done
