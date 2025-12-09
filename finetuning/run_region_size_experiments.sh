#!/bin/bash
#SBATCH --job-name=run_region_size_experiments
#SBATCH --account=pi-jfranke
#SBATCH --output=run_region_size_experiments%J.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
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

training_output_vars=(
    # Minimal: Use only the output variable for training
    "2m_temperature|2m_temperature"
    # "10m_wind_speed|10m_wind_speed"
)

subregions=(20x20 15x15 10x10 8x8 6x6 4x4)
regions=("finland" "ethiopia")
all_lead_times=(24 120 216)
nn_architecture="mlp"
model_name="pangu"

for region in "${regions[@]}"; do
    for subregion in "${subregions[@]}"; do
        for var_pair in "${training_output_vars[@]}"; do
            # Split the pair into training_vars and output_vars
            IFS='|' read -r training_vars output_vars <<< "$var_pair"
            train_start="2018-01-01"
            train_end="2021-12-31"
            test_start="2022-01-01"
            test_end="2022-12-31"

            # Build base command
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
                --nn_architecture=\"$nn_architecture\""
        
            # Execute command
            eval $cmd
        done
    done
done
