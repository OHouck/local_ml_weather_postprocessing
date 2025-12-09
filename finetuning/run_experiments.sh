#!/bin/bash
#SBATCH --job-name=run_experiments_pangu_mlp
#SBATCH --account=pi-jfranke
#SBATCH --output=run_experiments_pangu_mlp%J.txt
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
# Each entry is a paired configuration: "training_vars|output_vars"
# Training vars are space-separated variables used as model inputs
# Output vars are space-separated variables the model predicts
#
# Examples:
#   "2m_temperature|2m_temperature" - Predict temp using only temp
#   "2m_temperature 10m_u_component_of_wind|2m_temperature" - Predict temp using temp and wind
#
training_output_vars=(
    # Minimal: Use only the output variable for training
    "2m_temperature|2m_temperature"

    # partial:use 3 vars
    # "2m_temperature temperature_1000hPa specific_humidity_1000hPa|2m_temperature"

    # Full: Use all 6 variables for training (best performance from experiments)
    # "2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind temperature_1000hPa specific_humidity_1000hPa geopotential_1000hPa|2m_temperature"

    # "10m_wind_speed|10m_wind_speed"
)

subregions=(6x6)
# regions=("tropical" "temperate" "arid")
# regions=("flat" "mountainous" "hilly")
# subregions=(2x2)
# regions=("ethiopia" "india" "amazon" "usa_south" "tropical" "temperate" "arid" "flat" "mountainous" "hilly")
# regions=("africa" "asia" "europe" "north_america" "south_america" "oceania")
# regions=("north_america" "south_america" "oceania")
regions=("india" "usa_south" "ethiopia" "corn_belt")
all_lead_times=(24 120 216)
nn_architectures=("mlp")
model_names=("pangu")
loss_functions=("extreme_heat_loss")
# Define bootstrap regions
bootstrap_regions=("temperate" "tropical" "arid" "flat" "hilly" "mountainous")

for region in "${regions[@]}"; do
    for subregion in "${subregions[@]}"; do
        # Skip if subregion is 2x2 and region is india, ethiopia, amazon, or usa_south
        if [[ "$subregion" == "2x2" && ("$region" == "india" || "$region" == "ethiopia" || "$region" == "amazon" || "$region" == "usa_south" || "$region" == "corn_belt") ]]; then
            continue
        fi

        # Skip if subregion is 6x6 and region is any of the bootstrap regions
        if [[ "$subregion" == "6x6" && " ${bootstrap_regions[@]} " =~ " ${region} " ]]; then
            continue
        fi

        for nn_architecture in "${nn_architectures[@]}"; do
            for var_pair in "${training_output_vars[@]}"; do
                # Split the pair into training_vars and output_vars
                IFS='|' read -r training_vars output_vars <<< "$var_pair"

                for model_name in "${model_names[@]}"; do
                    for loss_function in "${loss_functions[@]}"; do
                        # Skip incompatible combinations
                        if [[ "$loss_function" == "extreme_heat_loss" && "$output_vars" != "2m_temperature" ]]; then
                            continue
                        fi
                        # Determine train/test dates based on model_name
                        if [[ "$model_name" == "aifs" ]]; then
                            train_start="2022-01-01"
                            train_end="2023-12-31"
                            test_start="2024-01-01"
                            test_end="2024-12-31"

                            # only current aifs variables are total_precipitation and 2m_temperature
                            if [[ "$output_vars" != "total_precipitation" && "$output_vars" != "2m_temperature" ]]; then
                                continue
                            fi
                        else
                            # Testing period
                            # train_start="2018-01-01"
                            # train_end="2018-01-31"
                            # test_start="2022-01-01"
                            # test_end="2022-01-31"

                            # PRODUCTION: Full period (uncomment for production runs)
                            train_start="2018-01-01"
                            train_end="2021-12-31"
                            test_start="2022-01-01"
                            test_end="2022-12-31"

                            # aifs is the only model with precipitation currently
                            if [[ "$output_vars" == "total_precipitation" ]]; then
                                continue
                            fi
                        fi

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
                        
                        # Add growing_season_only flag if model_name is aifs
                        if [[ "$model_name" == "aifs" ]]; then
                            cmd="$cmd --growing_season_only"
                        fi

                        if [[ "$loss_function" == "extreme_heat_loss" ]]; then
                            # only both doing this for 2m_temperature and for geographic regions
                            if [[ "$output_vars" == "2m_temperature" && ("$region" == "ethiopia" || "$region" == "india" || "$region" == "amazon" || "$region" == "usa_south" || "$region" == "corn_belt") ]]; then
                                cmd="$cmd --alternate_loss_fn=\"$loss_function\""
                            fi
                        fi
                        
                        # Add bootstrap flag if region is in bootstrap_regions
                        if [[ " ${bootstrap_regions[@]} " =~ " ${region} " ]]; then
                            cmd="$cmd --bootstrap"
                        fi
                        
                        # Execute command
                        eval $cmd
                    done
                done
            done
        done
    done
done
