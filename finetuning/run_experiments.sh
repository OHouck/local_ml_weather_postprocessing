#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=run_experiments
#SBATCH --account=pi-jfranke
#SBATCH --output=run_experiments-%J.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1 
# lat must be between -90 and 90
# lon must be between 0 and 360 (0 is at prime meridian)
#
source .venv/bin/activate
# simulataneous to run all lead times in a single model
# separate to run each lead time in a separate model
TRAIN_MODE=simultaneous
# TRAIN_MODE=simulataneous
echo "Training mode: $TRAIN_MODE"
# laptop
data_dir="/Users/ohouck/globus/forecast_data/raw/"
output_dir="/Users/ohouck/globus/forecast_data/processed/finetuning_output/"

# midway
# data_dir="/project/jfranke/ozma/forecast_data/raw/"
# output_dir="/project/jfranke/ozma/forecast_data/fine_tuning_output/"

regions=("corn_belt" "usa_south")
subregions=(6x6)
# regions=("tropical" "temperate" "arid")
# regions=("flat" "mountainous" "hilly")
# subregions=(2x2)
# regions=("ethiopia" "india" "amazon" "usa_south" "tropical" "temperate" "arid" "flat" "mountainous" "hilly")
all_lead_times=(24 120 216)
nn_architectures=("unet")
variables=("2m_temperature" "10m_wind_speed")
model_names=("pangu")
loss_functions=("mse")

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
            for variable in "${variables[@]}"; do
                for lead_time in "${all_lead_times[@]}"; do
                    for model_name in "${model_names[@]}"; do
                        for loss_function in "${loss_functions[@]}"; do
                            # Skip incompatible combinations
                            if [[ "$loss_function" == "extreme_heat_loss" && ${variable} != "2m_temperature" ]]; then
                                continue 
                            fi
                            # Determine train/test dates based on model_name
                            if [[ "$model_name" == "aifs" ]]; then
                                train_start="2022-01-01"
                                train_end="2023-12-31"
                                test_start="2024-01-01"
                                test_end="2024-12-31"
                                
                                # only current aifs variables are total_precipitation and 2m_temperature
                                if [[ "$variable" != "total_precipitation" && "$variable" != "2m_temperature" ]]; then
                                    continue
                                fi
                            else
                                train_start="2018-01-01"
                                train_end="2021-12-31"
                                test_start="2022-01-01"
                                test_end="2022-12-31"
                                # aifs is the only model with precipitation currently
                                if [[ "$variable" == "total_precipitation" ]]; then
                                    continue
                                fi
                            fi
                            
                            # Build base command
                            cmd="python3 finetuning/finetune.py \
                                --data_dir=\"$data_dir\" \
                                --output_dir=\"$output_dir\" \
                                --training_vars \"$variable\" \
                                --output_vars \"$variable\" \
                                --train_start=\"$train_start\" --train_end=\"$train_end\" \
                                --test_start=\"$test_start\" --test_end=\"$test_end\" \
                                --model_name=\"$model_name\" \
                                --region=\"$region\" \
                                --subregion=\"$subregion\" \
                                --lead_time_hours $lead_time \
                                --nn_architecture=\"$nn_architecture\""
                            
                            # Add growing_season_only flag if model_name is aifs
                            if [[ "$model_name" == "aifs" ]]; then
                                cmd="$cmd --growing_season_only"
                            fi

                            if [[ "$loss_function" == "extreme_heat_loss" ]]; then
                                # only both doing this for 2m_temperature and for geographic regions
                                if [[ "$variable" == "2m_temperature" && ("$region" == "ethiopia" || "$region" == "india" || "$region" == "amazon" || "$region" == "usa_south" || "$region" == "corn_belt") ]]; then
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
done