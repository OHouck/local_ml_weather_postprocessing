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

# ============================================================================
# DATA DIRECTORIES
# ============================================================================
# Uncomment the appropriate configuration for your environment

# laptop
# data_dir="/Users/ohouck/globus/forecast_data/raw/"
# output_dir="/Users/ohouck/globus/forecast_data/processed/finetuning_output/"

# midway
data_dir="/project/jfranke/ozma/forecast_data/raw/"
output_dir="/project/jfranke/ozma/forecast_data/fine_tuning_output/"

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

    # Full: Use all 6 variables for training (best performance from experiments)
    "2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind temperature_1000hPa specific_humidity_1000hPa geopotential_1000hPa|2m_temperature"
)

regions=("india")
subregions=(6x6)
# regions=("tropical" "temperate" "arid")
# regions=("flat" "mountainous" "hilly")
# subregions=(2x2)
# regions=("ethiopia" "india" "amazon" "usa_south" "tropical" "temperate" "arid" "flat" "mountainous" "hilly")
all_lead_times=(24 120 216)
nn_architectures=("unet")
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
            for var_pair in "${training_output_vars[@]}"; do
                # Parse training and output variables from the pair
                # Format: "training_var1 training_var2|output_var1 output_var2"
                IFS='|' read -r training_vars output_vars <<< "$var_pair"

                # Extract first output variable for compatibility checks
                first_output_var=$(echo $output_vars | awk '{print $1}')

                for model_name in "${model_names[@]}"; do
                    for loss_function in "${loss_functions[@]}"; do
                        # Skip incompatible combinations
                        if [[ "$loss_function" == "extreme_heat_loss" && "$first_output_var" != "2m_temperature" ]]; then
                            continue
                        fi

                        # Determine train/test dates based on model_name
                        if [[ "$model_name" == "aifs" ]]; then
                            # AIFS dates
                            train_start="2022-01-01"
                            train_end="2023-12-31"
                            test_start="2024-01-01"
                            test_end="2024-12-31"

                            # only current aifs variables are total_precipitation and 2m_temperature
                            if [[ "$first_output_var" != "total_precipitation" && "$first_output_var" != "2m_temperature" ]]; then
                                continue
                            fi
                        else
                            # Pangu/IFS dates - SHORT TEST PERIOD
                            # For quick testing: 1 month train, 1 month test
                            train_start="2018-01-01"
                            train_end="2018-01-31"
                            test_start="2022-01-01"
                            test_end="2022-01-31"

                            # PRODUCTION: Full period (uncomment for production runs)
                            # train_start="2018-01-01"
                            # train_end="2021-12-31"
                            # test_start="2022-01-01"
                            # test_end="2022-12-31"

                            # aifs is the only model with precipitation currently
                            if [[ "$first_output_var" == "total_precipitation" ]]; then
                                continue
                            fi
                        fi

                        echo ""
                        echo "=========================================="
                        echo "Running experiment:"
                        echo "  Region: $region ($subregion)"
                        echo "  Model: $model_name"
                        echo "  Architecture: $nn_architecture"
                        echo "  Training vars: $training_vars"
                        echo "  Output vars: $output_vars"
                        echo "  Loss function: $loss_function"
                        echo "=========================================="

                        # Build command - NO QUOTES around variable expansions for training/output vars
                        # This allows multiple space-separated arguments to be passed correctly
                        python3 finetuning/finetune.py \
                            --data_dir="$data_dir" \
                            --output_dir="$output_dir" \
                            --training_vars $training_vars \
                            --output_vars $output_vars \
                            --train_start="$train_start" --train_end="$train_end" \
                            --test_start="$test_start" --test_end="$test_end" \
                            --model_name="$model_name" \
                            --region="$region" \
                            --subregion="$subregion" \
                            --lead_time_hours ${all_lead_times[@]} \
                            --nn_architecture="$nn_architecture" \
                            $(if [[ "$model_name" == "aifs" ]]; then echo "--growing_season_only"; fi) \
                            $(if [[ "$loss_function" == "extreme_heat_loss" && "$first_output_var" == "2m_temperature" && ("$region" == "ethiopia" || "$region" == "india" || "$region" == "amazon" || "$region" == "usa_south" || "$region" == "corn_belt") ]]; then echo "--alternate_loss_fn=$loss_function"; fi) \
                            $(if [[ " ${bootstrap_regions[@]} " =~ " ${region} " ]]; then echo "--bootstrap"; fi)

                        if [ $? -ne 0 ]; then
                            echo "ERROR: Experiment failed!"
                            exit 1
                        fi
                    done
                done
            done
        done
    done
done

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY"
echo "=========================================="
