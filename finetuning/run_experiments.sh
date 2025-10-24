#!/usr/bin/env bash
# lat must be between -90 and 90
# lon must be between 0 and 360 (0 is at prime meridian)
#

source .venv/bin/activate

# simulataneous to run all lead times in a single model
# separate to run each lead time in a separate model
TRAIN_MODE=simultaneous
# TRAIN_MODE=simulataneous

echo "Training mode: $TRAIN_MODE"

    # # midway
    # --data_dir="/project/jfranke/ozma/data/raw/" \
    # --output_dir="/project/jfranke/ozma/data/finetuning_output/" \
    # # laptop
    # --data_dir="/Users/ohouck/globus/forecast_data/raw/" \
    # --output_dir="/Users/ohouck/globus/forecast_data/processed/finetuning_output/" \

# python3 finetuning/finetune.py \
#     --data_dir="/project/jfranke/ozma/data/raw/" \
#     --output_dir="/project/jfranke/ozma/data/fine_tuning_output/" \
#     --training_vars 2m_temperature \
#     --output_vars 2m_temperature \
#     --train_start="2022-01-01" --train_end="2023-12-31" \
#     --test_start="2024-01-01" --test_end="2024-12-31" \
#     --model_name="aifs" \
#     --region="india" \
#     --subregion="2x2" \
#     --lead_time_hours 216 \
#     --nn_architecture="mlp" \
#     --growing_season_only \
#     --alternate_loss_fn="extreme_heat_loss"

python3 finetuning/finetune.py \
    --data_dir="/Users/ohouck/globus/forecast_data/raw/" \
    --output_dir="/Users/ohouck/globus/forecast_data/processed/finetuning_output/" \
    --training_vars 2m_temperature \
    --output_vars 2m_temperature \
    --train_start="2018-01-01" --train_end="2021-12-31" \
    --test_start="2022-01-01" --test_end="2022-12-31" \
    --model_name="pangu" \
    --region="flat" \
    --subregion="2x2" \
    --lead_time_hours 120 \
    --nn_architecture="unet" \
    --growing_season_only

exit 0 

# regions=("ethiopia" "india" "amazon" "usa_south" "british_columbia")
regions=("india" "usa_south")
subregions=(6x6)

# regions=("tropical" "temperate" "arid" "flat" "mountainous" "hilly")
# subregions=(2x2)


all_lead_times=(24 120 216)

for region in "${regions[@]}"; do
    for subregion in "${subregions[@]}"; do
        
        # Train all lead times simultaneously (single model for all lead times)
        all_lead_times_str="${all_lead_times[*]}"  # Convert array to space-separated string
        echo "Running simultaneous fine-tuning for region: $region, subregion: $subregion, lead times: $all_lead_times_str hours"

        # 2m temperature - pangu
        echo "Running fine-tuning for 2m_temperature pangu (simultaneous)"
        python3 finetuning/finetune.py \
            --data_dir="/Users/ohouck/globus/forecast_data/raw" \
            --training_vars 2m_temperature \
            --output_vars 2m_temperature \
            --output_dir="/Users/ohouck/globus/forecast_data/processed/finetuning_output" \
            --train_start="2018-01-01" --train_end="2021-12-31" \
            --test_start="2022-01-01" --test_end="2022-12-31" \
            --model_name="pangu" \
            --region="$region" \
            --subregion="$subregion" \
            --lead_time_hours $all_lead_times_str \
            --model_type="mlp" \
            --alternate_loss_fn="extreme_heat_loss"

        # 2m temperature - ifs
        echo "Running fine-tuning for 2m_temperature ifs (simultaneous)"
        python3 finetuning/finetune.py \
            --data_dir="/Users/ohouck/globus/forecast_data/raw" \
            --training_vars 2m_temperature \
            --output_vars 2m_temperature \
            --output_dir="/Users/ohouck/globus/forecast_data/processed/finetuning_output" \
            --train_start="2018-01-01" --train_end="2021-12-31" \
            --test_start="2022-01-01" --test_end="2022-12-31" \
            --model_name="ifs" \
            --region="$region" \
            --subregion="$subregion" \
            --lead_time_hours $all_lead_times_str \
            --model_type="mlp" \
            --growing_season_only \
            --alternate_loss_fn="extreme_heat_loss"


        # # 10m wind speed - pangu
        # echo "Running fine-tuning for 10m_wind_speed pangu (simultaneous)"
        # python3 finetuning/finetune.py \
        #     --data_dir="/Users/ohouck/globus/forecast_data/raw" \
        #     --training_vars 10m_wind_speed \
        #     --output_vars 10m_wind_speed \
        #     --output_dir="/Users/ohouck/globus/forecast_data/processed/finetuning_output" \
        #     --train_start="2018-01-01" --train_end="2021-12-31" \
        #     --test_start="2022-01-01" --test_end="2022-12-31" \
        #     --model_name="pangu" \
        #     --region="$region" \
        #     --subregion="$subregion" \
        #     --lead_time_hours $all_lead_times_str \
        #     --model_type="unet" \

        # # 10m wind speed - ifs
        # echo "Running fine-tuning for 10m_wind_speed ifs (simultaneous)"
        # python3 finetuning/finetune.py \
        #     --data_dir="/Users/ohouck/globus/forecast_data/raw" \
        #     --training_vars 10m_wind_speed \
        #     --output_vars 10m_wind_speed \
        #     --output_dir="/Users/ohouck/globus/forecast_data/processed/finetuning_output" \
        #     --train_start="2018-01-01" --train_end="2021-12-31" \
        #     --test_start="2022-01-01" --test_end="2022-12-31" \
        #     --model_name="ifs" \
        #     --region="$region" \
        #     --subregion="$subregion" \
        #     --lead_time_hours $all_lead_times_str \
        #     --model_type="unet" \

        # 2m Temp - AIFS
        python3 finetuning/finetune.py \
            --data_dir="/Users/ohouck/globus/forecast_data/raw" \
            --training_vars 2m_temperature \
            --output_vars 2m_temperature \
            --output_dir="/Users/ohouck/globus/forecast_data/processed/finetuning_output" \
            --train_start="2021-01-01" --train_end="2023-12-31" \
            --test_start="2024-01-01" --test_end="2024-12-31" \
            --model_name="aifs" \
            --region="$region" \
            --subregion="$subregion" \
            --lead_time_hours $all_lead_times_str \
            --model_type="mlp" \
            --growing_season_only \
            --alternate_loss_fn="extreme_heat_loss"
        
        # # Total Daily Precipitation - AIFS
        # python3 finetuning/finetune.py \
        #     --data_dir="/Users/ohouck/globus/forecast_data/raw" \
        #     --training_vars total_precipitation \
        #     --output_vars total_precipitation \
        #     --output_dir="/Users/ohouck/globus/forecast_data/processed/finetuning_output" \
        #     --train_start="2021-01-01" --train_end="2023-12-31" \
        #     --test_start="2024-01-01" --test_end="2024-12-31" \
        #     --model_name="aifs" \
        #     --region="$region" \
        #     --subregion="$subregion" \
        #     --lead_time_hours $all_lead_times_str \
        #     --model_type="unet" \
        #     --growing_season_only
    done
done
