#!/usr/bin/env bash
# lat must be between -90 and 90
# lon must be between 0 and 360 (0 is at prime meridian)
#

source .venv/bin/activate

# simulataneous to run all lead times in a single model
# separate to run each lead time in a separate model
# TRAIN_MODE=simultaneous
TRAIN_MODE=separate

echo "Training mode: $TRAIN_MODE"

    # --data_dir="/Users/ohouck/test_wb_finetune_data" \
    # --data_dir="/Volumes/wd_external_hd/weatherbench" \
    # --data_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/processed/cleaned_weatherbench_downloads" \

# python3 finetuning/finetune.py \
#     --data_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/raw/" \
#     --output_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/fine_tuning_output" \
#     --training_vars 2m_temperature \
#     --output_vars 2m_temperature \
#     --train_start="2018-01-01" --train_end="2018-12-31" \
#     --test_start="2022-01-01" --test_end="2022-12-31" \
#     --model_name="pangu" \
#     --region="india" \
#     --subregion="4x4" \
#     --lead_time_hours 120

regions=("ethiopia" "india" "amazon" "usa_south" "british_columbia")
# regions=("tropical" "temperate" "arid")
subregions=(2x2 6x6 10x10)
# all_lead_times=(24 48 72 96 120 144 168)
all_lead_times=(24 120 240)

for region in "${regions[@]}"; do
    for subregion in "${subregions[@]}"; do
        
        if [[ "$TRAIN_MODE" == "simultaneous" ]]; then
            # Train all lead times simultaneously (single model for all lead times)
            all_lead_times_str="${all_lead_times[*]}"  # Convert array to space-separated string
            echo "Running simultaneous fine-tuning for region: $region, subregion: $subregion, lead times: $all_lead_times_str hours"

            
            # 2m temperature - pangu
            echo "Running fine-tuning for 2m_temperature pangu (simultaneous)"
            python3 finetuning/finetune.py \
                --data_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/raw/" \
                --training_vars 2m_temperature \
                --output_vars 2m_temperature \
                --output_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/fine_tuning_output" \
                --train_start="2018-01-01" --train_end="2021-12-31" \
                --test_start="2022-01-01" --test_end="2022-12-31" \
                --model_name="pangu" \
                --region="$region" \
                --subregion="$subregion" \
                --lead_time_hours $all_lead_times_str \
                --model_type="MLP"

            # 2m temperature - ifs
            echo "Running fine-tuning for 2m_temperature ifs (simultaneous)"
            python3 finetuning/finetune.py \
                --data_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/raw/" \
                --training_vars 2m_temperature \
                --output_vars 2m_temperature \
                --output_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/fine_tuning_output" \
                --train_start="2018-01-01" --train_end="2021-12-31" \
                --test_start="2022-01-01" --test_end="2022-12-31" \
                --model_name="ifs" \
                --region="$region" \
                --subregion="$subregion" \
                --lead_time_hours $all_lead_times_str \
                --model_type="MLP"

            # 10m wind speed - pangu
            echo "Running fine-tuning for 10m_wind_speed pangu (simultaneous)"
            python3 finetuning/finetune.py \
                --data_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/raw/" \
                --training_vars 10m_wind_speed \
                --output_vars 10m_wind_speed \
                --output_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/fine_tuning_output" \
                --train_start="2018-01-01" --train_end="2021-12-31" \
                --test_start="2022-01-01" --test_end="2022-12-31" \
                --model_name="pangu" \
                --region="$region" \
                --subregion="$subregion" \
                --lead_time_hours $all_lead_times_str \
                --model_type="MLP"

            # 10m wind speed - ifs
            echo "Running fine-tuning for 10m_wind_speed ifs (simultaneous)"
            python3 finetuning/finetune.py \
                --data_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/raw/" \
                --training_vars 10m_wind_speed \
                --output_vars 10m_wind_speed \
                --output_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/fine_tuning_output" \
                --train_start="2018-01-01" --train_end="2021-12-31" \
                --test_start="2022-01-01" --test_end="2022-12-31" \
                --model_name="ifs" \
                --region="$region" \
                --subregion="$subregion" \
                --lead_time_hours $all_lead_times_str
        else
            # Train each lead time separately (separate model for each lead time)
            echo "Running separate fine-tuning for region: $region, subregion: $subregion"
            for lead_time in "${all_lead_times[@]}"; do
                echo "Training lead time: $lead_time hours"
                
                # 2m temperature - pangu
                echo "Running fine-tuning for 2m_temperature pangu (lead time: $lead_time)"
                python3 finetuning/finetune.py \
                    --data_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/raw/" \
                    --training_vars 2m_temperature \
                    --output_vars 2m_temperature \
                    --output_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/fine_tuning_output" \
                    --train_start="2018-01-01" --train_end="2021-12-31" \
                    --test_start="2022-01-01" --test_end="2022-12-31" \
                    --model_name="pangu" \
                    --region="$region" \
                    --subregion="$subregion" \
                    --lead_time_hours $lead_time \
                    --model_type="MLP"

                # 2m temperature - ifs
                echo "Running fine-tuning for 2m_temperature ifs (lead time: $lead_time)"
                python3 finetuning/finetune.py \
                    --data_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/raw/" \
                    --training_vars 2m_temperature \
                    --output_vars 2m_temperature \
                    --output_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/fine_tuning_output" \
                    --train_start="2018-01-01" --train_end="2021-12-31" \
                    --test_start="2022-01-01" --test_end="2022-12-31" \
                    --model_name="ifs" \
                    --region="$region" \
                    --subregion="$subregion" \
                    --lead_time_hours $lead_time \
                    --model_type="MLP"

                # 10m wind speed - pangu
                echo "Running fine-tuning for 10m_wind_speed pangu (lead time: $lead_time)"
                python3 finetuning/finetune.py \
                    --data_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/raw/" \
                    --training_vars 10m_wind_speed \
                    --output_vars 10m_wind_speed \
                    --output_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/fine_tuning_output" \
                    --train_start="2018-01-01" --train_end="2021-12-31" \
                    --test_start="2022-01-01" --test_end="2022-12-31" \
                    --model_name="pangu" \
                    --region="$region" \
                    --subregion="$subregion" \
                    --lead_time_hours $lead_time \
                    --model_type="MLP"

                # 10m wind speed - ifs
                echo "Running fine-tuning for 10m_wind_speed ifs (lead time: $lead_time)"
                python3 finetuning/finetune.py \
                    --data_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/raw/" \
                    --training_vars 10m_wind_speed \
                    --output_vars 10m_wind_speed \
                    --output_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/fine_tuning_output" \
                    --train_start="2018-01-01" --train_end="2021-12-31" \
                    --test_start="2022-01-01" --test_end="2022-12-31" \
                    --model_name="ifs" \
                    --region="$region" \
                    --subregion="$subregion" \
                    --lead_time_hours $lead_time
            done
        fi
    done
done
