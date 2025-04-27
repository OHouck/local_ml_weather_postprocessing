#!/usr/bin/env bash
source .venv/bin/activate

regions=(amazon usa_south british_columbia india)
subregions=(2x2 4x4 6x6 8x8 10x10)
lead_times=(24 72 168)

for region in "${regions[@]}"; do
  for sub in "${subregions[@]}"; do
    for lt in "${lead_times[@]}"; do

      # 2‑m temperature – Pangu
      python finetune.py \
        --data_dir "~/wb_finetune_data" \
        --output_dir "~/ai_weather_outputs" \
        --model_name pangu \
        --training_vars 2m_temperature \
        --output_vars 2m_temperature \
        --region "$region" --subregion "$sub" --lead_time_hours "$lt"

      # 2‑m temperature – IFS
      python finetune.py \
        --data_dir "~/wb_finetune_data" \
        --output_dir "~/ai_weather_outputs" \
        --model_name ifs \
        --training_vars 2m_temperature \
        --output_vars 2m_temperature \
        --region "$region" --subregion "$sub" --lead_time_hours "$lt"

      # 10‑m wind – Pangu
      python finetune.py \
        --data_dir "~/wb_finetune_data" \
        --output_dir "~/ai_weather_outputs" \
        --model_name pangu \
        --training_vars 10m_v_component_of_wind 10m_u_component_of_wind \
        --output_vars 10m_v_component_of_wind 10m_u_component_of_wind \
        --region "$region" --subregion "$sub" --lead_time_hours "$lt"

      # 10‑m wind – IFS
      python finetune.py \
        --data_dir "~/wb_finetune_data" \
        --output_dir "~/ai_weather_outputs" \
        --model_name ifs \
        --training_vars 10m_v_component_of_wind 10m_u_component_of_wind \
        --output_vars 10m_v_component_of_wind 10m_u_component_of_wind \
        --region "$region" --subregion "$sub" --lead_time_hours "$lt"

      # 1000‑hPa temp – Pangu
      python finetune.py \
        --data_dir "~/wb_finetune_data" \
        --output_dir "~/ai_weather_outputs" \
        --model_name pangu \
        --training_vars temperature_1000hPa geopotential_1000hPa specific_humidity_1000hPa \
        --output_vars temperature_1000hPa \
        --region "$region" --subregion "$sub" --lead_time_hours "$lt"

      # 1000‑hPa temp – IFS
      python finetune.py \
        --data_dir "~/wb_finetune_data" \
        --output_dir "~/ai_weather_outputs" \
        --model_name ifs \
        --training_vars temperature_1000hPa geopotential_1000hPa specific_humidity_1000hPa \
        --output_vars temperature_1000hPa \
        --region "$region" --subregion "$sub" --lead_time_hours "$lt"

    done
  done
done