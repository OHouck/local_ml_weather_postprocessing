#!/usr/bin/env bash
# lat must be between -90 and 90
# lon must be between 0 and 360 (0 is at prime meridian)

source .venv/bin/activate

model_names=("pangu" "ifs")
regions=("pakistan" "south_pakistan" "north_india" "full_india" "uttar_pradesh" "pixel")
lead_times=(24 72 168)

for model_name in "${model_names[@]}"; do
    for region in "${regions[@]}"; do
        for lead_time in "${lead_times[@]}"; do
            python3 finetuning/finetune.py \
                --forecast_path="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr" \
                --obs_path="gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr" \
                --data_dir="~/wb_finetune_data" \
                --output_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/wb_finetune_test" \
                --training_vars 10m_v_component_of_wind 10m_u_component_of_wind \
                --output_vars 10m_v_component_of_wind 10m_u_component_of_wind \
                --train_start="2018-01-01" --train_end="2021-12-31" \
                --test_start="2022-01-01" --test_end="2022-12-31" \
                --model_name="$model_name" \
                --region="$region" \
                --lead_time_hours="$lead_time"
        done
    done
done

for model_name in "${model_names[@]}"; do
    for region in "${regions[@]}"; do
        for lead_time in "${lead_times[@]}"; do
            python3 finetuning/finetune.py \
                --forecast_path="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr" \
                --obs_path="gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr" \
                --data_dir="~/wb_finetune_data" \
                --output_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/wb_finetune_test" \
                --training_vars 2m_temperature geopotential_1000hPa specific_humidity_1000hPa \
                --output_vars 2m_temperature \
                --train_start="2018-01-01" --train_end="2021-12-31" \
                --test_start="2022-01-01" --test_end="2022-12-31" \
                --model_name="$model_name" \
                --region="$region" \
                --lead_time_hours="$lead_time"
        done
    done
done

for model_name in "${model_names[@]}"; do
    for region in "${regions[@]}"; do
        for lead_time in "${lead_times[@]}"; do
            python3 finetuning/finetune.py \
                --forecast_path="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr" \
                --obs_path="gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr" \
                --data_dir="~/wb_finetune_data" \
                --output_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/wb_finetune_test" \
                --training_vars 2m_temperature \
                --output_vars 2m_temperature \
                --train_start="2018-01-01" --train_end="2021-12-31" \
                --test_start="2022-01-01" --test_end="2022-12-31" \
                --model_name="$model_name" \
                --region="$region" \
                --lead_time_hours="$lead_time"
        done
    done
done



# holding forecast paths for different forecasts
# graphcast 
    # --forecast_path="gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr" \ 
# pangu low-res
    # --forecast_path="gs://weatherbench2/datasets/pangu/2018-2022_0012_64x32_equiangular_conservative.zarr" \
# pangu high-res
    # --forecast_path="gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr" \
# ERA5 low-res
    # --obs_path="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr" \
# ERA5 high-res
    # --obs_path="gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr" \

# IFS ground truth
    # --obs_path="gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr" \

# IFS forecast
    # --forecast_path="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr" \