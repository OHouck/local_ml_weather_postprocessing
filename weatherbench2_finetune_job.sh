# lat must be between -90 and 90
# lon must be between 0 and 360 (0 is at prime meridian)
python3 weatherbench2_finetune.py \
    --forecast_path="gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr" \
    --obs_path="gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr" \
    --output_dir="~/wb_finetune_test" \
    --model_name="pangu" \
    --lat_min=24 --lat_max=34 --lon_min=60 --lon_max=70 \
    --train_start="2018-01-01" --train_end="2021-12-30" \
    --test_start="2022-01-01" --test_end="2022-12-30" \
    --lead_time_hours=24 \
    --var_name="10m_v_component_of_wind" \
    --epochs=1000 --batch_size=32 --learning_rate=1e-5 

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