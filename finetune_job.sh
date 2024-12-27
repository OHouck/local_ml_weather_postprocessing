python3 weatherbench2_finetuning.py \
    --forecast_path="gs://weatherbench2/datasets/pangu/2018-2022_0012_64x32_equiangular_conservative.zarr" \
    --obs_path="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr" \
    --output_dir="~/wb_finetune_test" \
    --model_name="pangu_test" \
    --lat_min=24 --lat_max=37 --lon_min=60 --lon_max=78 \
    --train_start="2018-05-01" --train_end="2018-05-31" \
    --valid_start="2020-05-01" --valid_end="2020-05-31" \
    --var_name=temperature --level=850 \
    --epochs=3 --batch_size=32
