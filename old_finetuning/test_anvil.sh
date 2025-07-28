#!/bin/sh -l

module load anaconda/2024.02-py311
conda activate /home/x-ohouck/aiw_env

cd /anvil/projects/x-atm170020/ohouck/ai_weather_ag

python3 finetuning/finetune.py \
    --forecast_path="gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr" \
    --obs_path="gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr" \
    --output_dir="~/wb_finetune_test" \
    --region="full_india" \
    --train_start="2021-01-01" --train_end="2021-12-30" \
    --test_start="2022-01-01" --test_end="2022-12-30" \
    --use_cupy

#simple script to run  in order to check for package loading and other small things
