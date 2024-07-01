#!/bin/sh -l
# FILENAME: test_job.sbatch

#SBATCH --account=atm170020-gpu # Allocation name
#SBATCH -p gpu # GPU partition
#SBATCH --time=03:00:00
#SBATCH --mem=128G #24G total
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1 # Number of GPUs per node

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

#SBATCH --job-name weatherbench2 
#SBATCH -e job.e%j
#SBATCH -o job.o%j
#SBATCH --mail-user=ohouck@uchicago.edu
#SBATCH --mail-type=all # send email to above address at start and end of job

# load module and python enviroment

module load anaconda/2024.02-py311
conda activate /home/x-ohouck/aiw_env

# move to code directory
cd /anvil/projects/x-atm170020/ohouck/ai_weather_ag

# Set environment variables for Apache Beam
export BEAM_TEMP=/anvil/scratch/x-ohouck/beam-temp
OUTPUT_DIR="/anvil/projects/x-atm170020/ohouck/output/weatherbench2"
TIME_START="2020-01-01"
TIME_STOP="2020-07-01"
# see possible regions in evaluation.py
REGION="pakistan"

# Function to set model-specific parameters
set_model_params() {
    local model=$1
    case $model in
        "pangu_test")
            FORECAST_PATH="gs://weatherbench2/datasets/pangu/2018-2022_0012_64x32_equiangular_conservative.zarr"
            OBS_PATH="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
            CLIMATOLOGY_PATH="gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr"
            VARIABLES="temperature"
            ;;
        "pangu")
            FORECAST_PATH="gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr"
            OBS_PATH="gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
            CLIMATOLOGY_PATH="gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr"
            VARIABLES="2m_temperature,temperature"
            ;;
        "ifs_hres")
            FORECAST_PATH="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
            OBS_PATH="gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
            CLIMATOLOGY_PATH="gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr"
            VARIABLES="2m_temperature,temperature,total_precipitation_24hr"
            ;;
        "graphcast")
            FORECAST_PATH="gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr"
            OBS_PATH="gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
            CLIMATOLOGY_PATH="gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr"
            VARIABLES="2m_temperature,temperature,total_precipitation_24hr"
            ;;
        "neural_gcm")
            FORECAST_PATH="gs://weatherbench2/datasets/neuralgcm_deterministic/2020-240x121_equiangular_with_poles_conservative.zarr"
            OBS_PATH="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
            CLIMATOLOGY_PATH="gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr"
            VARIABLES="temperature,P_minus_E_cumulative"
            ;;
        *)
            echo "Unknown model: $model"
            exit 1
            ;;
    esac
}

# Function to run evaluation (setting num workers to number of cpus per task)
run_evaluation() {
    local model_name=$1
    set_model_params "$model_name"

    python evaluate.py \
        --forecast_path="$FORECAST_PATH" \
        --obs_path="$OBS_PATH" \
        --climatology_path="$CLIMATOLOGY_PATH" \
        --output_dir="$OUTPUT_DIR" \
        --output_file_prefix="${model_name}_${REGION}_" \
        --input_chunks="init_time=1,lead_time=1" \
        --eval_configs=deterministic \
        --time_start="$TIME_START" \
        --time_stop="$TIME_STOP" \
        --variables="$VARIABLES" \
        --regions="$REGION" \
        --use_beam=True \
        --runner=DirectRunner \
        -- \
        --direct_num_workers=32
}

# Run evaluations for different forecast models

# Pangu test evaluation (coarse resolution)
run_evaluation "pangu_test"

# # Pangu evaluation
# run_evaluation "pangu"

# # IFS HRES evaluation
# run_evaluation "ifs_hres"

# # GraphCast evaluation
# run_evaluation "graphcast"

# # Neural GCM evaluation
# run_evaluation "neural_gcm"
