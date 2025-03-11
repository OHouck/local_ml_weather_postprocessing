#!/bin/sh -l
# FILENAME: finetune_job.sh

#SBATCH --account=atm170020-gpu # Allocation name
#SBATCH --time=0-02:00:00
#SBATCH -p gpu # GPU partition
#SBATCH --mem=64
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1 # Number of GPUs per node

#SBATCH --ntasks-per-node=1 # total number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --job-name finetune_test
#SBATCH -e slurm_test.e%j
#SBATCH -o slurm_test.o%j
#SBATCH --mail-user=ohouck@uchicago.edu
#SBATCH --mail-type=all # send email to above address at start and end of job

# lat must be between -90 and 90
# lon must be between 0 and 360 (0 is at prime meridian)
module load anaconda/2024.02-py311
conda activate /home/x-ohouck/aiw_env

cd /anvil/projects/x-atm170020/ohouck/ai_weather_ag

set -x
srun -u --mpi=pmi2 \
    bash -c "
    TORCH_USE_CUDA_DSA=1 python3 finetuning/finetune.py \
    --forecast_path="gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr" \
    --obs_path="gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr" \
    --output_dir="/anvil/projects/x-atm170020/ohouck/finetuning_results" \
    --training_vars 10m_v_component_of_wind 10m_u_component_of_wind \
    --output_vars 10m_v_component_of_wind 10m_u_component_of_wind 
    "

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


# if running on anvil
    # --output_dir="/anvil/projects/x-atm170020/ohouck/finetuning_results" \

# IF running locally
    # --output_dir="~/wb_finetune_test" \
