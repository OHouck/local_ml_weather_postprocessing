#!/bin/bash
# Two-stage driver for the global station-level experiment.
#
# Stage 1 (prepare) selects stations, downloads HadISD, and extracts the
# forecast neighbourhoods into a memory-mapped .npy. It is one serial job,
# because the extraction is bound by reading the global archive once, not by
# the number of stations.
#
# Stage 2 (train) is a SLURM array. Stations are independent, so the array
# splits the qualifying list into shards and each task writes its own csv.
#
#   sbatch station_finetuning/run_station_experiments.sh prepare
#   sbatch station_finetuning/run_station_experiments.sh train      # after it finishes
#   bash   station_finetuning/run_station_experiments.sh combine
#
#SBATCH --job-name=station_experiments
#SBATCH --account=pi-jfranke
#SBATCH --output=station_experiments_%A_%a.txt
#SBATCH --partition=caslake
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-19

set -euo pipefail

source .venv/bin/activate

hostname=$(hostname)
if [[ "$hostname" == "oMac.local" ]]; then
    processed="/Users/ohouck/globus/forecast_data/processed"
elif [[ "$hostname" == *"midway3"* ]]; then
    processed="/project/jfranke/ozma/forecast_data/processed"
else
    echo "Unknown environment. Hostname: $hostname"
    exit 1
fi

stage="${1:-train}"
prefix="station_dataset_global"
work="${processed}/station_finetuning"
dataset="${work}/${prefix}.npz"
metadata="${work}/${prefix}_stations.csv"
shard_dir="${work}/shards_${prefix}"

case "$stage" in

prepare)
    # --station_selection global is the change that widens the draw from the
    # few hundred stations inside the study boxes to every active HadISD station.
    # Add --latitude_band -23.5 23.5 to target the tropics instead.
    python3 prepare_station_dataset.py \
        --station_selection global \
        --model_name pangu \
        --parallel_downloads 6 \
        --output_prefix "${prefix}"
    ;;

train)
    mkdir -p "${shard_dir}"
    task="${SLURM_ARRAY_TASK_ID:-0}"

    # The array size is the only shard knob: train_station_models.py takes
    # every shard_count-th qualifying station, so changing --array above needs
    # no matching change here.
    TORCH_THREADS="${SLURM_CPUS_PER_TASK:-8}" python3 train_station_models.py \
        --dataset "${dataset}" \
        --station_metadata "${metadata}" \
        --specifications configs/tuned_specifications.json \
        --output "${shard_dir}/results_${task}.csv" \
        --shard_index "${task}" \
        --shard_count "${SLURM_ARRAY_TASK_COUNT:-1}"
    ;;

combine)
    # Concatenate the shard csvs, keeping one header.
    python3 -c "
import sys; sys.path.insert(0, 'station_finetuning')
from train_station_models import combine_station_results
combine_station_results('${shard_dir}/results_*.csv', '${work}/station_results_${prefix}.csv')
"
    ;;

*)
    echo "usage: $0 {prepare|train|combine}"
    exit 1
    ;;
esac
