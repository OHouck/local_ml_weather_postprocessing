#!/bin/bash
# Laptop driver for the sampled, continent-by-continent station run.
#
# The point of the continent split is disk, not speed. Forecast neighbourhoods
# cost about 1.6 MB per station, so a 2,000-station sample is roughly 3 GB if
# built in one piece. Building one continent at a time, training it, and then
# dropping its neighbourhood array keeps the high-water mark at the size of the
# largest single continent instead.
#
#   bash station_finetuning/run_station_sample.sh select
# train models for each station one continent at a time:
#   bash station_finetuning/run_station_sample.sh all
# combine the results into a single CSV:
#   bash station_finetuning/run_station_sample.sh combine
#
# 'all' runs every continent in turn and is resumable: a continent whose
# results csv already exists is skipped, so an interrupted run picks up where
# it stopped.

set -euo pipefail

cd "$(dirname "$0")/.."

processed=$(uv run python -c "
import sys; sys.path.insert(0, '.')
from helper_funcs import setup_directories
print(setup_directories()['processed'])")

work="${processed}/station_finetuning"
manifest="${work}/station_sample_manifest.csv"
results_dir="${work}/sample_results"

# Stations per income group, worldwide. The quota is global, which is why
# 'select' has to run once before any continent.
stations_per_income_group="${STATIONS_PER_INCOME_GROUP:-1000}"

# Set KEEP_ARRAYS=1 to keep each continent's neighbourhood array after
# training, e.g. to re-run training with a different specification.
keep_arrays="${KEEP_ARRAYS:-0}"

run_one_continent() {
    local continent="$1"
    local prefix="station_sample_${continent}"
    local results="${results_dir}/results_${continent}.csv"

    if [[ -f "${results}" ]]; then
        echo "== ${continent}: already done, skipping"
        return
    fi

    echo "== ${continent}: preparing"
    uv run python station_finetuning/prepare_station_dataset.py \
        --manifest "${manifest}" \
        --continent "${continent}" \
        --model_name pangu \
        --parallel_downloads 6 \
        --output_prefix "${prefix}"

    echo "== ${continent}: training"
    # --drop_neighbourhoods frees the chunk's forecast array once its results
    # are written. The pipeline does it rather than this script, so the shell
    # holds no knowledge of the dataset's file layout.
    # Written as an if rather than `[[ ]] && x=`, whose non-zero exit would
    # trip `set -e` if it ever became the last statement here.
    local drop_flag="--drop_neighbourhoods"
    if [[ "${keep_arrays}" == "1" ]]; then
        drop_flag=""
    fi

    uv run python station_finetuning/train_station_models.py \
        --dataset "${work}/${prefix}.npz" \
        --station_metadata "${work}/${prefix}_stations.csv" \
        --specifications station_finetuning/configs/tuned_specifications.json \
        --output "${results}" \
        ${drop_flag}
}

stage="${1:-all}"
case "${stage}" in

select)
    uv run python station_finetuning/select_station_sample.py \
        --stations_per_income_group "${stations_per_income_group}"
    ;;

all)
    mkdir -p "${results_dir}"
    # Read the chunk names out of the manifest rather than hardcoding them, so
    # a chunk the selection produced cannot be silently skipped here.
    chunks=$(uv run python -c "
import sys; sys.path.insert(0, 'station_finetuning')
from station_observations import read_station_metadata
print(' '.join(sorted(read_station_metadata('${manifest}')['continent'].dropna().unique())))")
    echo "chunks: ${chunks}"
    for continent in ${chunks}; do
        run_one_continent "${continent}"
    done
    ;;

continent)
    mkdir -p "${results_dir}"
    run_one_continent "${2:?usage: $0 continent <name>}"
    ;;

combine)
    uv run python -c "
import sys; sys.path.insert(0, 'station_finetuning')
from train_station_models import combine_station_results
combine_station_results('${results_dir}/results_*.csv', '${work}/station_results_sample.csv')
"
    ;;

*)
    echo "usage: $0 {select|all|continent <name>|combine}"
    exit 1
    ;;
esac
