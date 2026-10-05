"""Choose the stations the continent-by-continent run will use, and record them.

This is the one global step. It looks at every active HadISD station, works
out each one's income group and continent, finds out which ones actually report
at 00 UTC often enough to be usable, and writes a manifest naming the sample.

The sample has two strata, recorded in the manifest's 'sample_stratum' column:

- 'balanced': an equal number of high-income and low-and-middle-income
  stations worldwide, 2,000 in total by default. This is the like-for-like
  sample the income-gap comparison is built on.
- 'regional_boost': every other usable station in the under-observed regions
  (Africa, and South America outside Argentina, which is already well
  sampled). The balanced draw leaves most of these out because Asia dominates
  the low-and-middle-income pool, yet they are where the paper's claims are
  strongest, so the run takes all of them. Filter them out to recover the
  balanced comparison.

Continents are not a quota. They are just how the run is later chopped into
pieces small enough for one machine, so the sample has to be chosen here, once,
before any continent is processed -- otherwise each chunk would fill a quota of
its own and the worldwide totals would not hold.

Nothing here reads the forecast archive. The only cost is downloading HadISD
for the candidate pool, which is resumable: a re-run skips what is already on
disk.

Example:
    uv run python station_finetuning/select_station_sample.py \
        --stations_per_income_group 1000
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper_funcs import setup_directories  # noqa: E402

from station_observations import (download_station_files,
                                  find_stations_globally,
                                  load_station_observations, training_coverage)
from station_sampling import (classify_stations, in_regions,
                              select_income_balanced_sample)

# The study period and the coverage bar are imported rather than restated, so
# a station that passes the screen here cannot fail it there for a definition
# that drifted.
from prepare_station_dataset import (FIRST_REQUIRED_DAY, LAST_REQUIRED_DAY,
                                     MINIMUM_TRAINING_COVERAGE, TRAINING_YEARS)

# How many candidates to screen per station wanted. Pass rates for the 70%
# 00 UTC bar, measured on ISD-Lite, were about 77% for high-income stations and
# 57% for low-and-middle-income ones, so 2.0 clears the target for both with room for
# the sampling noise. Raising it costs downloads, not archive reads.
DEFAULT_OVERSAMPLE = 2.0

# Continents whose every usable station is taken, beyond the balanced draw, and
# countries inside them left to the balanced draw alone (ISO alpha-3; Argentina
# already contributes many stations).
BOOST_CONTINENTS = ["africa", "south_america"]
BOOST_EXCLUDED_COUNTRIES = ["ARG"]

# Chunk name for selected stations that match none of the six continents. They
# are a handful of offshore sites; they still count towards the quota, so they
# get a chunk rather than being dropped.
UNASSIGNED_CHUNK = "unassigned"


def main():
    """Select the station sample and write the manifest csv.

    Inputs: read from the command line; run with --help for the options.

    Returns:
        None. Writes '<processed>/station_finetuning/<output>.csv'.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations_per_income_group", type=int, default=1000,
                        help="stations per income group; the default 1000 gives "
                             "a 2000-station worldwide sample, half high income "
                             "and half low and middle income")
    parser.add_argument("--oversample", type=float, default=DEFAULT_OVERSAMPLE,
                        help="candidates screened per station wanted")
    parser.add_argument("--parallel_downloads", type=int, default=6,
                        help="concurrent HadISD downloads")
    parser.add_argument("--seed", type=int, default=58)
    parser.add_argument("--output", default="station_sample_manifest.csv")
    parsed_arguments = parser.parse_args()

    directories = setup_directories()
    work_directory = os.path.join(directories["processed"], "station_finetuning")
    hadisd_directory = os.path.join(work_directory, "hadisd")

    # ---- 1. every station that was active across the study period ----------
    candidates = find_stations_globally(
        work_directory, FIRST_REQUIRED_DAY, LAST_REQUIRED_DAY)
    candidates = classify_stations(candidates, directories)
    print(f"{len(candidates)} active stations; "
          f"{candidates['income_group'].notna().sum()} carry an income group",
          flush=True)

    # ---- 2. screen an oversampled pool -------------------------------------
    # Coverage is only knowable after downloading, so more candidates are
    # screened than are wanted and the surplus is discarded below.
    screened_per_group = int(round(
        parsed_arguments.stations_per_income_group * parsed_arguments.oversample))
    # Every boost-region candidate is screened too, since all of them that
    # pass are wanted.
    in_boost_region = in_regions(candidates, BOOST_CONTINENTS,
                                 BOOST_EXCLUDED_COUNTRIES)
    screening_pool = pd.concat([
        select_income_balanced_sample(candidates, screened_per_group,
                                      parsed_arguments.seed),
        candidates[in_boost_region]]).drop_duplicates(
            subset=["usaf", "wban"]).reset_index(drop=True)
    print(f"screening {len(screening_pool)} candidates ({screened_per_group} "
          f"per income group plus {int(in_boost_region.sum())} in "
          f"{BOOST_CONTINENTS})", flush=True)

    stations_present = download_station_files(
        screening_pool, hadisd_directory,
        parallel_downloads=parsed_arguments.parallel_downloads)
    print(f"{stations_present} of {len(screening_pool)} candidates have "
          f"HadISD data", flush=True)

    # ---- 3. measure 00 UTC coverage over the training years ----------------
    # training_coverage is shared with prepare_station_dataset.py, so a station
    # that passes here also passes there.
    training_times = pd.date_range(f"{TRAINING_YEARS[0]}-01-01",
                                   f"{TRAINING_YEARS[-1]}-12-31", freq="D")
    observations = load_station_observations(
        screening_pool, hadisd_directory, training_times)
    screening_pool["training_coverage"] = training_coverage(
        observations, training_times, TRAINING_YEARS)

    usable = screening_pool[
        screening_pool["training_coverage"] >= MINIMUM_TRAINING_COVERAGE]
    print(f"{len(usable)} of {len(screening_pool)} clear the "
          f"{MINIMUM_TRAINING_COVERAGE:.0%} 00 UTC bar", flush=True)

    # ---- 4. the final balanced sample --------------------------------------
    balanced = select_income_balanced_sample(
        usable, parsed_arguments.stations_per_income_group, parsed_arguments.seed)
    boost = usable[in_regions(usable, BOOST_CONTINENTS,
                              BOOST_EXCLUDED_COUNTRIES)]
    # Balanced rows come first, so a station in both keeps that stratum.
    selected = pd.concat([
        balanced.assign(sample_stratum="balanced"),
        boost.assign(sample_stratum="regional_boost")]).drop_duplicates(
            subset=["usaf", "wban"]).reset_index(drop=True)
    boost_count = int((selected["sample_stratum"] == "regional_boost").sum())

    # A continent is only a routing key, so a station that matches none of the
    # six still belongs in the sample. Give it a chunk of its own rather than
    # dropping a station the quota already counted.
    without_continent = int(selected["continent"].isna().sum())
    if without_continent:
        print(f"{without_continent} selected stations match no continent "
              f"(offshore and open-ocean sites); routing them to "
              f"'{UNASSIGNED_CHUNK}'", flush=True)
    selected["continent"] = selected["continent"].fillna(UNASSIGNED_CHUNK)

    manifest_path = os.path.join(work_directory, parsed_arguments.output)
    selected.to_csv(manifest_path, index=False)

    print(f"\nwrote {manifest_path}: {len(selected)} stations worldwide, "
          f"{len(balanced)} balanced "
          f"({parsed_arguments.stations_per_income_group} per income group) "
          f"plus {boost_count} regional boost", flush=True)
    # The continent rows are how the run will be chunked, not a quota: nothing
    # was balanced across them.
    print(pd.crosstab(selected["continent"],
                      [selected["sample_stratum"], selected["income_group"]],
                      margins=True).to_string(), flush=True)
    per_continent_gb = (selected.groupby("continent").size() * 1.6e-3)
    print(f"\nforecast neighbourhoods, GB per continent:", flush=True)
    print(per_continent_gb.round(2).to_string(), flush=True)
    print(f"total {per_continent_gb.sum():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
