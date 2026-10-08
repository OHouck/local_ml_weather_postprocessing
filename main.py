"""
Run the whole analysis of "Tailoring machine learning weather predictions for
local impacts", section by section.

Each section is one script with a run() function, and can also be run on its
own (uv run python <script>). Comment out a line in main() to skip a step.
Every step reads its inputs from, and writes its outputs to, the data
directories in common.setup_directories(); figures go to erc_figures/.

    1. Download and prepare the gridded forecasts
           data_preparation/download_forecasts.py, data_preparation/create_land_patches.py
    2. Download ERA5 and the IFS HRES analysis
           data_preparation/download_reanalysis.py
    3. Download and prepare the station data
           data_preparation/select_stations.py, data_preparation/prepare_station_data.py
    4. Gridded hyperparameter search and model specification tests
           gridded_post_processing/hyperparameter_search.py,
           gridded_post_processing/architecture_comparison.py
    5. Gridded post-processing
           gridded_post_processing/run_post_processing.py
    6. Station specification search
           station_post_processing/specification_search.py
    7. Station post-processing
           station_post_processing/run_station_post_processing.py
    8. Gridded figures
           figures/gridded_figures.py
    9. Station figures
           figures/station_figures.py

Run with:
    uv run python main.py
"""

from data_preparation import (create_land_patches, download_forecasts,
                              download_reanalysis, prepare_station_data, select_stations)
from figures import gridded_figures, station_figures
# Section 4 is off by default in main(), so two of these imports go unused
from gridded_post_processing import (architecture_comparison,  # noqa: F401
                                     hyperparameter_search, run_post_processing)
from station_post_processing import (run_station_post_processing,
                                     specification_search)


def main():
    """
    Run every section of the analysis in order.

    Inputs:
        None.

    Returns:
        None.
    """
    # 1. Download and prepare the gridded forecasts (skips years already on disk)
    download_forecasts.run()
    create_land_patches.run()

    # 2. Download ERA5 and the IFS HRES analysis (skips years already on disk)
    download_reanalysis.run()

    # 3. Download and prepare the station data
    select_stations.run()
    prepare_station_data.run()

    # 4. Gridded hyperparameter search and architecture comparison. Both take
    #    days and their results are already saved (tuned_hyperparameters.json
    #    and the figure 5 outputs), so they are off by default.
    # hyperparameter_search.run()
    # architecture_comparison.run()

    # 5. Gridded post-processing of every land patch (resumes where it stopped)
    run_post_processing.run()

    # 6. Station network specification search on the 2021 development year
    specification_search.run()

    # 7. Station post-processing for Pangu and IFS (resumes where it stopped)
    run_station_post_processing.run()

    # 8. Gridded figures
    gridded_figures.run()

    # 9. Station figures
    station_figures.run()


if __name__ == "__main__":
    main()
