"""
Stand alone version of map_global_improvements from figures_finetuning.py
to create global maps of forecast improvements.

imports directory set up from helper_funcs.py

Data inputs:

Post-processed zarr files from finetuning output directory structure. 
These are all created by post_process.py with different runs managed by run_experiments.sh

The helper function all_patch_data is used
as a data processing function but i have saved the output and commented it out 
so it shouldn't have to be used. 

"""

import os
import glob
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from binsreg import binsregselect, binsreg, binsqreg, binsglm, binstest, binspwc

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from finetuning.process_forecasts import calculate_rmse
from figures_finetuning import *
from helper_funcs import setup_directories, generate_output_path


if __name__ == "__main__":
    dirs = setup_directories()

    # Model configuration: MLP snapshot ensemble x3
    # Matches finetuning/run_experiments.sh:
    #   --nn_architecture mlp --snapshot_ensemble=3
    model_kwargs = dict(
        nn_architecture="mlp",
        snapshot_ensemble=3,
        block_ensemble=False,
        block_holdout=1,
        subregion="6x6",
    )

    # large boxplot comparing IFS and pangu. currently in appendix
    # model_compare_boxplot( 
    #     dirs=dirs,
    #     models=["pangu", "ifs"],
    #     variables=["2m_temperature", "10m_wind_speed"],
    #     regions=None,  # Use default continents
    #     save_dir=None,  # Auto-generate based on parameters
    #     train_start="2018-01-01",
    #     train_end="2021-12-31",
    #     test_start="2022-01-01",
    #     test_end="2022-12-31",
    #     **model_kwargs
    # )


    #=============================================
    # Global Improvement Map Plots
    #=============================================
    for model in ["pangu", "ifs"]:
        for variable in ["2m_temperature", "10m_wind_speed"]:
            print(f"Creating global improvement map for {model} - {variable}")
            map_global_improvements(dirs=dirs, model=model,
                                    variable=variable, map_type="original",
                                    pixel_level=False, **model_kwargs)
    #=============================================
    # Binscatter Plots
    #=============================================
    # overlaying lead times for single model: Currently plot used for main paper
    # for model in ["pangu", "ifs"]:
    #     for x_metric in ["sdor", "equator_distance"]:
    #         _ = lead_time_compare_binscatter(
    #             dirs=dirs,
    #             model=model,
    #             x_metric=x_metric,
    #             include_mean_bias_correction_baseline=True,
    #             **model_kwargs
    #         )

    # Create plot comparing model binscatters: Currently unused
    # for x_metric in ["sdor", "equator_distance"]:
    #     for variable in ["2m_temperature", "10m_wind_speed"]:
    #         _ = model_compare_binscatter(
    #             dirs=dirs,
    #             variable=variable,
    #             x_metric=x_metric,
    #             **model_kwargs
    #         )