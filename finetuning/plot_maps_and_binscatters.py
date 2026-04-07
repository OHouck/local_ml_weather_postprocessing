"""
Stand alone version of map_global_improvements from figures_finetuning.py
to create global maps of forecast improvements.

imports directory set up from helper_funcs.py

Data inputs:

Post-processed zarr files from finetuning output directory structure. 
These are all created by finetune.py with different runs managed by run_experiments.sh

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

    # Model configuration: MLP with block-k3 + snapshot ensemble
    model_kwargs = dict(
        nn_architecture="mlp",
        block_ensemble=True,
        block_holdout=3,
        snapshot_ensemble=1,
        subregion="6x6",
    )

    #=============================================
    # Global Improvement Map Plots
    #=============================================
    for model in ["pangu", "ifs"]:
        for variable in ["2m_temperature", "10m_wind_speed"]:
            print(f"Creating global improvement map for {model} - {variable}")
            map_global_improvements(dirs=dirs, model=model,
                                    variable=variable, map_type="improvement",
                                    pixel_level=True, **model_kwargs)
    #=============================================
    # Binscatter Plots
    #=============================================
    # overlaying lead times for single model: Currently plot used for main paper
    for model in ["pangu", "ifs"]:
        for x_metric in ["sdor", "equator_distance"]:
            _ = lead_time_compare_binscatter(
                dirs=dirs,
                model=model,
                x_metric=x_metric,
                **model_kwargs
            )

    # Create plot comparing model binscatters: Currently in Appendix
    for x_metric in ["sdor", "equator_distance"]:
        for variable in ["2m_temperature", "10m_wind_speed"]:
            _ = model_compare_binscatter(
                dirs=dirs,
                variable=variable,
                x_metric=x_metric,
                **model_kwargs
            )