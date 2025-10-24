import torch
import pickle
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
from datetime import timedelta
from aurora import Batch, Metadata, Aurora, rollout

sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories

def aurora_batch_to_xarray(preds: List[Batch]) -> xr.Dataset:
    """
    Convert Aurora predictions to a single xarray Dataset.
    
    Parameters
    ----------
    preds : List[Batch]
        List of Aurora Batch predictions from rollout. Each element represents
        one forecast lead time (6-hour increments starting at 12 hours).
        
    Returns
    -------
    xr.Dataset
        Combined dataset with both surface and atmospheric variables.
        
    Coordinates
    -----------
    init_time : datetime64
        Initialization time of the forecast
    lead_time : timedelta64
        Forecast lead time (12h, 18h, 24h, ...)
    latitude : float
        Latitude coordinates (degrees North)
    longitude : float
        Longitude coordinates (degrees East)
    level : int
        Atmospheric pressure levels (hPa)
        
    Variables
    ---------
    Surface variables (dimensions: init_time, lead_time, latitude, longitude):
        - 2t: 2-meter temperature (K)
        - 10u: 10-meter U wind component (m/s)
        - 10v: 10-meter V wind component (m/s)
        - msl: Mean sea level pressure (Pa)
        
    Atmospheric variables (dimensions: init_time, lead_time, level, latitude, longitude):
        - t: Temperature (K)
        - u: U wind component (m/s)
        - v: V wind component (m/s)
        - q: Specific humidity (kg/kg)
        - z: Geopotential (m²/s²)
    """
    
    if len(preds) == 0:
        raise ValueError("preds list is empty")
    
    # Extract basic information from first prediction
    first_pred = preds[0]
    n_steps = len(preds)
    
    # Get spatial coordinates
    lat = first_pred.metadata.lat.numpy()
    lon = first_pred.metadata.lon.numpy()
    
    # Get atmospheric levels
    levels = np.array(list(first_pred.metadata.atmos_levels))
    
    # Determine initialization time (6 hours before first prediction)
    # Aurora uses 2 time steps as input, the last one is at 06:00
    # The first prediction is 6 hours later (12:00)
    first_valid_time = first_pred.metadata.time[0]
    init_time = first_valid_time - timedelta(hours=6)
    
    # Create lead times array (12h, 18h, 24h, ...)
    lead_times = np.array([timedelta(hours=12 + 6*i) for i in range(n_steps)])
    
    # Convert to timedelta64 for xarray
    lead_times_td64 = pd.to_timedelta(lead_times).to_numpy()
    
    # Initialize data dictionaries
    surf_data = {}
    atmos_data = {}
    
    # Get variable names from first prediction
    surf_vars = list(first_pred.surf_vars.keys())
    atmos_vars = list(first_pred.atmos_vars.keys())
    
    # Extract surface variables
    # Original shape: [batch, time, lat, lon] -> we want [lead_time, lat, lon]
    for var_name in surf_vars:
        # Stack all lead times
        var_data = np.stack([
            pred.surf_vars[var_name][0, 0].numpy()  # Remove batch and time dims
            for pred in preds
        ], axis=0)  # Shape: [lead_time, lat, lon]
        
        # Add init_time dimension
        surf_data[var_name] = (
            ["init_time", "lead_time", "latitude", "longitude"],
            var_data[np.newaxis, ...]  # Add init_time dim: [1, lead_time, lat, lon]
        )
    
    # Extract atmospheric variables
    # Original shape: [batch, time, level, lat, lon] -> we want [lead_time, level, lat, lon]
    for var_name in atmos_vars:
        # Stack all lead times
        var_data = np.stack([
            pred.atmos_vars[var_name][0, 0].numpy()  # Remove batch and time dims
            for pred in preds
        ], axis=0)  # Shape: [lead_time, level, lat, lon]
        
        # Add init_time dimension
        atmos_data[var_name] = (
            ["init_time", "lead_time", "level", "latitude", "longitude"],
            var_data[np.newaxis, ...]  # Add init_time dim: [1, lead_time, level, lat, lon]
        )
    
    # Combine all variables
    all_data = {**surf_data, **atmos_data}
    
    # Create the dataset
    ds = xr.Dataset(
        all_data,
        coords={
            "init_time": [np.datetime64(init_time)],
            "lead_time": lead_times_td64,
            "latitude": lat,
            "longitude": lon,
            "level": levels,
        }
    )
    
    # Add metadata attributes
    ds.attrs.update({
        "title": "Aurora Weather Forecast",
        "source": "Aurora foundation model",
        "init_time": str(init_time),
        "n_lead_times": n_steps,
        "lead_time_increment": "6 hours",
        "first_lead_time": "12 hours",
        "spatial_resolution": "0.25 degrees",
    })
    
    # Add variable attributes
    var_attrs = {
        # Surface variables
        "2t": {
            "long_name": "2 metre temperature",
            "units": "K",
            "standard_name": "air_temperature",
        },
        "10u": {
            "long_name": "10 metre U wind component",
            "units": "m s-1",
            "standard_name": "eastward_wind",
        },
        "10v": {
            "long_name": "10 metre V wind component", 
            "units": "m s-1",
            "standard_name": "northward_wind",
        },
        "msl": {
            "long_name": "Mean sea level pressure",
            "units": "Pa",
            "standard_name": "air_pressure_at_mean_sea_level",
        },
        # Atmospheric variables
        "t": {
            "long_name": "Temperature",
            "units": "K",
            "standard_name": "air_temperature",
        },
        "u": {
            "long_name": "U component of wind",
            "units": "m s-1",
            "standard_name": "eastward_wind",
        },
        "v": {
            "long_name": "V component of wind",
            "units": "m s-1", 
            "standard_name": "northward_wind",
        },
        "q": {
            "long_name": "Specific humidity",
            "units": "kg kg-1",
            "standard_name": "specific_humidity",
        },
        "z": {
            "long_name": "Geopotential",
            "units": "m2 s-2",
            "standard_name": "geopotential",
        },
    }
    
    for var_name, attrs in var_attrs.items():
        if var_name in ds:
            ds[var_name].attrs.update(attrs)
    
    # Add coordinate attributes
    ds["init_time"].attrs.update({
        "long_name": "Forecast initialization time",
        "standard_name": "forecast_reference_time",
    })
    
    ds["lead_time"].attrs.update({
        "long_name": "Forecast lead time",
        "units": "hours",
    })
    
    ds["latitude"].attrs.update({
        "long_name": "Latitude",
        "units": "degrees_north",
        "standard_name": "latitude",
    })
    
    ds["longitude"].attrs.update({
        "long_name": "Longitude", 
        "units": "degrees_east",
        "standard_name": "longitude",
    })
    
    ds["level"].attrs.update({
        "long_name": "Pressure level",
        "units": "hPa",
        "standard_name": "air_pressure",
        "positive": "down",
    })
    
    return ds

def _prepare(x: np.ndarray) -> torch.Tensor:
    """Prepare a variable.

    This does the following things:
    * Select the first two time steps: 00:00 and 06:00.
    * Insert an empty batch dimension with `[None]`.
    * Flip along the latitude axis to ensure that the latitudes are decreasing.
    * Copy the data, because the data must be contiguous when converting to PyTorch.
    * Convert to PyTorch.
    """
    return torch.from_numpy(x[:2][None][..., ::-1, :].copy())



def run_aurora(day: str, num_lead_times: int, device: str, dirs: dict):
    """
    day: str
        Day to run the forecast for, in "YYYY-MM-DD" format.
    num_lead_times: int
        Number of 6-hour lead times to predict.
    device: str
        Device to run the model on, e.g. "cpu" or "cuda".
    dirs: dict 
        Directory structure dictionary from setup_directories().
    """

    download_path = Path("~/downloads/hres_0.25")
    download_path = download_path.expanduser()
    download_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = os.path.join(dirs['raw'], "aurora-0.25-finetuned.ckpt")
    static_var_path = os.path.join(dirs['raw'], "aurora-0.25-static.pickle")

    with open(static_var_path, "rb") as f:
        static_vars = pickle.load(f)

    surf_vars_ds = xr.open_dataset(download_path / f"{day}-surface-level.nc", engine="netcdf4")
    atmos_vars_ds = xr.open_dataset(download_path / f"{day}-atmospheric.nc", engine="netcdf4")

    checkpoint_path = os.path.join(dirs['raw'], "aurora-0.25-finetuned.ckpt")
    static_var_path = os.path.join(dirs['raw'], "aurora-0.25-static.pickle")


    batch = Batch(
        surf_vars={
            "2t": _prepare(surf_vars_ds["2m_temperature"].values),
            "10u": _prepare(surf_vars_ds["10m_u_component_of_wind"].values),
            "10v": _prepare(surf_vars_ds["10m_v_component_of_wind"].values),
            "msl": _prepare(surf_vars_ds["mean_sea_level_pressure"].values),
        },
        static_vars={k: torch.from_numpy(v) for k, v in static_vars.items()},
        atmos_vars={
            "t": _prepare(atmos_vars_ds["temperature"].values),
            "u": _prepare(atmos_vars_ds["u_component_of_wind"].values),
            "v": _prepare(atmos_vars_ds["v_component_of_wind"].values),
            "q": _prepare(atmos_vars_ds["specific_humidity"].values),
            "z": _prepare(atmos_vars_ds["geopotential"].values),
        },
        metadata=Metadata(
            # Flip the latitudes! We need to copy because converting to PyTorch, because the
            # data must be contiguous.
            lat=torch.from_numpy(surf_vars_ds.latitude.values[::-1].copy()),
            lon=torch.from_numpy(surf_vars_ds.longitude.values),
            # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
            # `datetime.datetime`s. Note that this needs to be a tuple of length one:
            # one value for every batch element. Select element 1, corresponding to time
            # 06:00.
            time=(surf_vars_ds.time.values.astype("datetime64[s]").tolist()[1],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds.level.values),
        ),
    )

    model = Aurora()

    # Load from local checkpoint file
    model.load_checkpoint_local(checkpoint_path)

    model.eval()
    model = model.to(device)

    with torch.inference_mode():
        preds = [pred.to(device) for pred in rollout(model, batch, steps=num_lead_times)]

    model = model.to(device)

    output_path = os.path.join(dirs['raw'], "aurora_raw", f"aurora_forecast_{day}.pt")
    torch.save(preds, output_path)

    print("Predictions done!")
    return(preds)



def main():

    dirs = setup_directories()
    day = "2022-05-11"
    num_lead_times = 2  # Number of 6-hour lead times to predict.

    aurora_preds = run_aurora(day, num_lead_times, device="cpu", dirs=dirs)
    preds_ds = aurora_batch_to_xarray(aurora_preds)

    print(preds_ds)


if __name__ == "__main__":
    main()

    # fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))

    # for i in range(ax.shape[0]):
    #     pred = preds[i]

    #     ax[i, 0].imshow(pred.surf_vars["2t"][0, 0].numpy() - 273.15, vmin=-50, vmax=50)
    #     ax[i, 0].set_ylabel(str(pred.metadata.time[0]))
    #     if i == 0:
    #         ax[i, 0].set_title("Aurora Prediction")
    #     ax[i, 0].set_xticks([])
    #     ax[i, 0].set_yticks([])

    #     ref = surf_vars_ds["2m_temperature"][2 + i].values[::-1, :]
    #     ax[i, 1].imshow(ref - 273.15, vmin=-50, vmax=50)
    #     if i == 0:
    #         ax[i, 1].set_title("HRES T0")
    #     ax[i, 1].set_xticks([])
    #     ax[i, 1].set_yticks([])

    # plt.tight_layout()
    # plt.show()