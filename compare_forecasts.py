# Author: Ozzy Houck
# Date Created 5/27/2024

# Purpose: Compare t2m measurements between ERA5, FourCastNet, and PanguWeather

import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Tuple

class ForecastComparison:
    """
    A class to compare different weather forecast models.
    """

    def __init__(self, forecast_path: str, fig_path: str, date: str, region: str):
        """
        Initialize the ForecastComparison object.

        :param forecast_path: Path to the forecast data files
        :param fig_path: Path to save the output figures
        :param date: Date of the forecast
        :param region: Region of interest for the forecast
        """
        self.forecast_path = forecast_path
        self.fig_path = fig_path
        self.date = date
        self.region = region
        self.combined = None
        self.bbox = self._set_bounding_box()

    def _set_bounding_box(self) -> List[float]:
        """
        Set the bounding box for the region of interest.

        :return: List of [lon_min, lat_min, lon_max, lat_max]
        """
        # Define bounding boxes for supported regions
        region_bounds = {
            "Global": (-90, 90, -180, 180),
            "Midwest": (35, 50, -100, -80),
            "Pakistan": (24, 37, 60, 78)
        }
        if self.region not in region_bounds:
            raise ValueError("Region not supported")
        lat_min, lat_max, lon_min, lon_max = region_bounds[self.region]
        return [lon_min, lat_min, lon_max, lat_max]

    def load_and_preprocess_data(self):
        """
        Load the forecast data and preprocess it.
        """
        # Load the data
        self.combined = xr.open_dataset(f"{self.forecast_path}/combined_forecasts_{self.date}.nc")
        # Remove the first time step
        self.combined = self.combined.isel(time=slice(1, None))
        self._filter_by_bbox()
        self._calculate_errors()

    def _filter_by_bbox(self):
        """
        Filter the data to only include the region of interest.
        """
        lon_min, lat_min, lon_max, lat_max = self.bbox
        self.combined = self.combined.where(
            (self.combined.longitude >= lon_min) & (self.combined.longitude <= lon_max) &
            (self.combined.latitude >= lat_min) & (self.combined.latitude <= lat_max),
            drop=True
        )

    def _calculate_errors(self):
        """
        Calculate the error and squared error for each forecast model.
        """
        for model in ['pangu', 'ifs', 'fourcastnet']:
            self.combined[f'{model}_error'] = self.combined[f'{model}_t2m'] - self.combined['era5_t2m']
            self.combined[f'{model}_error_squared'] = self.combined[f'{model}_error'] ** 2

    def create_forecast_gifs(self):
        """
        Create GIF animations for each forecast and error.
        """
        forecasts = ['era5_t2m', 'ifs_t2m', 'pangu_t2m', 'fourcastnet_t2m', 
                     'ifs_error', 'pangu_error', 'fourcastnet_error']
        for forecast in forecasts:
            self._make_forecast_gif(forecast)

    def _make_forecast_gif(self, forecast: str):
        """
        Create a GIF animation for a specific forecast or error.

        :param forecast: Name of the forecast variable
        """
        fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        data = self.combined[forecast].isel(time=0)
        contour = ax.contourf(self.combined.longitude, self.combined.latitude, data, 
                              transform=ccrs.PlateCarree())
        plt.colorbar(contour, ax=ax, label='Temperature (C)')

        def animate(i):
            ax.clear()
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            data = self.combined[forecast].isel(time=i)
            ax.contourf(self.combined.longitude, self.combined.latitude, data, 
                        transform=ccrs.PlateCarree())
            time = pd.to_datetime(data.time.values).round('h')
            ax.set_title(f"{forecast} at time {time}")

        anim = FuncAnimation(fig, animate, frames=len(self.combined.time), repeat=True)
        anim.save(f"{self.fig_path}/{forecast}_{self.region}_{self.date}.gif", writer='pillow', fps=3)
        plt.close(fig)

    def calculate_losses(self) -> Dict[str, np.ndarray]:
        """
        Calculate RMSE and freeze loss for each forecast model.

        :return: Dictionary of loss arrays for each model
        """
        losses = {}
        for model in ['ifs', 'pangu', 'fourcastnet']:
            losses[f'{model}_rmse'] = np.sqrt(self.combined[f'{model}_error_squared'].mean(dim=['latitude', 'longitude']))
            losses[f'{model}_freeze_loss'] = self._unexpected_freeze_loss(
                self.combined['era5_t2m'].values, 
                self.combined[f'{model}_t2m'].values
            )
        return losses

    @staticmethod
    def _unexpected_freeze_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the unexpected freeze loss.

        :param y_true: True temperature values
        :param y_pred: Predicted temperature values
        :return: Array of freeze loss values
        """
        loss_list = []
        for t in range(y_true.shape[0]):
            y_true_t, y_pred_t = y_true[t].flatten(), y_pred[t].flatten()
            # Double the penalty for unexpected freezes
            loss = np.where((y_true_t < 0) & (y_pred_t > 0), 
                            ((y_true_t - y_pred_t) * 2) ** 2, 
                            (y_true_t - y_pred_t) ** 2)
            loss_list.append(np.sqrt(np.mean(loss)))
        return np.array(loss_list)

    def plot_rmse_over_time(self, losses: Dict[str, np.ndarray]):
        """
        Plot RMSE and freeze loss over time for each model.

        :param losses: Dictionary of loss arrays for each model
        """
        time_hours = (self.combined.time.values - self.combined.time.values[0]).astype('timedelta64[h]')
        plt.figure(figsize=(12, 6))
        colors = {'ifs': 'lightgreen', 'pangu': 'darkgreen', 'fourcastnet': 'blue'}
        for model, color in colors.items():
            plt.plot(time_hours, losses[f'{model}_rmse'], label=f'{model.upper()}', color=color)
            plt.plot(time_hours, losses[f'{model}_freeze_loss'], label=f'{model.upper()} Freeze Loss', 
                     color=color, linestyle='dashed')
        plt.xlabel('Time (Hours)')
        plt.ylabel('RMSE (C)')
        plt.title('RMSE By Time')
        plt.legend()
        plt.savefig(f"{self.fig_path}/rmse_by_time_{self.region}.png")
        plt.close()

    def plot_rmse_maps(self):
        """
        Create and save RMSE maps for each forecast model, focusing on the specified region.
        """
        rmse_time = {model: np.sqrt(self.combined[f'{model}_error_squared'].mean(dim='time'))
                        for model in ['ifs', 'pangu', 'fourcastnet']}
        vmin, vmax = min(map(np.min, rmse_time.values())), max(map(np.max, rmse_time.values()))

        for model, data in rmse_time.items():
            fig = self._plot_rmse_map(data, f'{model.upper()} RMSE Map', vmin=vmin, vmax=vmax)
            fig.savefig(f"{self.fig_path}/{model}_rmse_map_{self.region}.png")
            plt.close(fig)

    def _plot_rmse_map(self, data: xr.DataArray, title: str, cmap: str = 'viridis', 
                    vmin: float = None, vmax: float = None) -> plt.Figure:
        """
        Create an RMSE map for a single model, focusing on the specified region.

        :param data: RMSE data for the model
        :param title: Title of the plot
        :param cmap: Colormap to use
        :param vmin: Minimum value for the colorbar
        :param vmax: Maximum value for the colorbar
        :return: The created figure
        """
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        
        # Plot the data
        im = ax.pcolormesh(data.longitude, data.latitude, data, transform=ccrs.PlateCarree(),
                        cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, label='RMSE (°C)')
        
        # Set the extent to the bounding box of the region
        ax.set_extent(self.bbox, crs=ccrs.PlateCarree())
        
        # Set title
        ax.set_title(title)
        
        return fig


def main():
    """
    Main function to run the forecast comparison.
    """
    forecast_path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/forecasts"
    fig_path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/figures"
    date = "2024-04-01"
    region = "Midwest"

    # Create ForecastComparison object and run analysis
    comparison = ForecastComparison(forecast_path, fig_path, date, region)
    comparison.load_and_preprocess_data()
    # comparison.create_forecast_gifs()
    losses = comparison.calculate_losses()
    comparison.plot_rmse_over_time(losses)
    comparison.plot_rmse_maps()

if __name__ == "__main__":
    main()