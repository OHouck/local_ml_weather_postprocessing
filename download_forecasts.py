# Author: Ozzy Houck
# Date Created 5/26/2024

# Purpose download ERA5 and ai generated forecast data for a specific date
# To do: Add in IFS forecast download and move configuration to a config file

from ecmwfapi import ECMWFDataServer
import cdsapi # to download ERA5 data
import os


output_path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/forecasts"
num_days = 10 # 240 hours
step_length = 6
date = "2024-04-01"
time_start = "00:00"
bbox = [90, -180, -90, 180]
variables = ['2m_temperature', 'total_precipitation']

# set up inputs for ai-models download
ai_models_date = date.replace("-", "")
ai_models_time_start = time_start.replace(":", "")

# make string of steps separated by '/' for IFS download
step_list = [i for i in range(0, num_days * 24 + 1, step_length)]
step_list = "/".join(map(str, step_list))


def download_ifs_forecast(date, step_list, bbox):
    # Create a new ECMWF API client
    # help generating pulls from: https://apps.ecmwf.int/datasets/data/tigge/levtype=sfc/type=cf/?date_year_month=202404&origin-time=ecmf;00:00:00&step=0,6,12,18&param=167
    server = ECMWFDataServer()

    # I think I want this one actually
    server.retrieve({
        "class": "ti",
        "dataset": "tigge",
        "date": date,
        "expver": "prod",
        "grid": "0.25/0.25",
        "levtype": "sfc",
        "origin": "ecmf",
        "param": "167/228228", # these are t2m and total precipitation
        "step": step_list,
        "time": "00:00:00",
        "area": bbox,
        "type": "fc",
        "target": output_path + "/ifs_forecast_" + date + ".grib",
    })

def download_era5(date, variables, bbox):

    # set up inputs for ERA5 download
    year = date.split("-")[0]
    month = date.split("-")[1]
    day_start = date.split("-")[2]
    day_end = str(int(day_start) + num_days -1) # ignore case where day_end > 31
    # list of days with padding 0s for < 10
    day_range = [str(i).zfill(2) for i in range(int(day_start), int(day_end) + 1)]

    c = cdsapi.Client()

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'grib',
            'variable': variables,
            'year': year,
            'month': month,
            'day': day_range,
            'time': [
                '00:00', '06:00', '12:00',
                '18:00',
            ], # want time steps at 6 hour intervals to match ai-models
            'area': bbox,
        },
        output_path + '/era5_' + date + '.grib')

# Download the IFS forecast and ERA5 data
download_ifs_forecast(date, step_list, bbox)
exit()
# download_era5(date, variables, bbox)

# Download the FourCastNet forecast (need to have set asset path before running) mine is set in my zshrc
fourcastnet_command = f"ai-models --input cds --date {ai_models_date} --time {ai_models_time_start} --path {output_path}/fourcastnet_{date}.grib --download-assets fourcastnet"
# download pangu forecast
pangu_command = f"ai-models --input cds --date {ai_models_date} --time {ai_models_time_start} --path {output_path}/pangu_{date}.grib --download-assets panguweather"
# download forecastnetv2
fourcastnetv2_command = f"ai-models --input cds --date {ai_models_date} --time {ai_models_time_start} --path {output_path}/fourcastnetv2_{date}.grib --download-assets fourcastnetv2-small"

# os.system(fourcastnet_command)
# os.system(pangu_command)
os.system(fourcastnetv2_command)
