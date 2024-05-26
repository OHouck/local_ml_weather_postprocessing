# Author: Ozzy Houck
# Date Created 5/26/2024

# Purpose download IFS and ai generated forecast data for a specific date

from ecmwfapi import ECMWFDataServer
import cdsapi # to download ERA5 data
import os

output_path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/"
num_days = 5 
date = "2024-01-01"
time_start = "00:00"
bbox = [90, -180, -90, 180]
variables = ['2m_temperature', 'total_precipitation']

# set up inputs for ai-models download
ai_models_date = date.replace("-", "")
ai_models_time_start = time_start.replace(":", "")

# set up inputs for ERA5 download
year = date.split("-")[0]
month = date.split("-")[1]
day_start = date.split("-")[2]
day_end = str(int(day_start) + num_days -1) # ignore case where day_end > 31
# list of days with padding 0s for < 10
day_range = [str(i).zfill(2) for i in range(int(day_start), int(day_end) + 1)]


# # Download IFS forecast
# server = ECMWFDataServer()


# server.retrieve({
#     "class": "od",
#     "dataset": "tigge",
#     "date": date,
#     "expver": "prod",
#     "levtype": "sfc",
#     "param": "167.128",
#     "step": "0/1/2/3/6/12/18/24/48/72/96/120/144/168/192/216/240",
#     "stream": "oper",
#     "time": "00:00:00",
#     "type": "cf",
#     "target": date + "idf_forecast" + date + ".grib",
# })

# Download the FourCastNet forecast (need to have set asset path before running) mine is set in my zshrc
fourcastnet_command = f"ai-models --input cds --date {ai_models_date} --time {ai_models_time_start} --path {output_path}/fourcastnet_{date}.grib --download-assets fourcastnet"
os.system(fourcastnet_command)

pangu_command = f"ai-models --input cds --date {ai_models_date} --time {ai_models_time_start} --path {output_path}/pangu_{date}.grib --download-assets panguweather"


exit()


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
        ],
        'area': bbox,
    },
    output_path + 'ERA5' + date + '.grib')