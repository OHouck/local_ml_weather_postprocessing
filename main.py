# Author: Ozzy Houck
# Date Created: 6/6/2024

# Purpose: run the main functions to download, combine, and compare forecasts

# 1. Download forecasts (this part should probably be run on the server)
exec(open("download_forecasts.py").read())

# 2. Combine forecasts
exec(open("combine_forecasts.py").read())

# 3. Compare forecasts
exec(open("compare_forecasts.py").read())