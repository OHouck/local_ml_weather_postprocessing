import xarray


# open datasets and compare.

new_pangu_fc_path  = "/Users/ohouck/test_wb_finetune_data/train_usa_south/2018-01/pangu_train_forecast_data_2018-01.nc"
# new_pangu_obs_path = "/Users/ohouck/test_wb_finetune_data/train_usa_south/2018-01/pangu_train_obs_data_2018-01.nc"


old_pangu_fc_path = "/Users/ohouck/wb_finetune_data/train_usa_south/2018-01/pangu_train_forecast_data_2018-01.nc"
old_pangu_obs_path = "/Users/ohouck/wb_finetune_data/train_usa_south/2018-01/pangu_train_obs_data_2018-01.nc"

old_ifs_fc_path = "/Users/ohouck/wb_finetune_data/train_usa_south/2018-01/ifs_train_forecast_data_2018-01.nc"
old_ifs_obs_path = "/Users/ohouck/wb_finetune_data/train_usa_south/2018-01/ifs_train_obs_data_2018-01.nc"


india_path = "/Users/ohouck/wb_finetune_data/train_india/2018-01/pangu_train_forecast_data_2018-01.nc"

india = xarray.open_dataset(india_path)

print(india)

exit()


new_pangu_fc = xarray.open_dataset(new_pangu_fc_path)
# new_pangu_obs = xarray.open_dataset(new_pangu_obs_path)

old_pangu_fc = xarray.open_dataset(old_pangu_fc_path)
old_pangu_obs = xarray.open_dataset(old_pangu_obs_path)

old_ifs_fc = xarray.open_dataset(old_ifs_fc_path)
old_ifs_obs = xarray.open_dataset(old_ifs_obs_path)

def get_stats(ds):
    max_2m_temp= ds["2m_temperature"].max().values
    min_2m_temp= ds["2m_temperature"].min().values
    mean_2m_temp= ds["2m_temperature"].mean().values

    return max_2m_temp, min_2m_temp, mean_2m_temp

# function that returns first value of 2m_temperature
def get_stats(ds):
    return ds["2m_temperature"].values[0][0][0]


# print summary of datasets
print("New Pangu FC Dataset:")
print(get_stats(new_pangu_fc))

print("New Pangu Obs Dataset:")
# print(get_stats(new_pangu_obs))

print("Old Pangu FC Dataset:")
print(get_stats(old_pangu_fc))
print("Old Pangu Obs Dataset:")
print(get_stats(old_pangu_obs))

print("Old IFS FC Dataset:")
print(get_stats(old_ifs_fc))
print("Old IFS Obs Dataset:")
print(get_stats(old_ifs_obs))