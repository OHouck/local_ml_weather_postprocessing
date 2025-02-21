import xarray as xr
import io
import gzip


file = "/Users/ohouck/Desktop/metar_data/2005/03/01/point/metar/netcdf/20050301_1800.gz"

with gzip.open(file, "rb") as f:
    data = f.read()

ds = xr.open_dataset(io.BytesIO(data))



# print("Data variables:", ds.data_vars)



vars_to_keep = ["staticIds","stationName", "locationName", "latitude", 
                     "longitude", "elevation", "temperature", "windDir", 
                     "windSpeed", "precip6Hour"]
ds_subset = ds[vars_to_keep]
# filter to latitude and longitude of california
ds_subset = ds_subset.where((ds_subset.latitude > 32) & (ds_subset.latitude < 42) &
                            (ds_subset.longitude > -124) & (ds_subset.longitude < -114), 
                            drop=True)


# print(ds_subset)

df = ds_subset.to_dataframe()

print(df.shape)