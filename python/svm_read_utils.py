
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt


ds = xr.open_zarr(r"G:\RTS滑塌提取中间过程\青藏高原\merge_time\landsat_RNDVI_RSWIR2")
ds_fill = xr.open_zarr(r"G:\RTS滑塌提取中间过程\青藏高原\merge_time\landsat_NDVI_fillna")
ds_index = xr.open_zarr(r"G:\RTS滑塌提取中间过程\青藏高原\merge_time\landsat_NDVI_SWIR2")
ds_zndvi = xr.open_zarr(r"G:\RTS滑塌提取中间过程\青藏高原\merge_time\landsat_Z_RNDVI")
ds_zswir2 = xr.open_zarr(r"G:\RTS滑塌提取中间过程\青藏高原\merge_time\landsat_Z_SWIR2")
slope = xr.open_dataset(r"H:\datasets\dem\CJY_slope_120m_gcs.tif")


def slope_120m():
    ds = xr.open_dataset(r"H:\datasets\dem\CJY_slope_30m.tif")
    ds_reduce = ds["band_data"].isel(band=0).coarsen(x=4, boundary="pad").mean().coarsen(y=4, boundary="pad").mean()
    ds_reduce.rio.to_raster(r"H:\datasets\dem\CJY_slope_120m.tif")


def read_ratio(y, x, smooth=3, interpolate_nan=True, z_std=False):

    sud_ds = ds.sel(x=x, y=y, method="nearest")

    df: pd.DataFrame = pd.DataFrame(columns=["RNDVI", "RSWIR2"], index=ds.time)
    df['RNDVI'] = sud_ds['NDVI'].values
    df['RSWIR2'] = sud_ds['SWIR2'].values

    # df.replace({0: np.nan}, inplace=True)
    print(z_std)
    if interpolate_nan:
        df = df.bfill().ffill()

    if z_std:
        df = (df - df.mean(axis=0)) / df.std(axis=0)
        df.columns = ['Z_RNDVI', 'Z_RSWIR2']
    else:
        df = df / 655.34

    if smooth:
        df = df.rolling(window=smooth, center=True, min_periods=1).mean()

    return df


def read_index(y, x, smooth=3, interpolate_nan=True, recovery=True):

    sub_ds = ds_index.sel(x=x, y=y, method="nearest")

    df: pd.DataFrame = pd.DataFrame(columns=["NDVI", "SWIR2"], index=ds.time)
    df['NDVI'] = sub_ds['NDVI'].values
    df['SWIR2'] = sub_ds['SWIR2'].values

    if interpolate_nan:
        df = df.bfill().ffill()

    if recovery:
        df['NDVI'] = df['NDVI'] / 127 - 128 / 127
        df['SWIR2'] = df['SWIR2'] / 254 - 1 / 254

    if smooth:
        df = df.rolling(window=smooth, center=True, min_periods=1).mean()

    return df


def read_BNDVI(y, x, smooth=3, interpolate_nan=True, z_std=False):
    ds_reduce = ds_index['NDVI'].coarsen(x=133, boundary="pad").mean().coarsen(y=133, boundary="pad").mean()
    df = ds_reduce.sel(x=x, y=y, method="nearest").to_dataframe() / 127 - 128 / 127
    if smooth:
        df = df['NDVI'].rolling(window=smooth, center=True, min_periods=1).mean()
    else:
        df = df['NDVI']
    return df


def read_ratio1(y, x):
    sub_ds_ndvi = ds_zndvi.sel(x=x, y=y, method="nearest")
    sub_ds_swir2 = ds_zswir2.sel(x=x, y=y, method="nearest")
    df: pd.DataFrame = pd.DataFrame(columns=["Z_RNDVI", "Z_RSWIR2"], index=ds.time)
    df['Z_RNDVI'] = sub_ds_ndvi['NDVI'].values / 6553.4
    df['Z_RSWIR2'] = sub_ds_swir2['SWIR2'].values / 6553.4
    return df


def read_index1(y, x):
    sud_ds_ndvi = ds_fill.sel(x=x, y=y, method="nearest")
    df: pd.DataFrame = pd.DataFrame(columns=["NDVI"], index=ds.time)
    df['NDVI'] = sud_ds_ndvi['NDVI'].values / 127 - 128 / 127
    return df

def read_slope(y, x):
    sub_ds = slope.sel(x=x, y=y, method="nearest")
    return sub_ds['band_data'].sel(band=1).values


if __name__ == "__main__":
    lat, lon = 34.71378195, 92.89349779
    # df_r = read_ratio(lat, lon, z_std=True)
    # df_index = read_index(lat, lon)
    # slope = read_slope(lat, lon)
    read_BNDVI(lat, lon)
    print(1)