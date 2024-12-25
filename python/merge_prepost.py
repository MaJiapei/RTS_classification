from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import zarr
import dask
from dask.diagnostics import ProgressBar

p = Path(r"G:\RTS滑塌提取中间过程\NDVI")
compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)


def check():
    names = ["0000000000-0000000000",
             "0000000000-0000037888",
             "0000000000-0000075776",
             "0000000000-0000113664",
             "0000037888-0000000000",
             "0000037888-0000037888",
             "0000037888-0000075776",
             "0000037888-0000113664", ]

    i = 1
    for year in range(1986, 2024):
        files = list(p.glob(f"*{year}*.tif"))
        print(f"{len(files)} files in {year}")
        for name in names:
            file = p / f"NVDI_mean{year:04d}-{name}.tif"
            if not file.exists():
                print(f"{i} : {file}")
                i += 1


def _float_to_int(value, min_val=-50, max_val=50, dtype=np.int16):
    # 将范围映射到int16的范围
    int_value_dict = {np.int16: [-32767, 32767], np.uint8: [1, 255]}
    int_min = int_value_dict.get(dtype)[0]
    int_max = int_value_dict.get(dtype)[1]

    # 放大因子
    scale_factor = (int_max - int_min) / (max_val - min_val)

    # 浮点数转换为整数并四舍五入
    int_val = xr.where(np.isnan(value), int_min-1, value * scale_factor)

    return int_val


def merge_spatial():
    x = slice(73.14075518, 104.57644046, )
    y = slice(39.74554332, 25.82838556, )
    mask = xr.open_zarr(r"E:\work\热融滑塌提取-青藏高原\热融滑塌提取\data\mask_tp").chunk({"x": 1330, "y": 1330})['mask']
    chunk_size = 1330
    for year in [2011, 2012, 2013]:
        files = list(p.glob(f"*{year}*.tif"))
        ds = xr.open_mfdataset(files, combine='by_coords').chunk({"x": chunk_size, "y": chunk_size})

        red = ds["band_data"].sel(band=1)
        nir = ds["band_data"].sel(band=2)
        swir2 = ds["band_data"].sel(band=3)
        ds = ds.drop_vars("band_data")

        red = xr.where((red == 0) | (mask == 0), np.nan, red / 254 - 1 / 254)
        nir = xr.where((nir == 0) | (mask == 0), np.nan, nir / 254 - 1 / 254)
        ndvi = (nir - red) / (nir + red)
        ndvi = xr.where(ndvi < 0.005, np.nan, ndvi)
        ndvi = xr.where(ndvi.isnull(), 0, (ndvi + 1) * 127 + 1)

        swir2 = xr.where((swir2 == 0) | (mask == 0), 0, swir2)
        ds['NDVI'] = ndvi
        ds['SWIR2'] = swir2
        encodings = {"NDVI": {"dtype": "uint8", "compressor": compressor, "_FillValue": 0},
                     "SWIR2": {"dtype": "uint8", "compressor": compressor, "_FillValue": 0}}

        obj0 = ds.to_zarr(fr"G:\RTS滑塌提取中间过程\青藏高原\merge_spatial\Landsat\{year}", mode="w", compute=True, encoding=encodings)
        with ProgressBar():
            dask.compute(*[obj0], scheduler='threads')


def calculate_ratio():
    for year in range(1986, 2024):
        ds = xr.open_zarr(fr"G:\RTS滑塌提取中间过程\青藏高原\merge_spatial\Landsat\{year}")
        ds_reduce = ds.coarsen(x=133, boundary="pad").mean().coarsen(y=133, boundary="pad").mean()
        ds_reduce = ds_reduce.sel(x=ds.x, y=ds.y, method="nearest")
        ds_reduce['x'] = ds.x
        ds_reduce['y'] = ds.y
        ratio = xr.where(ds_reduce > 0.005, ds / ds_reduce, np.nan)
        ratio = xr.where((ratio > 50) | (ratio < -50), np.nan, ratio)
        ratio = _float_to_int(ratio)
        # .sel(x=slice(92.77125535, 93.28591613), y=slice(34.89112488, 34.60953793))
        ratio = ratio.chunk({'x': 1330, 'y': 1330})

        obj = ratio.to_zarr(fr"G:\RTS滑塌提取中间过程\青藏高原\merge_spatial\Landsat_ratio\{year}", mode='w', compute=False,
                          encoding={"NDVI": {"dtype": "int16", "compressor": compressor, "_FillValue": -32768},
                                    "SWIR2": {"dtype": "int16", "compressor": compressor, "_FillValue": -32768},
                                    })
        with ProgressBar():
            obj.compute(scheduler='threads')


def merge_time():
    """
    :return:
    """
    """
    """

    p = Path(r"G:\RTS滑塌提取中间过程\青藏高原\merge_spatial\Landsat_ratio")
    files = list(p.glob("*"))
    time_var = xr.Variable('time', [pd.to_datetime(i_file.stem[-4:]) for i_file in files])
    ds0 = xr.concat([xr.open_zarr(i, mask_and_scale=False) for i in files], dim=time_var)
    ds0 = ds0.chunk({"time": -1})
    obj1 = ds0.to_zarr(r"G:\RTS滑塌提取中间过程\青藏高原\merge_time\landsat_RNDVI_RSWIR2", compute=False, mode='w',)
    with ProgressBar():
        dask.compute(*[obj1], scheduler='threads')


def to_raster():
    xs = slice(89.3188598793051654, 97.8936385923396415)
    ys = slice(35.9982782120499962, 32.3592928276102327)
    ds = xr.open_zarr(r"G:\RTS滑塌提取中间过程\青藏高原\merge_spatial\Landsat\2023", mask_and_scale=False)
    ds = ds.sel(x=xs, y=ys)
    ndvi = xr.where(ds['NDVI'] == 0, np.nan, ds['NDVI'] / 127 - 128 / 127)
    ndvi.rio.write_crs("epsg:4326", inplace=True)
    ndvi.rio.to_raster(r"E:\work\热融滑塌提取-青藏高原\热融滑塌提取\data\ndvi_2023.tif", COMPRESS="LZW")


def pre_post():
    ds = xr.open_zarr(r"G:\RTS滑塌提取中间过程\青藏高原\merge_time\landsat_RNDVI_RSWIR2")
    # mask = xr.open_zarr(r"E:\work\热融滑塌提取-青藏高原\热融滑塌提取\data\mask_tp")['mask']
    # ds = ds.sel(x=slice(89.3188598793051654, 97.8936385923396415),
    #             y=slice(35.9982782120499962, 32.3592928276102327))

    # x0, y1, x1,  y0 = 92.58047119, 34.98848066, 92.83780158, 35.12927414
    # xs = slice(x0, x1)
    # ys = slice(y0, y1)

    da = ds['SWIR2']

    da = da.bfill(dim='time').ffill(dim='time')
    da = (da - da.mean(dim='time')) / da.std(dim='time')
    da = da.rolling(time=3, min_periods=1, center=True).mean()
    da = _float_to_int(da, min_val=-5, max_val=5)
    da = da.chunk({"time": -1, "x": 200, "y": 200})

    obj = da.to_zarr(r"G:\RTS滑塌提取中间过程\青藏高原\merge_time\landsat_Z_SWIR2", mode='w', compute=True,
                     encoding={"SWIR2": {"dtype": "int16", "compressor": compressor, "_FillValue": -32768}})
    # with ProgressBar():
    #     obj.compute(scheduler='threads')


def ndvi_fillna():
    ds = xr.open_zarr(r"G:\RTS滑塌提取中间过程\青藏高原\merge_time\landsat_NDVI_SWIR2", mask_and_scale=False)
    da = xr.where(ds['NDVI'] == 0, np.nan, ds['NDVI'])
    da = da.bfill(dim='time').ffill(dim='time')
    da = da.rolling(time=3, min_periods=1, center=True).mean()
    obj = da.to_zarr(r"G:\RTS滑塌提取中间过程\青藏高原\merge_time\landsat_NDVI_fillna", mode='w', compute=True,
                     encoding={"NDVI": {"dtype": "uint8", "compressor": compressor, "_FillValue": 0}})



if __name__ == "__main__":
    from dask.distributed import Client

    dask.config.set({"temporary-directory": r"G:\dask_cache",
                     "distributed.worker.memory.spill": 0.8,
                     "distributed.worker.memory.pause": 0.9})
    client = Client(n_workers=1,
                    threads_per_worker=8,
                    memory_limit='80GB')
    # # merge_spatial()
    # pre_post()
    # to_raster()
    # print(_float_to_int16(-50, -50, 50))
    # merge_time()
    # # calculate_ratio()
    ndvi_fillna()