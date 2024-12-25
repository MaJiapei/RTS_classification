
from osgeo import gdal
import numpy as np
import xarray as xr
import rioxarray as rxr
import zarr
import geopandas as gp
from raster_utils import vector2raster


def main():
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)

    data = xr.open_zarr(r"G:\RTS滑塌提取中间过程\青藏高原\merge_time\landsat_RNDVI_Z")

    ds = xr.open_dataset(r"H:\datasets\青藏高原新绘制冻土分布图（2017）\QTP_Per_1984.tif")
    da = ds["band_data"].sel(band=1)
    da = da.sel(x=data.x, y=data.y, method="nearest")
    da['x'] = data.x
    da['y'] = data.y

    river_buffer_name = r"E:\work\热融滑塌提取-青藏高原\热融滑塌提取\data\mask\river_buffer60.shp"
    river_buffer_name1 = r"E:\work\热融滑塌提取-青藏高原\热融滑塌提取\data\mask\river_buffer60_geo.shp"
    water_body_name = r"E:\work\热融滑塌提取-青藏高原\热融滑塌提取\data\mask\water_body.shp"
    river_buffer_raster_name = r"E:\work\热融滑塌提取-青藏高原\热融滑塌提取\data\mask\water_body.tif"
    lake_name = r"H:\datasets\矢量数据\Lakes_in_Tibet_Plateau\2021\TP_Lake_2021.shp"
    # (x_min, pixel_size, 0, y_max, 0, -pixel_size)
    # x_size, _, x_min, _, y_size, y_max = data.rio.transform()[0:6]
    # geo_transform = (x_min, x_size, 0, y_max, 0, y_size)
    # rows, cols = da.shape
    #
    # river = gp.read_file(river_buffer_name1)
    # lake = gp.read_file(lake_name)
    # water_body = gp.pd.concat([river, lake])
    # water_body.to_file(water_body_name)
    #
    # vector2raster(water_body_name, river_buffer_raster_name, rows, cols, geo_transform=geo_transform)

    river_ds = xr.open_dataset(river_buffer_raster_name).chunk({"x": 1300, "y": 1300})
    river_ds = river_ds["band_data"].sel(band=1)
    river_ds = river_ds.sel(x=data.x, y=data.y, method="nearest")
    river_ds['x'] = data.x
    river_ds['y'] = data.y
    mask = xr.where((river_ds == 0) & (da == 1), 1, 0)

    mask.astype(np.bool_)
    mask.name = "mask"
    encodings = {"mask": {"dtype": "bool", "compressor": compressor}}
    mask = mask.chunk({"x": 1300, "y": 1300})
    mask.to_zarr(r"E:\work\热融滑塌提取-青藏高原\热融滑塌提取\data\mask_tp", mode="w", encoding=encodings)
    mask.rio.to_raster(r"E:\work\热融滑塌提取-青藏高原\热融滑塌提取\data\mask_tp.tif", dtype='uint8', COMPRESS="LZW")


if __name__ == "__main__":
    main()