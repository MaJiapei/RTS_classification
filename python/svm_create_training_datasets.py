import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gp
from tqdm import tqdm

from svm_read_utils import read_ratio, read_slope, read_index


def label_complete(df):
    if not all(df['ID']):
        df['LON'] = df.geometry.x
        df['LAT'] = df.geometry.y
        df['ID'] = df.apply(lambda x: f"S{x.name}" if not x['ID'] else x['ID'], axis=1)
        df.to_file(r"E:\work\热融滑塌提取-青藏高原\热融滑塌提取\data\Label_add_update_label_complete.shp")
    return df


def create_data(out_name=None, **kwargs):
    df = gp.read_file(r"E:\work\热融滑塌提取-青藏高原\热融滑塌提取\data\Label_add_update.shp")
    df = label_complete(df)

    num = len(df)
    X = np.empty((num, 38 * 3 + 1))
    y = np.empty((num))
    lines = 0

    for k, v in tqdm(df.iloc[0:].iterrows(), total=num):
        lon, lat = v['LON'], v['LAT']
        df_r = read_ratio(lat, lon, z_std=True)
        df_index = read_index(lat, lon)
        slope = read_slope(lat, lon)

        if not np.isnan(df_r['Z_RNDVI'].values).any():
            X[lines, 0:38] = df_index['NDVI'].values
            X[lines, 38:76] = df_r['Z_RNDVI'].values
            X[lines, 76:114] = df_r['Z_RSWIR2'].values
            X[lines, 114] = slope
            y[lines] = v['STATUS']
            lines += 1
        else:
            print(f"All nan in {k}")

    X = X[:lines]
    y = y[:lines]

    if out_name:
        np.savez(out_name, X=X, y=y)
    else:
        return X, y


if __name__ == "__main__":
    create_data(out_name=r"TS_model\svm_training_z_fill_0525.npz", z=True)
    # plot_training_data(29)

