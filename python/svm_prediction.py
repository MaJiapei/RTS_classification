
import numpy as np
import joblib
from svm_read_utils import read_ratio, read_index

svm_model_c = joblib.load(r'TS_model/svm_model.pkl')
scaler_model_c = joblib.load(r'TS_model/scaler_model.pkl')


def predict(lat, lon, **kwargs):
    svm_model = svm_model_c
    scaler_model = scaler_model_c

    df = read_ratio(lat, lon, z_std=True, **kwargs)
    df_index = read_index(lat, lon)

    X = np.empty((38 * 3))
    X[0:38] = df['Z_RNDVI'].values
    X[38:76] = df['Z_RSWIR2'].values
    X[76:] = df_index['NDVI'].values

    if not np.isnan(X).any():
        x = np.reshape(X, (1, len(X)))
        s_x = scaler_model.transform(x)
        return svm_model.predict(s_x), svm_model.predict_proba(s_x)


if __name__ == '__main__':
    from utils_plot import union_plot

    y, x =35.63481702, 88.49309765
    union_plot(y, x, show_s2=True)
    print(predict(y, x))