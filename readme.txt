GEE: The JS code in Google Earth Engine for generating NDVI timeseries, Including data preprocessing, annual scale aggregation, masking, and export.
python: Offline processing code, including conversion of NDVI data into Dask's zarr format, calculating the ratio of NDVI to its background value, model training, prediction, etc.
model: Trained SVM model and data standardization model.
training_datasets: Training data for the SVM model.
results: results data, including the predicted RTS probability by SVM model (SVM_predicted_RTS_probability.tif), the RTS start year identified by the breakpoint detection algorithm (RTS_break_points_start_year.tif), and the RTS features after post-processing (RTS_predicted.gpkg). The values in the predicted RTS probability file are scaled from 0.5-1 to 50-100. The true RTS start time can be acquired by multiplying the values in the file of RTS start year with 1986.
