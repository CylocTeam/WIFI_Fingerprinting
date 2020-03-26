import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import similarity as sm
import radiomap as rm

if __name__ == "__main__":
    training_data = pd.read_csv("sample_data/TrainingData.csv")
    validation_data = pd.read_csv("sample_data/ValidationData.csv")
    wap_column_names = training_data.filter(regex=("WAP\d*")).columns

    training_data[training_data[wap_column_names] == 100] = np.nan  # 100 indicates a nan
    spatial_mean = np.mean(training_data[wap_column_names], axis=1)
    training_data[wap_column_names] = training_data[wap_column_names].sub(spatial_mean, axis=0)  # spatial mean norm

    validation_data[validation_data[wap_column_names] == 100] = np.nan  # 100 indicates a nan
    v_spatial_mean = np.mean(validation_data[wap_column_names], axis=1)
    validation_data[wap_column_names] = validation_data[wap_column_names].sub(v_spatial_mean, axis=0)  # spatial mean norm

    # Assuming we know the correct floor
    # TODO: add floor estimation
    cur_scan_id = 10
    cur_scan_gt = validation_data.iloc[cur_scan_id]
    cur_scan_vals = cur_scan_gt[wap_column_names]
    relev_training_data = training_data[training_data["FLOOR"] == cur_scan_gt["FLOOR"]]
    weights = sm.similarity_calculation(cur_scan_vals, relev_training_data[wap_column_names])

    weighted_mean_lon = np.average(relev_training_data.LONGITUDE[~np.isnan(weights)],
                                   weights=weights[~np.isnan(weights)])
    weighted_mean_lat = np.average(relev_training_data.LATITUDE[~np.isnan(weights)],
                                   weights=weights[~np.isnan(weights)])
    error = np.sqrt((cur_scan_gt.LONGITUDE-weighted_mean_lon)**2 +
                    (cur_scan_gt.LATITUDE-weighted_mean_lat)**2)
    title_str = "SCAN_ID = " + str(cur_scan_id) + "\nError = " + str(np.round(error, 2)) + " [m]"

    plt.figure()
    plt.scatter(relev_training_data.LONGITUDE, relev_training_data.LATITUDE, c=weights)
    plt.colorbar()
    plt.scatter(cur_scan_gt.LONGITUDE, cur_scan_gt.LATITUDE, c='r', marker='x')
    plt.scatter(weighted_mean_lon, weighted_mean_lat, c='m', marker='*')
    plt.title(title_str)
    plt.xlabel("Local X"); plt.ylabel("Local Y")
    plt.grid(alpha=0.3)
    plt.show()