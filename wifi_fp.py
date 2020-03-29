import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import similarity as sm
import radiomap as rm


def initial_data_processing(df):
    """

    :param df:
    :return:
    """
    wap_column_names = df.filter(regex=("WAP\d*")).columns
    df[df[wap_column_names] == 100] = -100  # 100 indicates an AP that wasn't detected
    spatial_mean = np.mean(df[wap_column_names], axis=1)
    df[wap_column_names] = df[wap_column_names].sub(spatial_mean, axis=0)  # spatial mean normalization
    return df


if __name__ == "__main__":
    training_data = pd.read_csv("sample_data/TrainingData.csv")
    validation_data = pd.read_csv("sample_data/ValidationData.csv")
    wap_column_names = training_data.filter(regex=("WAP\d*")).columns

    training_data = initial_data_processing(training_data)
    validation_data = initial_data_processing(validation_data)

    rm_per_area, aps_per_area = rm.create_radio_map(training_data, [1, 1])

    # Assuming we know the correct floor
    # TODO: add floor estimation
    cur_scan_id = 100
    cur_scan_gt = validation_data.iloc[cur_scan_id]
    cur_scan_vals, cur_bid, cur_floor = cur_scan_gt[wap_column_names], cur_scan_gt["BUILDINGID"], cur_scan_gt["FLOOR"]

    relev_training_data = rm_per_area[cur_bid, cur_floor]
    weights = sm.similarity_calculation(cur_scan_vals, relev_training_data[wap_column_names])

    # TODO: fix location estimation via RM
    weighted_mean_lon = np.average(relev_training_data.x[~np.isnan(weights)],
                                   weights=weights[~np.isnan(weights)])
    weighted_mean_lat = np.average(relev_training_data.y[~np.isnan(weights)],
                                   weights=weights[~np.isnan(weights)])
    error = np.sqrt((cur_scan_gt.LONGITUDE-weighted_mean_lon)**2 +
                    (cur_scan_gt.LATITUDE-weighted_mean_lat)**2)
    title_str = "SCAN_ID = " + str(cur_scan_id) + "\nError = " + str(np.round(error, 2)) + " [m]"

    plt.figure()
    # plt.scatter(relev_training_data.x, relev_training_data.y, c=weights)
    plt.scatter(relev_training_data.x, relev_training_data.y, c=weights)
    plt.colorbar()
    plt.scatter(cur_scan_gt.LONGITUDE, cur_scan_gt.LATITUDE, c='r', marker='x')
    plt.scatter(weighted_mean_lon, weighted_mean_lat, c='m', marker='*')
    plt.title(title_str)
    plt.xlabel("Local X"); plt.ylabel("Local Y")
    plt.grid(alpha=0.3)
    plt.show()