import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stt

# TODO: change RM to consider difference floors
# TODO: receive RM (x,y) from outside
def create_radio_map(training_data):
    """
    Get training data from difference devices (location + all APS RSSI) and creates a RadioMap
    :param training_data: Dataframe containing reference locations + RSSI
    :return: Radio Map (numpy array)
    """
    # grid data
    lon_min = min(training_data.LONGITUDE)
    lon_max = max(training_data.LONGITUDE)
    lat_min = min(training_data.LATITUDE)
    lat_max = max(training_data.LATITUDE)

    grid_anchor = [lon_min, lat_min]
    grid_size = [int(lon_max-lon_min)+1, int(lat_max-lat_min)+1]

    wap_column_names = training_data.filter(regex=("WAP\d*")).columns
    num_of_wap = len(wap_column_names)
    agg_list = {i: mean_relev_rssi for i in wap_column_names}

    # create radiomap
    RSSI_RM = np.full((grid_size[0], grid_size[1], num_of_wap,), np.nan)

    # find mean RSSI value for each WAP for each grid point
    training_data["grid_pnt"] = tuple(map(lambda p: [int(p[0]-grid_anchor[0]), int(p[1]-grid_anchor[1])],
                                          zip(training_data.LONGITUDE, training_data.LATITUDE)))
    training_data_gridgroups = training_data.groupby(by="grid_pnt")
    training_data_agg = training_data_gridgroups.agg(agg_list)

    # Update RM via the training data
    grid_pnt_list = list(zip(*training_data_agg.index.to_list()))
    RSSI_RM[grid_pnt_list[0], grid_pnt_list[1], :] = training_data_agg[wap_column_names].to_numpy()
    return RSSI_RM

def mean_relev_rssi (rssi_list):
    """
    removes the non-relevant RSSI values (nans, i.e. RSSI=100) and returns the average
    functionized for out comfort.
    :param rssi_list: RSSI value list
    :return: average RSSI value, without nans (=100)
    """
    return np.mean(rssi_list[rssi_list < 100])

def similarity_calculation (cur_ap_rssi, df, p=1):
    """
    Calculate the similarity between 2 RSSI rank vectors
    :param ranks1: cur rssi vector that is compared
    :param ranks2: radiomap rssi vector
    :return:
    """

    # TODO: find literature about weighting where not all AP's were seen. This causes issues here
    df[np.isnan(df) & ~np.isnan(cur_ap_rssi)] = -200 # assumption about RSSI values that weren't found
    weights = ((abs(df.sub(cur_ap_rssi, axis=1))**p).sum(axis=1))**(1/p)
    weights[weights == 0] = np.nan
    return 1 / weights

if __name__== "__main__":
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
    weights = similarity_calculation(cur_scan_vals, relev_training_data[wap_column_names])

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