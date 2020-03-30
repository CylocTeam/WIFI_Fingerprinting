import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import similarity as sm
import radiomap as rm
import unsupervised as usp

def initial_data_processing(df):
    """

    :param df:
    :return:
    """
    wap_column_names = df.filter(regex=("WAP\d*")).columns
    df[df[wap_column_names] == 100] = np.nan  # 100 indicates an AP that wasn't detected
    spatial_mean = np.mean(df[wap_column_names], axis=1)
    df[wap_column_names] = df[wap_column_names].sub(spatial_mean, axis=0)  # spatial mean normalization
    return df


if __name__ == "__main__":
    training_data = pd.read_csv("sample_data/TrainingData.csv")
    validation_data = pd.read_csv("sample_data/ValidationData.csv")
    wap_column_names = training_data.filter(regex=("WAP\d*")).columns

    clusters_data = usp.cluster_training_data(training_data, NR=2)

    # Assuming we know the correct floor
    cur_scan_id = 10
    cur_scan_gt = validation_data.iloc[cur_scan_id]
    cur_scan_vals, cur_bid, cur_floor = cur_scan_gt[wap_column_names], cur_scan_gt["BUILDINGID"], cur_scan_gt["FLOOR"]
