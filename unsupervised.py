import pandas as pd
import numpy as np
import json

def cluster_training_data(df, NR=2, min_density=5):
    """
    Take WIFI training data and cluster each APs rssi values into different clusters.
    Process specified in 'DCCLA: Automatic Indoor Localization Using Unsupervised Wi-Fi Fingerprinting'
    This is the first step detailed in the article
    :param df: DataFrame containing all WIFI data (RSSI value for each AP for each time point)
    :param NR: Neighborhood range, RSSI maximal difference to be considered a single cluster
    :param min_density: Minimal cluster size
    :return: Dictonary containing list of lists. Key=AP name. For each AP list containing cluster [min_value, max_value]
    of cluster. So: D[AP1] = [[cluster1_min, cluster1_max], [cluster2_min, cluster2_max], ...]
    """
    wap_column_names = df.filter(regex=("WAP\d*")).columns
    cluster_data = {}
    for col in wap_column_names:
        cluster_data[col] = []
        sorted_col = df[col].sort_values()
        cur_col = (sorted_col.diff() > NR).cumsum()
        cur_col = cur_col.mask(np.isnan(sorted_col), np.nan)
        for c in np.unique(cur_col[~np.isnan(cur_col)]):
            clust = sorted_col[cur_col == c]
            clen, cmin, cmax = len(clust), min(clust), max(clust)
            if clen < min_density:
                continue
            cluster_data[col].append([cmin, cmax])
    return cluster_data

def wifi_log_data_parser(file):
    """
    Parse WIFI scans data from application "Sensor Log" into a comfortable DataFrame format
    :param file: path to file
    :return: Dataframe containing all data from file (with all columns specified from file)
    """
    fo = open(file, "r")
    rfile = fo.read()
    splt = rfile.split("\n")
    parsed_data = {}
    for line in splt[3:-1]:
        line_splt = line.split("|")
        line_dict = json.loads(line_splt[2])
        for k in line_dict.keys():
            if k not in parsed_data:
                parsed_data[k] = []
            parsed_data[k].append(line_dict[k])
    return pd.DataFrame(parsed_data)
