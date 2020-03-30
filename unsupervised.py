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
            cluster_data[col].append([cmin, cmax + NR])
    return cluster_data


def create_fingerprint_wb(df, cluster_data):
    ap_clusts_df = match_lines_to_clusters(df, cluster_data)

    # unique rows that don't contain (-1) value are the fingerprints
    missing_rows = (ap_clusts_df < 0).any(axis=1)
    ap_clusts_df = ap_clusts_df.drop(index=missing_rows[missing_rows].index)

    return ap_clusts_df.drop_duplicates().reindex()


def match_lines_to_clusters(df, cluster_data):
    wap_column_names = df.filter(regex=("WAP\d*")).columns
    ap_clusts = {}
    for ap in wap_column_names:
        cur_ap_meas = df[ap]
        cur_ap_clust_data = cluster_data[ap]
        ap_clst = pd.Series(np.nan, index=cur_ap_meas.index)
        cur_ap_meas = cur_ap_meas[~np.isnan(cur_ap_meas)]
        for idx, val in enumerate(cur_ap_clust_data):
            clust_min, clust_max = val
            ind = cur_ap_meas.loc[(cur_ap_meas > clust_min) & (cur_ap_meas < clust_max)].index
            ap_clst[ind] = idx
        ap_clst[(~np.isnan(cur_ap_meas)) & (np.isnan(ap_clst))] = -1  # mark that cluster wasn't found
        ap_clusts[ap] = ap_clst
    return pd.DataFrame(ap_clusts, index=df.index)


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
