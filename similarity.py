import numpy as np
import scipy.stats as stt
import pandas as pd

def similarity_calc_rank(cur_ap_rssi, rm_rssi):
    """
    OLD VERSION
    calulcate the similarity between 2 RSSI Rank vectors
    :param cur_ap_rssi: vector 1 to compare, the cur point's
    :param rm_rssi: radiomap rssi vector
    :return: weight value of similarity
    """
    ranks_cur = stt.rankdata(cur_ap_rssi, method='dense'); ranks_cur[cur_ap_rssi > 0] = 0
    ranks_rm = stt.rankdata(rm_rssi, method='dense'); ranks_rm[rm_rssi > 0] = 0
    spearman_footrule = sum(abs(ranks_cur[cur_ap_rssi < 0] - ranks_rm[cur_ap_rssi < 0]))
    return 1/(spearman_footrule + 1)


def similarity_calculation (cur_ap_rssi, df, p=1):
    """
    Calculate the similarity between 2 RSSI vectors using p-metric as distance
    :param ranks1: cur rssi vector that is compared
    :param ranks2: radiomap rssi dataframe containing WAP RSSI values for each grid point
    :return: Series containing weight value for each grid point
    """
    c, d = cur_ap_rssi.to_numpy(), df.to_numpy()
    weights = np.nansum(abs((d - c)**p), axis=1) ** (1/p)
    weights[np.isnan(d).all(axis=1)] = np.nan
    return pd.Series(1 / weights, index=df.index)

