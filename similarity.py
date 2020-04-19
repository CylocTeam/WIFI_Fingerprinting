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
    weights = (np.nansum(abs((d - c)**p), axis=1) ** (1/p)) #/ (np.sum(d > -100, axis=1))
    weights[np.isnan(d).all(axis=1)] = np.nan
    return pd.Series(1 / weights, index=df.index)


def rm_similarity_calculation (cur_ap_rssi, rm, p=1):
    """
    calculate the p-norm similarity between an rssi vector and radiomap
    :param cur_ap_rssi: vector including the current RSSI data to compare to the radiomap (array of 1xN)
    :param rm: radiomap containing fingerprint data (narray of A x B x N)
    :param p: norm parameter
    :return: narray of size (AxB) with similarity value at each point of the radiomap
    """
    c = cur_ap_rssi.to_numpy()
    c_shft = np.moveaxis(c.reshape((-1,) + (1,) * (rm.ndim - 1)), 0, -1)
    cc = c_shft.repeat(np.size(rm, axis=0), axis=0).repeat(np.size(rm, axis=1), axis=1)
    weights_map = (np.nansum(abs(rm - cc) ** p, axis=2)) ** (1 / p)
    weights_map[weights_map == 0] = np.nan
    weights_map[np.any(np.isnan(rm), axis=2)] = np.nan # amount of unallowed nans in vector (aka how many undetected APS)
    weights_map = 1/weights_map - 1/np.nansum(abs(cur_ap_rssi)) # first term is the weight, 2nd is centering
    weights_map[weights_map == 0] = np.nan
    return weights_map
