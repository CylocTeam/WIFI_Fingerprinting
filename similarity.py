import numpy as np
import scipy.stats as stt

def similarity_calc_rank(cur_ap_rssi, rm_rssi):
    ranks_cur = stt.rankdata(cur_ap_rssi, method='dense'); ranks_cur[cur_ap_rssi > 0] = 0
    ranks_rm = stt.rankdata(rm_rssi, method='dense'); ranks_rm[rm_rssi > 0] = 0
    spearman_footrule = sum(abs(ranks_cur[cur_ap_rssi < 0] - ranks_rm[cur_ap_rssi < 0]))
    return 1/(spearman_footrule + 1)


def similarity_calculation (cur_ap_rssi, df, p=1):
    """
    Calculate the similarity between 2 RSSI rank vectors
    :param ranks1: cur rssi vector that is compared
    :param ranks2: radiomap rssi vector
    :return:
    """

    # TODO: find literature about weighting where not all AP's were seen. This causes issues here
    df = df.mask(np.isnan(df) & ~np.isnan(cur_ap_rssi), -200) # assumption about RSSI values that weren't found
    weights = ((abs(df.sub(cur_ap_rssi, axis=1))**p).sum(axis=1))**(1/p)
    weights[weights == 0] = np.nan
    return 1 / weights
