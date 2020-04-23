import matplotlib.pyplot as plt
from similarity import *
from radiomap import *
from scipy.stats import multivariate_normal as mvn
from datetime import datetime

def initial_data_processing(df, res_mean=0, res_sigma=1):
    """
    initial procession of data (normalization + missing APs replacing)
    :param df: Dataframe with data to process
    :return: Dataframe with processed data
    """
    df = df.sort_values(by=["PHONEID", "TIMESTAMP"])
    wap_column_names = df.filter(regex=("WAP\d*")).columns
    df[df[wap_column_names] == 100] = np.nan  # 100 indicates an AP that wasn't detected

    # group by device ID
    pid_grp = df.groupby(["PHONEID"])
    phone_nrm = pd.DataFrame()
    phone_nrm["min"] = pid_grp.agg({cn: np.nanmin for cn in wap_column_names}).min(axis=1)
    phone_nrm["std"] = pid_grp.agg({cn: rayleigh_dist_std for cn in wap_column_names}).sum(axis=1)
    df[wap_column_names] = df[wap_column_names].subtract(phone_nrm["min"].loc[df["PHONEID"]].values, axis=0)
    df[wap_column_names] = df[wap_column_names].divide(phone_nrm["std"].loc[df["PHONEID"]].values, axis=0)
    df[wap_column_names] = df[wap_column_names].mul(res_sigma).add(res_mean)
    return df


def rayleigh_dist_std(df):
    """
    ML estimation of Rayleigh standard deviation (biased)
    :param df:
    :return:
    """
    if len(df) == 0:
        return np.Inf
    return np.sqrt((df ** 2).sum() / (2 * len(df)))


def perform_kalman_filter_fp(df, radiomap, plot=False):
    wap_column_names = df.filter(regex=("WAP\d*")).columns
    xx, yy = radiomap.get_map_ranges()
    XX, YY = np.meshgrid(xx, yy)
    rm_rssi, rm_rssi_var = radiomap.get_ap_maps_ndarray(wap_column_names)
    rm_rssi = np.where(~np.isnan(rm_rssi), rm_rssi, 0)

    kf_results = {"loc": [], "err": [], "real_err": []}
    ppi = np.diag([np.mean(np.diff(xx) ** 2) / 12, np.mean(np.diff(yy) ** 2) / 12])
    sigma = np.sqrt(np.nanmean(rm_rssi_var, axis=(0,1))) # we'll use an average sigma to avoid single-sample errors

    loc = [np.nanmean(xx), np.nanmean(yy)] # initialize location at the center of map
    err = np.diag([1000, 1000]) ** 2 # before iterating we have no information
    for ind, row in df.iterrows():
        # state transition of stationary model (simplest one)
        rlv_cl = ~np.isnan(row[wap_column_names]) & ~np.isnan(sigma)
        yk, rlv_rssi, rlv_sigma = row[wap_column_names[rlv_cl]], rm_rssi[:,:,rlv_cl], sigma[rlv_cl]

        bstk = mvn.pdf(np.dstack([XX, YY]), loc, err)
        bstk /= np.sum(bstk)
        bstk_s = bstk[:, :, np.newaxis]

        y_hat = np.nanmean(bstk_s * rlv_rssi, axis=(0,1))
        p_hat = [(bstk * XX).sum(),  (bstk * YY).sum()]

        # calculate the covariance matrices
        ppdf = np.dstack([XX, YY]) - p_hat
        pydf = rlv_rssi - y_hat

        ppxk = np.einsum("ijk,ijw", ppdf*bstk_s, ppdf) + ppi # PXXk
        ppyk = np.einsum("ijk,ijw", ppdf*bstk_s, pydf) # PXYk
        pyyk = np.einsum("ijk,ijw", pydf*bstk_s, pydf) + np.diag(rlv_sigma) # PYYk

        # calculate the new location estimation and eerror
        K = np.matmul(ppyk, np.linalg.inv(pyyk))
        loc = loc + np.matmul(K, (yk - y_hat))
        err = ppxk - np.matmul(K, ppyk.T)
        real_err = np.array([row.LONGITUDE, row.LATITUDE]) - loc
        kf_results["loc"].append(loc)
        kf_results["err"].append(err)
        kf_results["real_err"].append(real_err)

        if plot:
            a, b, t = calculate_error_ellipse(err)
            pid, bid, fid, ts = row[["PHONEID", "BUILDINGID", "FLOOR", "TIMESTAMP"]]
            time = datetime.fromtimestamp(ts)
            ttl = "\n".join(["", "Phone: " + str(pid), "BUILDING: " + str(bid) + ", FLOOR: " + str(fid),
                             "TIME: " + str(time), "Error: " + str(np.linalg.norm(real_err)) + " [m]",
                             "Ellipse HJA: " + str(a) + " [m]"])
            fig = plt.gcf(); fig.clf()
            plt.imshow(np.linalg.norm(pydf, axis=2), extent=radiomap.extent, origin="lower", vmin=0)
            plt.colorbar()
            plt.scatter(row.LONGITUDE, row.LATITUDE, c="r", marker="x")
            plt.scatter(loc[0], loc[1], c="m", marker="*")
            plot_ellipse(loc, a, b, t, color="m")
            plt.grid()
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title(ttl)
            plt.draw()
            plt.pause(.2)
    return kf_results


def calculate_error_ellipse(err, prc=0.9):
    #kappa = -2*np.log(1-prc)
    kappa = 1
    eigval, eigvec = np.linalg.eig(err)
    hmja, hmia = kappa*np.sqrt(np.max(eigval)), kappa*np.sqrt(np.min(eigval))
    hmj_ind = np.argmax(eigval)
    ang = np.arctan2(eigvec[hmj_ind, 1], eigvec[hmj_ind, 0])
    return hmja, hmia, ang


def plot_ellipse(center, a,b,t, color="b"):
    theta = np.arange(0, 2*np.pi, 0.01)
    ppx, ppy = np.matmul([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]],[a*np.cos(theta), b*np.sin(theta)])
    plt.plot(center[0] + ppx, center[1] + ppy, color=color)
    return


def wknn_find_location(weights, xx, yy, K):
    """
    Estimate the location Weighted K Nearest Neighbor wise, according to weights in the similarity map
    A simple weighted means of K top elements
    :param weights: (NxM) array of similarity weights
    :param xx: (1xN) array of x-values
    :param yy: (1xM) array of y-values
    :param K: number of elements to add into the calculation
    :return: (x,y) estimated location
    """
    W = np.nan_to_num(weights, nan=0)
    X,Y = np.meshgrid(xx, yy)
    w, x, y = W.flatten(), X.flatten(), Y.flatten()

    # find top K elements and return their weighted mean
    selem_ind = w.argsort(axis=None)[-K:] # biggest K elem indices
    wx = np.average(x[selem_ind], weights=w[selem_ind])
    wy = np.average(y[selem_ind], weights=w[selem_ind])
    return wx,wy


def calculate_line_location(line, rm_per_area, qtile=0.95, plot_flag=False):
    """
    Perform a localization (WKNN-wise) on a specific line in the validation set.
    If plot_flag is specificed, also plot the data into a figure
    :param line: current line containing validation (RSSI values, location, building an floor IDs)
    :param rm_per_area: dictionary containing all radiomaps with (BID, FLOORID) as key
    :param qtile: quantile of weights to use of WKNN
    :param plot_flag: if specified as True, data will be plotted into a figure
    :return: estimated location and error, (x,y,error)
    """
    cur_bid, cur_floor = line["BUILDINGID"], line["FLOOR"]
    wap_column_names = line.filter(regex=("WAP\d*")).index
    relev_aps = wap_column_names[~np.isnan(line[wap_column_names])]
    radiomap = rm_per_area[cur_bid, cur_floor]

    rm_rssi, rm_rssi_var = radiomap.get_ap_maps_ndarray(relev_aps)
    rm_rssi = np.where(~np.isnan(rm_rssi), rm_rssi, 0) # according to rayleigh normalization
    weights = rm_similarity_calculation(line[relev_aps], rm_rssi, p=1)
    xx, yy = radiomap.get_map_ranges()

    weights_qtile = np.nanquantile(weights, qtile)
    num_of_elem = np.sum(weights[~np.isnan(weights)] > weights_qtile)
    if num_of_elem == 0:
        return np.nan, np.nan, np.nan

    wmx, wmy = wknn_find_location(weights, xx, yy, num_of_elem)
    error = np.linalg.norm([line.LONGITUDE - wmx, line.LATITUDE - wmy])

    if plot_flag:
        num_of_aps = np.sum(~np.isnan(line[wap_column_names]))

        fig = plt.gcf(); fig.clf()
        plt.imshow(weights/num_of_aps, extent=radiomap.extent, origin="lower", vmin=0)
        plt.colorbar()
        plt.scatter(line.LONGITUDE, line.LATITUDE, c="r", marker="x")
        plt.scatter(wmx, wmy, c="m", marker="*")
        plt.grid()
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title('Validation results for 1-metric\nError = ' + str(round(error, 2)) + ' [m]\n' +
                  "BID = " + str(cur_bid) + ", FLOOR = " + str(cur_floor) + "\n" +
                  "# of APs = " + str(num_of_aps))
        plt.draw()
        plt.pause(1)
    return wmx, wmy, error


if __name__ == "__main__":
    training_data = pd.read_csv("sample_data/TrainingData.csv")
    validation_data = pd.read_csv("sample_data/ValidationData.csv")
    wap_column_names = training_data.filter(regex=("WAP\d*")).columns
    training_data = training_data.drop(columns=["RELATIVEPOSITION", "USERID", "SPACEID"])

    training_data = initial_data_processing(training_data, res_sigma=1000)
    validation_data = initial_data_processing(validation_data, res_sigma=1000)

    rm_per_area = create_radiomap_objects(training_data, [2, 2], padding=(50,50))

    validation_results = pd.DataFrame(np.nan, columns=('FPx', 'FPy', 'error'), index=validation_data.index)

    plt.figure()
    grp_res = {}
    vgrp = validation_data.groupby(["PHONEID", "BUILDINGID", "FLOOR"])
    for name, cvgrp in vgrp:
        phone_id, building_id, floor = name
        radiomap = rm_per_area[building_id, floor]
        grp_res[name] = perform_kalman_filter_fp(cvgrp, radiomap, plot=True)

