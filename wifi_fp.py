import matplotlib.pyplot as plt
from similarity import *
from radiomap import *
from scipy.stats import multivariate_normal as mvn
from datetime import datetime
import warnings

# ## GLOBAL DEFAULT VARIABLES
__MINIMAL_RSSI_VALUE = 0
__RAYLEIGH_RSSI_VAR = None
__INPUT_NAN_VALUE = 100
__KALMAN_STARTING_ERR = 10
__GRID_SIZE = [2, 2]
__GRID_PADDING = [50, 50]
__STATE_MODEL_NOISE = np.diag([8.3, 8.3])

def initial_data_processing(df, res_min=0, res_var=None):
    """
    initial procession of data (normalization + missing APs replacing)
    :param res_var: normalized rayleigh distribution variance
    :param res_min: normalized rayleigh distribution minimal value
    :param df: Dataframe with data to process
    :return: Dataframe with processed data
    """
    df = df.sort_values(by=["PHONEID", "TIMESTAMP"])
    wap_column_names = df.filter(regex="WAP\d*").columns
    df[df[wap_column_names] == __INPUT_NAN_VALUE] = np.nan  # 100 indicates an AP that wasn't detected

    # group by device ID
    pid_grp = df.groupby(["PHONEID"])
    phone_nrm = pd.DataFrame()
    phone_nrm["min"] = pid_grp.agg({cn: np.nanmin for cn in wap_column_names}).min(axis=1)
    phone_nrm["std"] = pid_grp.agg({cn: rayleigh_dist_std for cn in wap_column_names}).sum(axis=1)
    if not res_var:
        res_var = phone_nrm["std"].iloc[0] # if no set var, var is the var of the first phone

    df[wap_column_names] = df[wap_column_names].subtract(phone_nrm["min"].loc[df["PHONEID"]].values, axis=0)
    df[wap_column_names] = df[wap_column_names].divide(phone_nrm["std"].loc[df["PHONEID"]].values, axis=0)
    df[wap_column_names] = df[wap_column_names].mul(res_var).add(res_min)
    return df


def rayleigh_dist_std(df):
    """
    ML estimation of Rayleigh standard deviation (biased)
    :param df: dataframe containing rayleigh distributed data
    :return: ML estimation of rayleigh std
    """
    if len(df) == 0:
        return np.Inf
    return np.sqrt((df ** 2).sum() / (2 * len(df)))


def perform_kalman_filter_fp(df, radiomap, plot=False, qtile=0.9):
    """
    Perform Fingerprint Kalman Filter location estimation for dataframe of a specific device located within radiomap.
    Also plot the data to a figure and show it.
    :param df: DataFrame containing AP + location + device data
    :param radiomap: radiomap object containing data for all APs
    :param plot: Flag, whether or not to plot each iteration's data to figure
    :return: dictonary containing (location, est_error, real_error) for each iteration
    """
    wap_column_names = df.filter(regex="WAP\d*").columns
    xx, yy = radiomap.get_map_ranges()
    sgrid = np.dstack(np.meshgrid(xx, yy))
    rm_rssi, rm_rssi_var = radiomap.get_ap_maps_ndarray(wap_column_names)

    # find the relevant APs of the device-area
    empty_slices = np.isnan(rm_rssi_var).all(axis=(0, 1)) | np.isnan(rm_rssi).all(axis=(0, 1))
    relev_ind = ~empty_slices | (~np.isnan(df[wap_column_names])).any(axis=0)
    rm_rssi, rm_rssi_var, wap_column_names = rm_rssi[:, :, relev_ind], rm_rssi_var[:, :, relev_ind], wap_column_names[relev_ind]
    rm_rssi = np.where(~np.isnan(rm_rssi), rm_rssi, __MINIMAL_RSSI_VALUE-1)

    # initialize the data formats
    kf_results = {"loc": [], "err": [], "real_err": []}  # store results
    ppi = np.diag([np.mean(np.diff(xx) ** 2) / 12, np.mean(np.diff(yy) ** 2) / 12])

    # initialize location,error with no info
    loc, err = [np.nanmean(xx), np.nanmean(yy)], np.diag([((np.max(xx)-np.min(xx))**2)/12,
                                                          ((np.max(yy)-np.min(yy))**2)/12])

    for ind, row in df.iterrows():
        # STATE TRANSITION
        # stationary model (simplest one)
        xk_min, pk_min = loc, err + __STATE_MODEL_NOISE

        # ## ESTIMATION STEP
        # get current step data
        sigma = np.where(np.isnan(rm_rssi_var), 0, rm_rssi_var)
        rlv_cl = ~np.isnan(row[wap_column_names]) & ~np.all(rm_rssi == 0, axis=(0, 1)) & ~(sigma == 0).all(axis=(0, 1))
        yk, rlv_rssi, sigma = row[wap_column_names[rlv_cl]], rm_rssi[:, :, rlv_cl], sigma[:, :, rlv_cl]

        # location-cell weights
        bstk_nnrm = mvn.pdf(sgrid, xk_min, pk_min)
        bstk_s = bstk_nnrm[:, :, np.newaxis] / np.sum(bstk_nnrm)

        # prior estimations
        y_hat = np.nanmean(bstk_s * rlv_rssi, axis=(0, 1))
        p_hat = np.sum(sgrid * bstk_s, axis=(0, 1))

        # ## UPDATE STEP
        # calculate location and rssi diffs from prior estimation
        ppdf = sgrid - p_hat
        pydf = rlv_rssi - y_hat

        # calculate the covariance matrices
        pxxk = np.einsum("ijk,ijw", ppdf * bstk_s, ppdf) + ppi
        pxyk = np.einsum("ijk,ijw", ppdf * bstk_s, pydf)
        pyyk = np.einsum("ijk,ijw", pydf * bstk_s, pydf) + np.diag(np.nansum(sigma * bstk_s, axis=(0, 1)))

        # calculate the new location estimation and eerror
        K = np.matmul(pxyk, np.linalg.inv(pyyk))
        loc = xk_min + np.matmul(K, (yk - y_hat))
        err = pxxk - np.matmul(K, pxyk.T)
        real_err = np.array([row.LONGITUDE, row.LATITUDE]) - loc
        kf_results["loc"].append(loc)
        kf_results["err"].append(err)
        kf_results["real_err"].append(real_err)

        if plot:
            a, b, t = calculate_error_ellipse(err, qtile)
            pid, bid, fid, ts = row[["PHONEID", "BUILDINGID", "FLOOR", "TIMESTAMP"]]
            time = datetime.fromtimestamp(ts)
            ttl = "\n".join(["", "Phone: " + str(pid), "BUILDING: " + str(bid) + ", FLOOR: " + str(fid),
                             "TIME: " + str(time), "Error: " + str(round(np.linalg.norm(real_err), 2)) + " [m]",
                             "Ellipse Axis: " + str(round(a, 2)) + " [m], " + str(round(b, 2)) + " [m]"])
            fig = plt.gcf()
            fig.clf()
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
    """
    Get an error matrix and calculate the axis and angle of ellipse
    :param err: 2X2 Error matrix
    :param prc: percentile of error to contain (def: 90%)
    :return: (half major axis size, half minor axis size, angle)
    """
    kappa = -2*np.log(1-prc)
    eigval, eigvec = np.linalg.eig(err)
    hmja, hmia = kappa * np.sqrt(np.max(eigval)), kappa * np.sqrt(np.min(eigval))
    hmj_ind = np.argmax(eigval)
    ang = np.arctan2(eigvec[hmj_ind, 1], eigvec[hmj_ind, 0])
    return hmja, hmia, ang


def plot_ellipse(center, a, b, t, color="b"):
    """
    Get data of an ellipse and plot it to figure
    :param center: center of ellipse
    :param a: half major axis size
    :param b: half minor axis size
    :param t: angle of ellipse (to x direction)
    :param color: color of line plot
    """
    theta = np.arange(0, 2 * np.pi, 0.01)
    ppx, ppy = np.matmul([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], [a * np.cos(theta), b * np.sin(theta)])
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
    X, Y = np.meshgrid(xx, yy)
    w, x, y = W.flatten(), X.flatten(), Y.flatten()

    # find top K elements and return their weighted mean
    selem_ind = w.argsort(axis=None)[-K:]  # biggest K elem indices
    wx = np.average(x[selem_ind], weights=w[selem_ind])
    wy = np.average(y[selem_ind], weights=w[selem_ind])
    return wx, wy


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
    rm_rssi = np.where(~np.isnan(rm_rssi), rm_rssi, __MINIMAL_RSSI_VALUE)  # according to rayleigh normalization
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

        fig = plt.gcf()
        fig.clf()
        plt.imshow(weights / num_of_aps, extent=radiomap.extent, origin="lower", vmin=0)
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
    # Read training and validation data
    training_data = pd.read_csv("sample_data/TrainingData.csv")
    validation_data = pd.read_csv("sample_data/ValidationData.csv")

    # simple initial data processing of training and validation data
    training_data = training_data.drop(columns=["RELATIVEPOSITION", "USERID", "SPACEID"])
    training_data = initial_data_processing(training_data, res_var=__RAYLEIGH_RSSI_VAR, res_min=__MINIMAL_RSSI_VALUE)
    validation_data = initial_data_processing(validation_data, res_var=__RAYLEIGH_RSSI_VAR,
                                              res_min=__MINIMAL_RSSI_VALUE)

    # create the radiomap object from training set
    rm_per_area = create_radiomap_objects(training_data, __GRID_SIZE, padding=__GRID_PADDING)

    # iterate per (device, building, floor) and perform Fingerprint Kalman Filter estimation per line
    grp_res = {}
    vgrp = validation_data.groupby(["PHONEID", "BUILDINGID", "FLOOR"])
    for name, cvgrp in vgrp:
        # TODO: add building and floor estimation via mean Jacaard metric for APs
        phone_id, building_id, floor = name
        radiomap = rm_per_area[building_id, floor]
        with warnings.catch_warnings():  # suppress runtime errors
            warnings.simplefilter("ignore", category=RuntimeWarning)
            grp_res[name] = perform_kalman_filter_fp(cvgrp, radiomap, plot=True)
