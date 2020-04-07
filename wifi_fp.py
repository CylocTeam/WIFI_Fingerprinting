import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import similarity as sm
import radiomap as rm


def initial_data_processing(df):
    """
    initial procession of data (normalization + missing APs replacing)
    :param df: Dataframe with data to process
    :return: Dataframe with processed data
    """
    wap_column_names = df.filter(regex=("WAP\d*")).columns
    df[df[wap_column_names] == 100] = np.nan  # 100 indicates an AP that wasn't detected
    spatial_mean = np.mean(df[wap_column_names], axis=1)
    spatial_std = np.std(df[wap_column_names], axis=1)
    df[wap_column_names] = df[wap_column_names].sub(spatial_mean, axis=0) # spatial mean normalization
    return df


# intepolation functions... didn't seem relevant, considering the path we're going to take with the project

def interpolate_training_data(training_set, amount=1):
    grp = training_set.groupby(["BUILDINGID", "FLOOR", "PHONEID"])
    out_df = pd.DataFrame()
    for bid, floor, userid in grp.groups.keys():
        cur_grp = grp.get_group((bid, floor, userid))
        cur_grp_interp = interpolate_group(cur_grp,amount=amount)
        out_df = out_df.append(cur_grp_interp, ignore_index=True)
    return out_df


def interpolate_group(df, amount=1):
    # duplicate each line
    df = df.sort_values("TIMESTAMP")
    zeros = np.full(np.shape(df.values), np.nan).repeat(amount, axis=1)
    data = np.hstack([df.values, zeros]).reshape(-1, df.shape[1])
    df_ordered = pd.DataFrame(data, columns=df.columns)
    df_ordered.interpolate(method="linear", inplace=True)
    return df_ordered


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
    relev_aps = wap_column_names[~np.isnan(row[wap_column_names])]
    radiomap = rm_per_area[cur_bid, cur_floor]

    rm_rssi = radiomap.get_ap_maps_ndarray(relev_aps)
    weights = sm.rm_similarity_calculation(line[relev_aps], rm_rssi, p=2)
    xx, yy = radiomap.get_map_ranges()

    weights_qtile = np.nanquantile(weights, qtile)
    num_of_elem = np.sum(weights[~np.isnan(weights)] >= weights_qtile)
    if num_of_elem == 0:
        return np.nan, np.nan, np.nan

    wmx, wmy = wknn_find_location(weights, xx, yy, num_of_elem)
    error = np.linalg.norm([line.LONGITUDE - wmx, line.LATITUDE - wmy])

    if plot_flag:
        plt.figure()
        plt.imshow(weights, extent=radiomap.extent, origin="lower")
        plt.scatter(line.LONGITUDE, line.LATITUDE, c="r", marker="x")
        plt.scatter(wmx, wmy, c="m", marker="*")
        plt.grid()
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title('Validation results for 1-metric\nError = ' + str(round(error, 2)) + ' [m]\n' +
                  "BID = " + str(cur_bid) + ", FLOOR = " + str(cur_floor))
        plt.colorbar()
        plt.show()

    return wmx, wmy, error


if __name__ == "__main__":
    training_data = pd.read_csv("sample_data/TrainingData.csv")
    validation_data = pd.read_csv("sample_data/ValidationData.csv")
    wap_column_names = training_data.filter(regex=("WAP\d*")).columns
    training_data = training_data.drop(columns=["RELATIVEPOSITION", "USERID", "SPACEID"])

    training_data = initial_data_processing(training_data)
    training_data = interpolate_training_data(training_data,amount=2)
    validation_data = initial_data_processing(validation_data)

    rm_per_area = rm.create_radiomap_objects(training_data, [2, 2])

    validation_results = pd.DataFrame(np.nan, columns=('FPx', 'FPy', 'error'), index=validation_data.index)
    for index, row in validation_data.iterrows():
        cur_x, cur_y, cur_error = calculate_line_location(row, rm_per_area, qtile=0.95)
        validation_results.loc[index, ['FPx', 'FPy', 'error']] = [cur_x, cur_y, cur_error]
    verrors = validation_results.error[~np.isnan(validation_results.error)]

    error_90 = round(np.quantile(verrors, 0.9))
    error_med = round(np.median(verrors, axis=None), 2)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(verrors, 100, density=True, histtype='step', cumulative=True)
    plt.xlabel('Error [m]')
    plt.ylabel('Percentile')
    plt.title('Validation results for 1-metric\n90% error at ' + str(error_90) + ' [m]\n'+
              'Median error at ' + str(error_med) + '[m]')
    plt.grid()
    plt.show()

    for ii in validation_results.index:
        calculate_line_location(validation_data.loc[ii], rm_per_area, plot_flag=True)
