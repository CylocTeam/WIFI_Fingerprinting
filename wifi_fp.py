import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import similarity as sm
import radiomap as rm


def initial_data_processing(df):
    """

    :param df:
    :return:
    """
    wap_column_names = df.filter(regex=("WAP\d*")).columns
    df[df[wap_column_names] == 100] = np.nan  # 100 indicates an AP that wasn't detected
    spatial_mean = np.mean(df[wap_column_names], axis=1)
    spatial_std = np.std(df[wap_column_names], axis=1)
    df[wap_column_names] = df[wap_column_names].sub(spatial_mean, axis=0) # spatial mean normalization
    return df


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
    W = np.nan_to_num(weights, nan=0)
    Y,X = np.meshgrid(yy,xx)
    w, x, y = W.flatten(), X.flatten(), Y.flatten()

    # find top K elements
    selem_ind = w.argsort(axis=None)[-K:] # biggest K elem indices
    wx = np.average(x[selem_ind], weights=w[selem_ind])
    wy = np.average(y[selem_ind], weights=w[selem_ind])
    return wx,wy


def calculate_line_location(line, rm_per_area, qtile=0.95):
    cur_bid, cur_floor = line["BUILDINGID"], line["FLOOR"]
    wap_column_names = line.filter(regex=("WAP\d*")).index
    relev_aps = wap_column_names[~np.isnan(row[wap_column_names])]
    radiomap = rm_per_area[cur_bid, cur_floor]

    rm_rssi = radiomap.get_ap_maps_ndarray(relev_aps)
    weights = sm.rm_similarity_calculation(line[relev_aps], rm_rssi)
    xx, yy = radiomap.get_map_ranges()

    weights_qtile = np.nanquantile(weights, qtile)
    num_of_elem = np.sum(weights >=weights_qtile)
    if num_of_elem == 0:
        return np.nan, np.nan, np.nan, np.nan

    wmx, wmy = wknn_find_location(weights, xx, yy, num_of_elem)
    error = np.linalg.norm([line.LONGITUDE - wmx, line.LATITUDE - wmy])

    return weights, wmx, wmy, error


def plot_line_calculation(line, rm_per_area, qtile=0.95):
    cur_bid, cur_floor = line["BUILDINGID"], line["FLOOR"]
    wap_column_names = line.filter(regex=("WAP\d*")).index
    relev_aps = wap_column_names[~np.isnan(row[wap_column_names])]
    radiomap = rm_per_area[cur_bid, cur_floor]

    rm_rssi = radiomap.get_ap_maps_ndarray(relev_aps)
    weights = sm.rm_similarity_calculation(line[relev_aps], rm_rssi)
    xx, yy = radiomap.get_map_ranges()

    weights_qtile = np.nanquantile(weights, qtile)
    num_of_elem = np.sum(weights >= weights_qtile)
    if num_of_elem == 0:
        return np.nan, np.nan, np.nan, np.nan

    wmx, wmy = wknn_find_location(weights, xx, yy, num_of_elem)
    error = np.linalg.norm([line.LONGITUDE - wmx, line.LATITUDE - wmy])

    plt.figure()
    plt.imshow(weights, extent=radiomap.extent, origin="lower")
    plt.scatter(line.LONGITUDE, line.LATITUDE, c="r", marker="x")
    plt.scatter(wmx, wmy, c="m", marker="*")
    plt.grid()
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title('Validation results for 1-metric\n90% error at ' + str(round(error, 2)) + ' [m]\n' +
              "BID: " + str(cur_bid) + ", FLOOR: " + str(cur_floor))
    plt.colorbar()
    plt.show()
    DEBUG = 1


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
        _, cur_x, cur_y, cur_error = calculate_line_location(row, rm_per_area, qtile=0.95)
        validation_results.loc[index, ['FPx', 'FPy', 'error']] = [cur_x, cur_y, cur_error]
    verrors = validation_results.error[~np.isnan(validation_results.error)]


    error_90 = np.quantile(verrors, 0.9)
    error_med = np.median(verrors)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(verrors, 100, density=True, histtype='step', cumulative=True)
    plt.xlabel('Error [m]')
    plt.ylabel('Percentile')
    plt.title('Validation results for 1-metric\n90% error at ' + str(round(error_90, 2)) + ' [m]\n'+
              'Median error at ' + str(round(error_med, 2)) + '[m]')
    plt.grid()
    plt.show()

    for ii in validation_results[validation_results.error > 40].index:
        plot_line_calculation(validation_data.loc[ii], rm_per_area)
