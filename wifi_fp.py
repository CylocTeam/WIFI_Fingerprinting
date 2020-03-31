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
    df[df[wap_column_names] == 100] = -100  # 100 indicates an AP that wasn't detected
    spatial_mean = np.mean(df[wap_column_names], axis=1)
    spatial_std = np.std(df[wap_column_names], axis=1)
    df[wap_column_names] = df[wap_column_names].sub(spatial_mean, axis=0).divide(spatial_std, axis=0)  # spatial mean normalization
    return df


def calculate_line_location(line, rm_per_area, qtile=0.95):
    wap_column_names = training_data.filter(regex=("WAP\d*")).columns
    cur_bid, cur_floor = line["BUILDINGID"], line["FLOOR"]
    radiomap = rm_per_area[cur_bid, cur_floor]

    weights = sm.similarity_calculation(line[wap_column_names], radiomap[wap_column_names])

    weights_qtile = np.nanquantile(weights, qtile)
    weighted_mean_lon = np.average(radiomap.x[weights >= weights_qtile],
                                   weights=weights[weights >= weights_qtile])
    weighted_mean_lat = np.average(radiomap.y[weights >= weights_qtile],
                                   weights=weights[weights >= weights_qtile])
    error = np.sqrt((line.LONGITUDE - weighted_mean_lon) ** 2 +
                    (line.LATITUDE - weighted_mean_lat) ** 2)

    return weighted_mean_lon, weighted_mean_lat, error


if __name__ == "__main__":
    training_data = pd.read_csv("sample_data/TrainingData.csv")
    validation_data = pd.read_csv("sample_data/ValidationData.csv")
    wap_column_names = training_data.filter(regex=("WAP\d*")).columns

    training_data = initial_data_processing(training_data)
    validation_data = initial_data_processing(validation_data)

    rm_per_area, aps_per_area = rm.create_radio_map(training_data, [2, 2])

    # Assuming we know the correct floor
    # TODO: add floor estimation
    # cur_scan_id = 20
    # cur_scan_gt = validation_data.iloc[cur_scan_id]
    # cur_scan_vals, cur_bid, cur_floor = cur_scan_gt[wap_column_names], cur_scan_gt["BUILDINGID"], cur_scan_gt["FLOOR"]
    #
    # weights, centroid, error = calculate_line_location(cur_scan_gt, rm_per_area)

    validation_results = pd.DataFrame(np.nan, columns=('FPx', 'FPy', 'error'), index=validation_data.index)
    for index, row in validation_data.iterrows():
        cur_x, cur_y, cur_error = calculate_line_location(row, rm_per_area, qtile=0.95)
        validation_results.loc[index, ['FPx', 'FPy', 'error']] = [cur_x, cur_y, cur_error]

    error_90 = np.quantile(validation_results.error, 0.9)
    print("90% error is: " + str(error_90) + " [m]")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(validation_results.error, 100, density=True, histtype='step', cumulative=True)
    plt.show()

