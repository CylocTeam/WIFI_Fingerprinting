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

def calculate_line_location(line, rm_per_area, aps_per_area, qtile=0.95):
    cur_bid, cur_floor = line["BUILDINGID"], line["FLOOR"]
    radiomap, wap_column_names = rm_per_area[cur_bid, cur_floor], aps_per_area[cur_bid, cur_floor]

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
    training_data = training_data.drop(columns=["RELATIVEPOSITION", "USERID", "SPACEID"])

    training_data = initial_data_processing(training_data)
    training_data = interpolate_training_data(training_data,amount=5)
    validation_data = initial_data_processing(validation_data)

    rm_per_area, aps_per_area = rm.create_radio_map(training_data, [2, 2])

    validation_results = pd.DataFrame(np.nan, columns=('FPx', 'FPy', 'error'), index=validation_data.index)
    for index, row in validation_data.iterrows():
        cur_x, cur_y, cur_error = calculate_line_location(row, rm_per_area, aps_per_area, qtile=0.95)
        validation_results.loc[index, ['FPx', 'FPy', 'error']] = [cur_x, cur_y, cur_error]

    error_90 = np.quantile(validation_results.error, 0.9)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(validation_results.error, 100, density=True, histtype='step', cumulative=True)
    plt.xlabel('Error [m]')
    plt.ylabel('Percentile')
    plt.title('Validation results for 1-metric\n90% error at ' + str(round(error_90, 2)) + ' [m]')
    plt.grid()
    plt.show()

