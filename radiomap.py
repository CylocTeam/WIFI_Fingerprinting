import numpy as np
import pandas as pd


def create_radio_map(training_data, grid_size=(1, 1)):
    """
    Create radiomap for each area in training data dataset. Each area will contain a unique (building_id, floor) key
    RM will be in form of a DataFrame with grid indices as index, and columns (x,y,WAPXXX...) with the average RSSI value
    for each AP in each (x,y).
    Function also returns a dictionary contatining a list of all APs in each area.
    :param training_data: DataFrame containing data to train RM with
    :param grid_size: tuple (dx,dy) of grid size in (x,y) direction
    :return: (rm_per_area, aps_per_area): dictionaries with keys (building_id, floor), containing the RM and AP of each area.
    """
    wap_column_names = training_data.filter(regex=("WAP\d*")).columns
    unique_areas = training_data[["BUILDINGID", "FLOOR"]].drop_duplicates()
    rm_per_area, aps_per_area = {}, {}
    for area in unique_areas.values:
        building_id, floor = area
        cur_training_data = training_data.loc[(training_data[["BUILDINGID", "FLOOR"]] == area).all(axis=1)]
        grid_x, grid_y = get_grid_edges(cur_training_data, grid_size)
        rcpt_aps = wap_column_names[(~np.isnan(cur_training_data[wap_column_names])).any(axis=0)]
        cur_rm = get_radiomap_at(cur_training_data, (grid_x, grid_y, grid_size))
        rm_per_area[building_id, floor], aps_per_area[building_id, floor] = cur_rm, rcpt_aps
    return rm_per_area, aps_per_area


def get_radiomap_at(training_data, grid):
    """
    Create radiomap in a specific extent, using training_data
    :param training_data: DataFrame containing data to train RM with
    :param grid: tuple (grid_lon, grid_lat, grid_size) = ([grid_min_lon, grid_max_lon], [grid_min_lat, grid_max_lat], (dx,dy))
    :return: Dataframe containing RM in the specified extent
    """
    wap_column_names = training_data.filter(regex=("WAP\d*")).columns
    grid_lon, grid_lat, grid_size = grid
    grid_anchor = [min(grid_lon), min(grid_lat)]
    GX, GY = np.meshgrid(np.arange(grid_lon[0], grid_lon[1], grid_size[0]),
                                 np.arange(grid_lat[0], grid_lat[1], grid_size[1]))
    grid_x, grid_y = np.int_(GX.flatten()), np.int_(GY.flatten())
    indices = coordinates_to_indices(grid_x, grid_y, grid_anchor, grid_size)

    RSSI_RM = pd.DataFrame(data={'x': grid_x, 'y': grid_y}, index=indices)
    for name in wap_column_names:
        RSSI_RM[name] = np.nan

    # each grid point will get the average measured RSSI value for each AP
    training_data.insert(0, "grid_pnt",coordinates_to_indices(training_data["LONGITUDE"], training_data["LATITUDE"], grid_anchor, grid_size))
    training_data_gridgroups = training_data.groupby(by="grid_pnt")
    training_data_agg = training_data_gridgroups.agg({i: np.mean for i in wap_column_names})

    RSSI_RM.loc[training_data_agg.index, wap_column_names] = training_data_agg[wap_column_names]
    return RSSI_RM


def coordinates_to_indices(x, y, grid_anchor, grid_size):
    """
    turn a list of coordinates into indices
    :param x: list of x values
    :param y: list of y values
    :param grid_anchor: (x0, y0) tuple containing the anchor point of the grid
    :param grid_size: (dx,dy) tuple containing the size of the grid in x,y directions
    :return: list of tuples containing the (x_ind,y_ind) of each coordinate couple
    """
    return list(zip(*(np.int_((x-grid_anchor[0])/grid_size[0]), np.int_((y-grid_anchor[1])/grid_size[1]))))


def get_grid_edges(cur_training_data, grid_size):
    """
    get the edges of the grid specificed by training data.
    the edges are the exntent where we actually have data points + small safety window
    :param cur_training_data: DataFrame containing data coordinates
    :param grid_size: (dx,dy) tuple containing the size of the grid in x,y directions
    :return: [grid_min_lon, grid_max_lon], [grid_min_lat, grid_max_lat]
    """
    min_x = np.min(cur_training_data.LONGITUDE) - grid_size[0]
    max_x = np.max(cur_training_data.LONGITUDE) + grid_size[0]
    min_y = np.min(cur_training_data.LATITUDE) - grid_size[1]
    max_y = np.max(cur_training_data.LATITUDE) + grid_size[1]
    return [min_x, max_x], [min_y, max_y]