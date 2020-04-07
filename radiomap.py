import numpy as np
import pandas as pd


class RadioMap:
    """
    RadioMap class holds the relevant radiomap data for Wifi fingerprinting. This includes the average RSSI values over
    a specific grid, and the area data (building and floor ids, extent, grid size etc)
    """
    def __init__(self, training_data, grid_size=(1,1), building=np.nan, floor=np.nan):
        """
        Initialize the Radiomap object with the necessary data
        :param training_data: training RSSI data (Dataframe with WAP format)
        :param grid_size: (dx,dy) size of grid cells
        :param building: ID of building
        :param floor: ID of floor in building
        """
        grid_x, grid_y = get_grid_edges(training_data, grid_size)
        self.grid_anchor = grid_anchor = (grid_x[0], grid_y[0])
        self.grid_size = grid_size
        self.extent = grid_x + grid_y
        self.map_size = map_size = [int((np.diff(grid_y)/grid_size[1])), int((np.diff(grid_x)/grid_size[0]))]
        self.radiomaps = get_radiomap_dict(training_data, (grid_anchor, map_size, grid_size))
        self.building = building
        self.floor = floor

    def get_map_ranges(self):
        """
        get the (x,y) ranges that are included within the radiomap
        :return: (xrange, yrange) numpy vectors
        """
        grid_size = self.grid_size
        exntent = self.extent

        xrange = np.arange(exntent[0], exntent[1], grid_size[0])
        yrange = np.arange(exntent[2], exntent[3], grid_size[1])
        return xrange, yrange

    def get_ap_list(self):
        """
        get a list of all APs included in the Radiomap
        :return: list of strings (keys)
        """
        return list(self.radiomaps.keys())

    def get_ap_maps_ndarray(self, ap_list):
        """
        get an narray with the RSSI data of specific AP list, in the order of the list
        :param ap_list: list of strings (AP name keys)
        :return: narray of size (map size(2d) X ap_list size) with RSSI data of APs
        """
        keys = self.radiomaps.keys()
        radio_list = []
        for ap in ap_list:
            if ap in keys:
                radio_list.append(self.radiomaps[ap])
            else:
                radio_list.append(np.full(self.map_size, np.nan))
        return np.array(radio_list)


def create_radiomap_objects(training_data, grid_size=(1, 1)):
    unique_areas = training_data[["BUILDINGID", "FLOOR"]].drop_duplicates()
    rm_per_area = {}
    for area in unique_areas.values:
        building_id, floor = area
        cur_training_data = training_data.loc[(training_data[["BUILDINGID", "FLOOR"]] == area).all(axis=1)]
        cur_rm = RadioMap(cur_training_data, grid_size=grid_size, building=building_id, floor=floor)
        rm_per_area[building_id, floor] = cur_rm
    return rm_per_area


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
    min_x = np.min(cur_training_data.LONGITUDE)
    max_x = min_x + np.ceil((np.max(cur_training_data.LONGITUDE) - min_x)/grid_size[0])*grid_size[0]
    min_y = np.min(cur_training_data.LATITUDE)
    max_y = min_y + np.ceil((np.max(cur_training_data.LATITUDE) - min_y)/grid_size[1])*grid_size[1]
    return [min_x, max_x], [min_y, max_y]


def get_radiomap_dict(training_data, grid):
    """
    Create radiomap in a specific extent, using training_data
    :param training_data: DataFrame containing data to train RM with
    :param grid: tuple (grid_lon, grid_lat, grid_size) = ([grid_min_lon, grid_max_lon], [grid_min_lat, grid_max_lat], (dx,dy))
    :return: Dataframe containing RM in the specified extent
    """
    # setting up necessary variables
    wap_column_names = training_data.filter(regex=("WAP\d*")).columns
    relev_aps = wap_column_names[(~np.isnan(training_data[wap_column_names])).any(axis=0)]
    radiomap_dict = {}
    grid_anchor, rm_size, grid_size = grid

    # find average rssi values for each AP in each point in space
    trn_indices = coordinates_to_indices(training_data["LONGITUDE"], training_data["LATITUDE"], grid_anchor, grid_size)
    training_data.insert(0, "grid_pnt", trn_indices)
    training_data_gridgroups = training_data.groupby(by="grid_pnt")
    training_data_agg = training_data_gridgroups.agg({i: np.nanmean for i in relev_aps})

    # create a RM for each AP and insert it into the dictionary
    for cur_ap in relev_aps:
        cur_rm = np.full(rm_size, np.nan)
        relev_agg = training_data_agg[~np.isnan(training_data_agg[cur_ap])]

        ind_x, ind_y = list(zip(*relev_agg.index))
        cur_rm[ind_y, ind_x] = relev_agg[cur_ap].tolist()
        radiomap_dict[cur_ap] = cur_rm

    return radiomap_dict

