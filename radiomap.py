import numpy as np


# class RadioCell:
#     """
#     RadioCell is an object that contains all the data of a specific radiomap cell
#     """
#     def __init__(self, df, id=None, loc=None, size=None):
#         rssi_d, rssi_cov_d = df.mean(), df.cov()
#         self.rssi_list = rssi_d # key = ap, val = mean rssi measurement
#         self.rssi_cov = rssi_cov_d # correlation between rssi measurements in cell
#         self.cell_id = id # (i,j) index of cell in map
#         self.cell_loc = loc # (x,y) of cell anchor
#         self.cell_size = size # total size of cell
#         self.num_of_measurements = len(df)


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
        rm, rm_var = get_radiomap_dict(training_data, (grid_anchor, map_size, grid_size), functions=[np.nanmean, np.nanvar])
        self.radiomaps = rm
        self.radiomap_var = rm_var
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
        radio_list, radio_ap_list = [], []
        for ap in ap_list:
            if ap in keys:
                radio_list.append(self.radiomaps[ap])
                radio_ap_list.append(self.radiomap_var[ap])
            else:
                radio_list.append(np.full(self.map_size, np.nan))
                radio_ap_list.append(np.full(self.map_size, np.nan))
        return np.dstack(radio_list), np.dstack(radio_ap_list)


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


def get_radiomap_dict(training_data, grid, functions=None):
    """
    Create radiomap in a specific extent, using training_data
    :param training_data: DataFrame containing data to train RM with
    :param grid: tuple (grid_lon, grid_lat, grid_size) = ([grid_min_lon, grid_max_lon], [grid_min_lat, grid_max_lat], (dx,dy))
    :return: Dataframe containing RM in the specified extent
    """
    if functions is None:
        functions = ["mean"]

    # setting up necessary variables
    wap_column_names = training_data.filter(regex=("WAP\d*")).columns
    relev_aps = wap_column_names[(~np.isnan(training_data[wap_column_names])).any(axis=0)]
    dct_list = tuple([{} for ii in range(0, len(functions))])
    grid_anchor, rm_size, grid_size = grid

    # find average rssi values for each AP in each point in space
    trn_indices = coordinates_to_indices(training_data["LONGITUDE"], training_data["LATITUDE"], grid_anchor, grid_size)
    training_data.insert(0, "grid_pnt", trn_indices)
    training_data_gridgroups = training_data.groupby(by="grid_pnt")
    training_data_agg = training_data_gridgroups.agg({i: functions for i in relev_aps})

    # create a RM for each AP and insert it into the dictionary
    for cur_ap in relev_aps:
        clm = training_data_agg[cur_ap].columns # column names might be different than func if func isn't a string
        for ii in range(0, len(functions)):
            fnc = clm[ii]
            cur_rm = np.full(rm_size, np.nan)
            relev_agg = training_data_agg[~np.isnan(training_data_agg[cur_ap, fnc])]

            if len(relev_agg) == 0:
                continue
            ind_x, ind_y = list(zip(*relev_agg.index))
            cur_rm[ind_y, ind_x] = relev_agg[cur_ap, fnc].tolist()
            dct_list[ii][cur_ap] = cur_rm

    return dct_list

