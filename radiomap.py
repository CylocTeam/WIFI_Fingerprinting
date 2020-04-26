import numpy as np
import pandas as pd
import scipy.spatial.distance as pydist


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
    def __init__(self, training_data, grid_size=(1,1), building=np.nan, floor=np.nan, interpolation=None):
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
        self.radiomaps = get_radiomap_dict(training_data, (grid_anchor, map_size, grid_size), functions=[np.nanmean, np.nanstd])
        self.building = building
        self.floor = floor

        if interpolation is not None:
            self.interpolate(method=interpolation)

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
                radio_list.append(self.radiomaps[ap]["RSSI_map"]['nanmean'])
                radio_ap_list.append(self.radiomaps[ap]["RSSI_map"]['nanstd'])
            else:
                radio_list.append(np.full(self.map_size, np.nan))
                radio_ap_list.append(np.full(self.map_size, np.nan))

        return np.dstack(radio_list), np.dstack(radio_ap_list)

    def interpolate(self, method='kernel'):
        """ method interpolates each rm for each AP using one of methods below:
            kernel - gaussian / gamma
            linear
        """
        for rm in self.radiomaps.items():
            rm_content = rm[1]
            # validate sustainable data
            if pydist.norm(np.diag(rm_content['cov'][0])) == 0: # single samples along all cells

                print('bulding: {bld}, floor: {flr}, AP={ap} '
                      'contains single point - Will not be interpolated'.format(bld=self.building, flr=self.floor, ap=rm[0]))
                continue

            if rm_content['irev_loc_vec'].shape[0] == 0:  # no cells to interpolate
                print('bulding: {bld}, floor: {flr}, AP={ap} '
                      'RSSI map is full - no interpolation made'.format(bld=self.building, flr=self.floor, ap=rm[0]))
                continue

            rssi_map = rm_content['RSSI_map']['nanmean']
            is_relev = np.isnan(rssi_map)
            y1 = rssi_map[~is_relev]
            k_s1s1 = rm_content['cov'][0]
            k_s1s2 = eval_kernel(rm_content['loc_vec'], rm_content['irev_loc_vec'], rm_content['cov'][1])
            # k_s2s2 = eval_kernel(rm_content['irev_loc_vec'], rm_content['irev_loc_vec'], rm_content['cov'][1])

            # try:
            mid_term = np.dot(k_s1s2.transpose(), (np.linalg.inv(k_s1s1)))
            y2 = np.dot(mid_term, y1)  # mean
            # y2_covmat = np.linalg.eig(k_s2s2 - mid_term.dot(k_s1s2))[0]

            # except:
            #     print('bulding: {bld}, floor: {flr}, AP={ap}\n'.format(bld=self.building, flr=self.floor, ap=rm[0]))
            #     continue
            try:
                ind_x, ind_y = list(zip(*rm_content['irev_loc_vec'].index))
                rssi_map[ind_y, ind_x] = y2
            except:
                print('bulding: {bld}, floor: {flr}, AP={ap} '
                      'unpacking error'.format(bld=self.building, flr=self.floor, ap=rm[0]))
                continue

            ## todo - if y2_conf too high for specific cell - use other interpolation method


def create_radiomap_objects(training_data, grid_size=(1, 1), interpolation=None):
    unique_areas = training_data[["BUILDINGID", "FLOOR"]].drop_duplicates()
    rm_per_area = {}
    for area in unique_areas.values:
        building_id, floor = area
        cur_training_data = training_data.loc[(training_data[["BUILDINGID", "FLOOR"]] == area).all(axis=1)]
        cur_rm = RadioMap(cur_training_data, grid_size=grid_size, building=building_id, floor=floor, interpolation=interpolation)
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
    if (functions is None) or len(functions) == 1:
        functions = [np.nanmean, np.nanstd]

    # setting up necessary variables
    wap_column_names = training_data.filter(regex=("WAP\d*")).columns
    relev_aps = wap_column_names[(~np.isnan(training_data[wap_column_names])).any(axis=0)]
    grid_anchor, rm_size, grid_size = grid

    # find average rssi values for each AP in each point in space
    trn_indices = coordinates_to_indices(training_data["LONGITUDE"], training_data["LATITUDE"], grid_anchor, grid_size)
    training_data.insert(0, "grid_pnt", trn_indices)
    training_data_gridgroups = training_data.groupby(by="grid_pnt")
    training_data_agg = training_data_gridgroups.agg({i: functions for i in relev_aps})
    training_data_agg_loc = training_data_gridgroups.agg({i: np.nanmean for i in ['LONGITUDE', 'LATITUDE']})

    # create a RM for each AP and insert it into the dictionary
    radiomap_dict = {}
    for cur_ap in relev_aps:
        clm = training_data_agg[cur_ap].columns  # column names might be different than func if func isn't a string
        cur_rm = np.full(rm_size, np.nan)
        radiomap_dict[cur_ap] = {'RSSI_map': {}, 'loc_vec': [], 'irev_loc_vec': [], 'cov': []}

        for ii in range(0, len(functions)):
            fnc = clm[ii]
            is_relev = ~np.isnan(training_data_agg[cur_ap, fnc])
            relev_agg = training_data_agg[is_relev]

            if len(relev_agg) == 0:
                continue

            relev_agg_fnc = relev_agg[cur_ap, fnc].tolist()
            if fnc == "nanstd":
                relev_std = relev_agg_fnc

            ind_x, ind_y = list(zip(*relev_agg.index))
            cur_rm[ind_y, ind_x] = relev_agg_fnc
            radiomap_dict[cur_ap]['RSSI_map'][fnc] = cur_rm

        relev_agg_loc = training_data_agg_loc[is_relev]
        irrelev_agg_loc = training_data_agg_loc[~is_relev]

        radiomap_dict[cur_ap]['loc_vec']      = relev_agg_loc    # [lon, lat]
        radiomap_dict[cur_ap]['irev_loc_vec'] = irrelev_agg_loc  # [lon, lat]
        radiomap_dict[cur_ap]['cov']      = train_kernel(relev_agg_loc, relev_agg_loc, train_std=relev_std)

    return radiomap_dict


def train_kernel(loc_a, loc_b, train_std=None):

    if train_std is None:
        sigma_f_sq = 1  # flactuation factor
        l_sq = 1  # length scale factor
    else:
        # train
        sigma_f_sq = np.mean(train_std)
        l_sq = 34.
        hyperparams = {'sigma_f_sq': sigma_f_sq, 'l_sq': l_sq}

    k = eval_kernel(loc_a, loc_b, hyperparams)

    # retain original STDs
    k = k - np.diag(np.diag(k)) + np.diag(train_std)

    return k, hyperparams


def eval_kernel(loc_a, loc_b, hyper_params={}):

    if hyper_params == {}:
        sigma_f_sq = 1  # flactuation factor
        l_sq = 1        # length scale factor
    else:
        sigma_f_sq = hyper_params['sigma_f_sq']
        l_sq       = hyper_params['l_sq']

    k = sigma_f_sq * np.exp(-1/(2*l_sq) * pydist.cdist(loc_a, loc_b))

    return k
