'''
Generate #grid mappings based on Nicolas's paper, can be replaced by another approach later
'''
from copy import deepcopy
from xmlrpc.client import Boolean
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, imshow, colorbar, title #grid
from numpy import array, zeros, floor, delete, arange, hstack, linspace, where, float32, abs, sign, poly1d, polyfit, clip, isclose
from os import listdir, path
from scipy.interpolate import interp1d
from radar_scenes.sequence import Sequence

from frame import get_frames, get_timestamps


MAP_SIZE = 100 # meters
# skew function, run when imported, speed up the skew_vr()
x = arange(0, 3.00, 0.25)
y = hstack((array([0, 3/4, 15/16, 31/32]), linspace(1, 2.75, 8)))
g = interp1d(x, y, kind="cubic")
# skew function from Nicolas
sx = array([0, 10, 20, 27.5, 40])
sy = array([0, 0.7, 0.9, 0.95, 1])
poly = poly1d(polyfit(sx, sy, 4))  # degree of freedom
lin = poly1d(polyfit(sx[-2:], sy[-2:], 1))


class GridMap:
    '''
    get #grid mappings with three channels, the user can choose to use blurring filter or not
    param: frame: a frame of one or several sensor scans accumulated together
        filter: true for using blurring filter 
    return: maps: #grid maps with three channels, amplitude, max Dopplker, min Doppler
    '''
    def __init__(self, frame, num_cell=608) -> None:
        self.num_cell = num_cell
        self.width_cell = MAP_SIZE / self.num_cell
        self.frame = frame
        self.amp_map = zeros((self.num_cell, self.num_cell), dtype=float32)
        self.max_doppler_map = zeros((self.num_cell, self.num_cell), dtype=float32)
        self.min_doppler_map = zeros((self.num_cell, self.num_cell), dtype=float32)
        self.count_map = zeros((self.num_cell, self.num_cell), dtype=float32) # number of points in each cell
        self.blurry_amp_map = zeros((self.num_cell, self.num_cell), dtype=float32)
        self.blurry_max_doppler_map = zeros((self.num_cell, self.num_cell), dtype=float32)
        self.blurry_min_doppler_map = zeros((self.num_cell, self.num_cell), dtype=float32)
    

    def get_max_amplitude_map(self):
        '''
        put a snippet of accumulated points into the #grid map. 
        Each cell's value is the signed maximum amplitude.
        '''
        for idx, x in enumerate(self.frame["x_cc"]):
            y = self.frame["y_cc"][idx]
            amp = self.frame["rcs"][idx]
            if (0 <= x <= MAP_SIZE) and (-MAP_SIZE/2 <= y <=MAP_SIZE/2):
                row = int(self.num_cell - floor(x/self.width_cell) - 1)
                # to make the origin of ego-vehicle coordinate at the bottom
                col = int(self.num_cell - floor((y+MAP_SIZE/2)/self.width_cell) - 1)
                self.count_map[row, col] += 1
                if self.amp_map[row, col] < amp:
                    self.amp_map[row, col] =  amp 
        self.blurry_amp_map = deepcopy(self.amp_map)


    def get_Doppler_map(self, mode:Boolean, skew:Boolean = False):
        '''
        param: mode: True for maximum, False for minimum
        '''
        doppler_map = zeros((self.num_cell, self.num_cell))
        for idx, x in enumerate(self.frame["x_cc"]):
            y = self.frame["y_cc"][idx]
            vr = self.frame["vr_compensated"][idx]
            if (0 <= x <= MAP_SIZE) and (-MAP_SIZE/2 <= y <=MAP_SIZE/2):
                # transfer ego-vehicle coordinate to the image coordinate
                row = int(self.num_cell - floor(x/self.width_cell) - 1)
                col = int(self.num_cell - floor((y+MAP_SIZE/2)/self.width_cell) - 1)
                state = doppler_map[row, col] < vr
                if (state and mode) or ((not state) and (not mode)):
                    # Xnor, equivalence gate, draw the truth table
                    doppler_map[row, col] =  vr 
        if mode:
            if skew:
                # spread velocity distribution
                self.max_doppler_map = skew_vr_origin(doppler_map)
            else:
                self.max_doppler_map = doppler_map
        else:
            if skew:
                self.min_doppler_map = skew_vr_origin(doppler_map)
            else:
                self.min_doppler_map = doppler_map    
        self.blurry_max_doppler_map = deepcopy(self.max_doppler_map)
        self.blurry_min_doppler_map = deepcopy(self.min_doppler_map)
        

    def filter_blurry(self):
        nonempty_X, nonempty_Y = where(self.count_map)
        for idx in range(len(nonempty_X)):
            row = nonempty_X[idx]
            col = nonempty_Y[idx]
            neighbours = self.find_neighbours(row, col)
            # copy values following the propagation scheme
            for coor in neighbours:
            # ignore neighour cells that are not empty
                if self.count_map[tuple(coor)] == 0:
                    self.blurry_amp_map[tuple(coor)] = self.amp_map[row, col]
                    self.blurry_max_doppler_map[tuple(coor)] = self.max_doppler_map[row, col]
                    self.blurry_min_doppler_map[tuple(coor)] = self.min_doppler_map[row, col]
                    # here propagation zone's overlap is not considered, can be added later


    def find_neighbours(self, row, col):
        '''
        Given the origin and Euclidean distance
        return a list of <row, col> for the neighbours with in the given Euclidean distance,
        considering the size of the #grid map
        '''
        neighbours = []
        if  self.count_map[row, col] >1:
            neighbours += [(row, col+1), (row, col-1), (row+1, col), (row-1, col)]
        if self.count_map[row, col] > 3:
            neighbours += [(row-1, col-1), (row+1, col-1), (row-1, col+1), (row+1, col+1)]
        if self.count_map[row, col] > 5:
            neighbours += [(row-2, col), (row+2, col), (row, col-2), (row, col+2)]
        if self.count_map[row, col] > 7:
            neighbours += [(row-1, col-2), (row+1, col-2), (row-1, col+2), (row+1, col+2),
                           (row-2, col-1), (row+2, col-1), (row-2, col+1), (row+2, col+1)]
        if self.count_map[row, col] > 10:
            neighbours += [(row, col-3), (row, col+3), (row-3, col), (row+3, col)]
        
        # exclude neighbour cells outside the #grid
        if neighbours != []:
            neighbours = array(neighbours)
            idx = where(neighbours[:, 1]>self.num_cell-1)
            neighbours = delete(neighbours, idx, axis=0)
            idx = where(neighbours[:, 0]>self.num_cell-1)
            neighbours = delete(neighbours, idx, axis=0)
            idx = where(neighbours[:, 1]<0)
            neighbours = delete(neighbours, idx, axis=0)
            idx = where(neighbours[:, 0]<0)
            neighbours = delete(neighbours, idx, axis=0)
        return neighbours


    def show_heatmap(self, mode, blury:bool=False):
        '''
        visualize the #gridmap, for deubgging
        '''
        major_ticks_top = linspace(0,self.num_cell, 11)
        plt.figure(figsize=(10,10))

        if not blury:
            if mode == 'amp':
                imshow(self.amp_map, cmap='hot', interpolation='nearest') # "Blues"
                plt.gca().set_xticks(major_ticks_top)
                plt.gca().set_yticks(major_ticks_top)
                #grid()
                title('Amplitude Map', fontsize=15)
                colorbar() 
                plt.show()
            elif mode == 'max':
                imshow(self.max_doppler_map, cmap='hot', interpolation='nearest')
                plt.gca().set_xticks(major_ticks_top)
                plt.gca().set_yticks(major_ticks_top)
                #grid()
                title('Max Doppler Map', fontsize=15)
                colorbar() 
                plt.show()
            elif mode == 'min':
                imshow(abs(self.min_doppler_map), cmap='hot', interpolation='nearest')
                plt.gca().set_xticks(major_ticks_top)
                plt.gca().set_yticks(major_ticks_top)
                #grid()
                title('Min Doppler Map', fontsize=15)
                colorbar() 
                plt.show()
            else:
                print('Wrong mode')
        else:
            if mode == 'amp':
                imshow(self.blurry_amp_map, cmap='hot', interpolation='nearest')
                plt.gca().set_xticks(major_ticks_top)
                plt.gca().set_yticks(major_ticks_top)
                #grid()
                title('Blurry Amplitude Map')
                colorbar() 
                plt.show()
            elif mode == 'max':
                imshow(self.blurry_max_doppler_map, cmap='hot', interpolation='nearest')
                plt.gca().set_xticks(major_ticks_top)
                plt.gca().set_yticks(major_ticks_top)
                #grid()
                title('Blurry Max Doppler Map')
                colorbar() 
                plt.show()
            elif mode == 'min':
                imshow(abs(self.blurry_min_doppler_map), cmap='hot', interpolation='nearest')
                plt.gca().set_xticks(major_ticks_top)
                plt.gca().set_yticks(major_ticks_top)
                #grid()
                title('Blurry Min Doppler Map')
                colorbar() 
                plt.show()                
            else:
                print('Wrong mode')

            
def skew_vr(vr):
    '''
    Using the depicted polynomial scaling function, the distribution is widened 
    in order to ease the feature extraction process

    in fig 7, a fourth order polynomial is used

    use supporting points and cubic interpolation to get the skew function, the value 
    of supporting points can be changed later. See the plot in statistic.ipynb
    '''
    new_vr =  sign(vr) * 40 * g(abs(vr)/40) # g is glibal variable, a function, speed up
    return new_vr          


def skew_vr_origin(im):
    neg_msk = sign(im) # record negative veloctiy
    im = abs(im)
    poly_msk = im < 27.5
    lin_msk = im >= 27.5
    im[poly_msk] = poly(im[poly_msk])
    im[lin_msk] = lin(im[lin_msk])

    im = clip(im, a_min=0.0, a_max=1.0, out=im)
    im[isclose(im, 0)] = 0
    return 40*im*neg_msk 


if __name__ == '__main__':
    # extract a frame, i.e., 4 continuous scenes from the start time, for DBSCAN
    path_to_dataset = "/home/s0001516/thesis/dataset/RadarScenes"
    # list all the folders
    list_sequences = listdir(path_to_dataset)
    # Define the *.json file from which data should be loaded
    filename = path.join(path_to_dataset, "train", "sequence_109", "scenes.json")
    sequence = Sequence.from_json(filename)
    timestamps = get_timestamps(sequence)
    cur_idx = 0
    radar_data = get_frames(sequence, cur_idx, timestamps, n_next_frames=28)
    map = gridMap(radar_data)
    map.get_max_amplitude_map()
    map.show_heatmap('amp', False)
    map.get_Doppler_map(True, skew=True)
    map.show_heatmap('max', False)
    map.get_Doppler_map(False, skew=True)
    map.show_heatmap('min', False)
    map.filter_blurry()
    map.show_heatmap('amp', True)