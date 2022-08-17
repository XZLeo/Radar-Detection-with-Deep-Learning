'''
Python file that loads Radarscenes properly 
so it can be used along with pointnet and DBSCAN later.
It does not only loads the dataset but transform it
'''
from argparse import ArgumentParser
from pathlib import Path
from numpy import array, ndarray, where, zeros
import numpy as np
#from pyparsing import original_text_for
from torch.utils.data import DataLoader
from dataset import BasicDataset
from labels import ClassificationLabel
import torch
import torch.nn.functional as nnf
from PointNet.Pnet_pytorch.provider import jitter_point_cloud_noVR_dataset
from collections import Counter #For making use of the backdoor of counting the number of classes

class PNETDataset(BasicDataset):
    def __init__(self, dataset_path, snip_path, n_points, jitter_data) -> None:
        self.n_points = n_points
        self.jitter_data = jitter_data
        super().__init__(dataset_path, snip_path)
        
    def __getitem__(self, index):
        '''
        param: n_points: clips (splits) or adds (upsample using k-nearest neighbor)
                         to every snippet for  
        '''
        is_upsampled = False
        aug_flg=False
        snippet,_ = super().load_snippet(index, aug_flg)              
                      
        X = array([snippet['x_cc'], snippet['y_cc'],
                   snippet['vr_compensated'], snippet['rcs'], 
                   snippet['range_sc']])
                            
        y = snippet['label_id']  
        
        cur_n_points = X.shape[1]
        n_features = X.shape[0] #for using when upsample method is called. Not needed in upsample_one_reflection
                
        #print(Counter(y))
        if (self.jitter_data):
            X, y = jitter_point_cloud_noVR_dataset(X, y)
        #print(Counter(y))  
        
        trid = np.array(convert_track_id(snippet, cur_n_points)) 

        if cur_n_points > self.n_points:
            y_temp = np.copy(y)
            X,y = clipping(X, y, self.n_points, cur_n_points, static_label=11, axis=1)
            trid = trid_clipping(trid, y_temp, self.n_points, cur_n_points)
        elif cur_n_points < self.n_points:
            X,y = upsample_one_reflection(X, y, cur_n_points, self.n_points)
            #X,y = upsample(snippet, n_features, self.n_points)
            trid = np.pad(trid, (0, self.n_points - cur_n_points), 'constant', constant_values=(0, -2))
            is_upsampled = True
        else:
            X=X; y=y #if coincidently the same n_points
        
        X = X.T; y = y.T    
        # map labels to 5 classes
        mapped_y = []
        for element in y:
            mapped_y.append(ClassificationLabel.label_to_clabel(element).value)
        mapped_y = array(mapped_y)
        
        return X, mapped_y, is_upsampled, trid


def convert_track_id(snippet:ndarray, n_points):
    '''
    Transforms the byte type of variable gotten from track_id into
    an numpy array per snippet with integer values.
    '''
    trid = -np.ones(n_points)
    track_ids = set(snippet["track_id"])
    count = 0
    for tr_id in track_ids:
        if tr_id != b'':
            idx = where(snippet["track_id"] == tr_id)[0] # get the index of non-empty track id
            if len(idx) <= 2: # only one point with same tr_id, ignore it
                #print('only one points labeled')
                continue
            trid[idx] = count 
            count += 1      
    return trid

def get_gt_clusters_id(trid, gt_xyl):
    '''
    Get annotated clusters from a 500ms snippet
    '''
    points_in_clusters = []
    labels_in_clusters = []
    track_ids = set(trid)
    for tr_id in track_ids:
        if tr_id != -1:  # an empty snippet only have track_id b''
            idx = where(trid == tr_id)[0] # get the index of non-empty track id
            if len(idx) <= 2: # only one point with same tr_id, ignore it
                #print('only one points labeled')
                continue
                #pass
            # more than 2 pionts in a cluster
            points = zeros((2, len(idx)))
            class_label = zeros(len(idx))
            points[0, :] = gt_xyl[idx,0].reshape((1,len(idx)))
            points[1, :] = gt_xyl[idx,1].reshape((1,len(idx)))
            #class_label = gt_xyl[idx[0],2]
            class_label = gt_xyl[idx,2]
            if np.all(class_label != 5): #clusters of static????????? LEOX
                points_in_clusters.append(points)
                labels_in_clusters.append(class_label)
    return points_in_clusters, labels_in_clusters

def clipping(X, y, n_points, curr_points, static_label, axis): 
    '''
    Erase static class labeled points when the snippet exceeds the 
    selected number of points until the number is reached
    (a.k.a., point cloud splitting)
    '''
    static_indices = np.where(y == static_label)
    #random_index = np.random.randint(0, curr_points - n_points)
    size_st_idx = static_indices[0].shape[0]
    st_id = static_indices[0][(size_st_idx - (curr_points - n_points)):]
    X = np.delete(X, st_id.tolist(), axis=axis)
    y = np.delete(y, st_id.tolist())
    return X,y

def upsample(snippet, n_features, n_points):  
    '''
    Performs a nearest neigbhor random upsampling (over random points) when the 
    snippet is behind in number of the selected number of points
    '''      
    xy = array([snippet['x_cc'], snippet['y_cc'],
                snippet['vr_compensated'], snippet['rcs'], snippet['range_sc'],#LEOX
                snippet['label_id']])    
    xy = torch.tensor(xy)
    xy1 = xy.unsqueeze(0).unsqueeze(0)
    xy2 = nnf.interpolate(xy1, size=(n_features+1,n_points), mode='nearest')
    xy = torch.squeeze(torch.squeeze(xy2, 0), 0)
    y = xy[-1, :].numpy()
    X = xy[:-1, :].numpy()
    return X,y

def upsample_one_reflection(X, y, cur_n_points, n_points):
    '''
    Performs an upsampling (over one static class point) when the 
    snippet is behind in number of the selected number of points
    ''' 
    static_indices = np.where(y == 11)[0]
    num_copies = n_points - cur_n_points
    X = X.T
    for _ in range(num_copies):
        X = np.vstack((X, X[static_indices[0]]))
        y = np.hstack((y, y[static_indices[0]]))
    return X.T, y.T

def soft_clipping(X, y, n_points):
    '''
    NAIVELY, erase any class labeled points when the snippet exceeds the 
    selected number of points until the number is reached
    (i.e., it might erase dynamic classes frrom dataset) (Not recommended
    but useful for certain experiments)
    '''
    X = X[:,0:n_points]
    y = y[0:n_points]
    return X,y

def trid_clipping(trid, y, n_points, curr_points):
    '''
    Erase the feature of track_id (for use as ground truth clusters in DBSCAN) 
    in the same static class labeled points clipped in the clipping method  
    when the snippet exceeds the selected number of points until the number 
    is reached
    '''
    #introuce condition of static class
    static_indices = np.where(y == 11)
    #random_index = np.random.randint(0, curr_points - n_points)
    size_st_idx = static_indices[0].shape[0]
    st_id = static_indices[0][(size_st_idx - (curr_points - n_points)):]
    trid = np.delete(trid, st_id.tolist())
    return trid
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-d',
        '--dataset_path',
        type=Path,
        default= "/home/s0001519/RadarProj/RadarScenes/train",
        help='file path to the train, test or validation set'
    )
    parser.add_argument(
        '-s',
        '--snippet_path',
        type=Path,
        default="/home/s0001519/RadarProj/venv/zenseact_repo/thesisdlradardetection/static/train.txt",
        help='file path of the start indexes'
    )
    parser.add_argument(
        "--n_points", 
        type=int,
        default=3097,
        help="Point Number per snippet [default: 3097]"
    )
    arguments = parser.parse_args()
    dataset = PNETDataset(arguments.dataset_path, arguments.snippet_path, arguments.n_points)
    generator = DataLoader(dataset)
    max_epochs = 1
    for epoch in range(max_epochs):
        # Training
        for local_batch, local_labels in generator:
            print(local_batch.shape)
            print(local_labels.shape)
            break
    