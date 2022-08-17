'''
Generate snippets on the fly, transform them into Grid mappings,
Max Doppler map, min Doppler map and RCS map and also extractr 
bounding boxes
'''
from argparse import ArgumentParser
from numpy import array, squeeze, float32
from pathlib import Path
from torch.utils.data import DataLoader
import torch

from dataset import BasicDataset
from yolo.preprocessing.boxes import get_AABB_snippet, get_OBB_snippet
from yolo.preprocessing.gridMap import GridMap
from yolo.preprocessing.coordinateTransfer import coor_transfer
#from yolo.visualize import visualize_cloud #only for code tresting


class AABBDataset(BasicDataset):
    def __init__(self, dataset_path, snip_path, blurry=False,
                 skew=False, num_cell=608, aug_flg:bool =False,
                 rot_flg:int = 0, mirr_flg='') -> None:
        '''
        aug_flg: augmentation flag, set False if don't want
        rot_flg: rotation flag: 0 for pass, 1 for rotate30, 2 for rotate randomly
        mirr_flg: flg for mirror symetry
        '''  
        super().__init__(dataset_path, snip_path, aug_flg)
        self.gridmaps = None
        self.boundingboxes = [] 
        self.aug_flg = aug_flg
        self.rot_flg = rot_flg
        self.mirr_flg = mirr_flg
        self.blurry = blurry
        self.skew = skew
        self.num_cell = num_cell
        self.eval = False

    def __getitem__(self, index):
        '''
        param: boxtype: true for axis aligned boxes, false for oriented boxes
        '''
        snippet, info = super().load_snippet(index, self.aug_flg, self.rot_flg, self.mirr_flg)
        #visualize_cloud(snippet) # only for code tresting

        # generate bounding boxes
        aligned_boxes = array(get_AABB_snippet(snippet)) 
        # coordinate transfer
        tran_aligned_boxes = coor_transfer(aligned_boxes)
        if tran_aligned_boxes is None:
            # lost boxes after rotation
            # print(info)
            self.boundingboxes = torch.from_numpy(array([])) # return empty
        else:
            self.boundingboxes = torch.from_numpy(tran_aligned_boxes)
        # generate grid mappings 
        gridmap = GridMap(snippet, self.num_cell)
        gridmap.get_max_amplitude_map()
        gridmap.get_Doppler_map(mode=True, skew=self.skew)
        gridmap.get_Doppler_map(mode=False, skew=self.skew)
        if self.blurry:
            gridmap.filter_blurry() # the user should be able to decide
            self.gridmaps = squeeze(array([gridmap.blurry_amp_map, 
                        gridmap.blurry_max_doppler_map, gridmap.blurry_min_doppler_map], dtype=float32))
        else:
            self.gridmaps = squeeze(array([gridmap.amp_map, 
                       gridmap.max_doppler_map, gridmap.min_doppler_map], dtype=float32))
        self.gridmaps = torch.from_numpy(self.gridmaps)
        # evaluation mode
        if self.eval:
            return snippet, self.gridmaps, self.boundingboxes    
        return None, self.gridmaps, self.boundingboxes

    def collate_fn(self, batch):
        # add batch index
        batch = [data for data in batch if data is not None]
        paths, imgs, bb_targets = list(zip(*batch))
        for i, boxes in enumerate(bb_targets):
            if boxes.shape[0] != 0: # skip empty boxes
                boxes[:, 0] = i 
        #restore torch tensor
        bb_targets = torch.cat(bb_targets, 0) 
        imgs = torch.stack([img for img in imgs])
        return paths, imgs, bb_targets

    def set_eval(self):
        '''
        set mode for validation or test
        '''
        self.eval = True
    

class OBBDataset(BasicDataset):
    def __init__(self, dataset_path, snip_path) -> None:
        super().__init__(dataset_path, snip_path)
        self.gridmaps = None
        self.boundingboxes = [] 

    def __getitem__(self, index):
        '''
        param: boxtype: true for axis aligned boxes, false for oriented boxes
        '''
        snippet = super().load_snippet(index)
        # generate grid mappings 
        map = GridMap(snippet)
        map.get_max_amplitude_map()
        map.get_Doppler_map(True)
        map.get_Doppler_map(False)
        map.filter_blurry()
        self.gridmaps = array([map.blurry_amp_map, 
                        map.blurry_max_doppler_map, map.blurry_min_doppler_map])
        # generate bounding boxes
        oriented_boxes, list_corners, list_ax = get_OBB_snippet(snippet)
        self.boundingboxes = array(oriented_boxes)
        return self.gridmaps, self.boundingboxes


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-d',
        '--dataset_path',
        type=Path,
        default= "/home/s0001516/thesis/dataset/RadarScenes/train",
        help='file path to the train, test or validation set'
    )
    parser.add_argument(
        '-s',
        '--snippet_path',
        type=Path,
        default="/home/s0001516/thesis/src/thesisdlradardetection/static/train_short.txt",
        help='file path of the start indexes'
    )
    parser.add_argument(
        '-a',
        '--augmentation',
        action='store_true',
        help='True for rotate the snippet for data augmentation'
    )
    parser.add_argument(
        '-r',
        '--rotate',
        type=int,
        default=0,
        help='0 do nothing, 1 for rotate the snippet for 30, 2 for rotate randomly'
    )
    parser.add_argument(
        '-m',
        '--mirror',
        type=str,
        help='flip a snippet horizontally or vertically or both'
    )
    parser.add_argument(
        "--blurry", 
        action='store_true',
        help="add blurry filter while generating gridMaps")
    parser.add_argument(
        "--skew", 
        action='store_true', 
        help="skew speed distribution while generating gridMaps")
    arguments = parser.parse_args()
    dataset = AABBDataset(arguments.dataset_path, arguments.snippet_path, 
                        arguments.blurry, arguments.skew, aug_flg=arguments.augmentation,
                        rot_flg=arguments.rotate, mirr_flg=arguments.mirror)
    # validation mode
    dataset.set_eval()
    generator = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    max_epochs = 1
    for epoch in range(max_epochs):
        for snip, local_batch, local_labels in generator:
            print(len(snip))
            print(snip[0][0])
            print(local_batch.shape)
            print(local_labels.shape)
            pass
        
