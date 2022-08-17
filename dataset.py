'''
A basic class for the both models
'''
from argparse import ArgumentParser
from os import listdir
from os.path import join, isdir
import logging
from numpy import array, sum, where
from pathlib import Path
from radar_scenes.sequence import Sequence
from torch.utils.data import Dataset, DataLoader

from snippet import clip
from labels import ClassificationLabel
from frame import get_frames, get_timestamps
from yolo.utils.augmentations import rotate, rotate30, mirror 

#from yolo.visualize import visualize_cloud #only for code tresting

import  matplotlib.pyplot as plt

"""Logger for printing."""
LOG = logging.getLogger(__name__)

class BasicDataset(Dataset):
    def __init__(self, dataset_path:Path, snip_path:Path,
                 aug_flg:bool =False, rot_flg:int = 0, mirr_flg='') -> None:
        '''
        dataset_path: path to all the folders of sequences
        snip_path: path to the record of snippet
        aug_flg: flag for data augmentation
        '''
        super().__init__()
        # flags for data augmentation
        self.aug_flg = aug_flg
        self.rot_flg = rot_flg
        self.mirr_flg = mirr_flg
        # aid load snippet
        self.list_sequence = []
        self.load_sequence(dataset_path)
        self.num_snip_seq = []
        self.list_start_idx = []
        self.list_num_future_frames = []  # number of snippter in each sequence
        self.load_snip_dix(snip_path)

    def __len__(self):
        'total number of snippets'
        return int(sum(self.num_snip_seq))

    def __getitem__(self, index):
        if index % 100 == 0:
            print('loading {}th/{} snippet'.format(index, sum(self.num_snip_seq)))
        snippet, _ = self.load_snippet(index, self.aug_flg, self.rot_flg, self.mirr_flg)
        # split feature and labels
        X = array([snippet['x_cc'], snippet['y_cc'],
            snippet['vr_compensated'], snippet['rcs']])
        y = snippet['label_id']
        # map labels to 5 classes
        mapped_y = []
        for element in y:
            mapped_y.append(ClassificationLabel.label_to_clabel(element).value)
        mapped_y = array(mapped_y)
        return X, mapped_y

    def load_snippet(self, index, aug_flag=False, rot_flg:int = 0, mirr_flg=''):
        '''
        generate one snippet
        index: index of snippet, e.g., the first snippet in sequence 1's index is 1
        aug_flg: augmentation flag, set False if don't want
        rot_flg: rotation flag: 0 for pass, 1 for rotate30, 2 for rotate randomly
        mirr_flg: flg for mirror symetry
        '''
        # find the sequence and start index (of the snippet) in the sequence
        sum_num_snip = 0
        for idx_seq, num_snip in enumerate(self.num_snip_seq):
            sum_num_snip += num_snip
            if sum_num_snip >= index+1: # index start from 0
                row = idx_seq # which sequence the snippet belongs to
                break
        col = int(index+1 - sum(self.num_snip_seq[0:row]) - 1)
        # load the start index in that sequence and number of future frames
        start_idx = self.list_start_idx[row][col]
        num_future_frames = self.list_num_future_frames[row][col]
        # Load snippet
        cur_seq = self.list_sequence[row] 
        timestamps = get_timestamps(cur_seq)
        snippet = get_frames(cur_seq, start_idx, 
                            timestamps, n_next_frames=num_future_frames)
        info = 'index {} row {} colum {} start index {} {} '.format(
                index, row, col, start_idx, cur_seq.sequence_name)
        #print(info)
        # visualize_cloud(snippet) # for test

        # rotation for data augmentation
        if aug_flag:
            if rot_flg == 1:
                snippet, theta = rotate30(snippet)
                info += 'rotation {}'.format(theta) # logger
            elif rot_flg == 2:
                snippet, theta = rotate(snippet) 
                info += 'rotation {}'.format(theta) # logger
            elif rot_flg == 0:
                pass
            else:
                print('Wrong rotation flag')
            snippet = mirror(snippet, mirr_flg)
            info += '{} mirror symetry'.format(mirr_flg)
        else:
            pass
         
        # print(info)
        # visualize_cloud(snippet) # only for code tresting    
        # clip the scene that is out of 100m*100m
        clip_snip = clip(snippet)
        return clip_snip, info

    def load_sequence(self, dataset_path):
        # load all sequences
        list_sequences = listdir(dataset_path)
        # remove files, only leave folders
        for nm_sequence in list_sequences:
            path_sequence = join(dataset_path, nm_sequence)
            if not isdir(path_sequence):
                list_sequences.remove(nm_sequence)

        for nm_sequence in list_sequences:
            path_sequence = join(dataset_path, nm_sequence)
            if isdir(path_sequence):
                # Define the *.json file from which data should be loaded
                filename = join(path_sequence, "scenes.json")
                sequence = Sequence.from_json(filename)
                self.list_sequence.append(sequence)

    def load_snip_dix(self, snip_path):
         # load the start index and number of future frames of each snippet
        try:
            with open(snip_path, "r") as f:
                line = f.readline()
                while line:
                    a = line.split()
                    start_idx = []
                    num_future_frames = []
                    for pair in a:
                        b = list(map(int, pair.split(',')))
                        start_idx.append(b[0])
                        num_future_frames.append(b[1])
                    self.list_start_idx.append(start_idx)
                    self.list_num_future_frames.append(num_future_frames)
                    self.num_snip_seq.append(len(start_idx))
                    line = f.readline()
        except IOError:
            print("Unable to open input file {}".format(snip_path))
        
        
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
    arguments = parser.parse_args()
    dataset = BasicDataset(arguments.dataset_path, arguments.snippet_path, 
                        arguments.augmentation, arguments.rotate, arguments.mirror)
    generator = DataLoader(dataset)
    max_epochs = 1
    for epoch in range(max_epochs):
        # Training
        for local_batch, local_labels in generator:
            print(local_batch.shape)
            print(local_labels.shape)