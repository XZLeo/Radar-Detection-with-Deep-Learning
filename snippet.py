'''
Get the start index of each 500ms snippet in a fashion of non overlapping, write to a file
avoid snippet that has no labled points
each snippet's start index and end index in its sequence is put to the txt file, 
a row in the txt file belongs to the same sequence, e.g,
start_snip1,num_future_frames1 start_snip2,num_future_frames2
'''
from argparse import ArgumentParser
from ast import arguments
import os
from pathlib import Path
from numpy import array, ndarray, where, any
import tqdm

from radar_scenes.sequence import Sequence
from frame import get_frames, get_timestamps

START_IDX = 0
SNIPPET_LENGTH = 5e5 # micro second

def is_zero_empty(snippet:ndarray) -> bool: 
    '''
    check if the snippet contains no labeled point
    '''
    track_ids = list(set(snippet["track_id"])) # set to remove repeated elemenet, then use list for index
    if len(track_ids) == 1:
        if track_ids[0] == b'':
            return True
    return False

def is_n_empty(degree, snippet:ndarray):
    '''
    <n-degree empty> means one of the clusters in that snippet contains points <= n
    delete that snippet
    '''
    flag = False
    track_ids = list(set(snippet["track_id"]))
    for tr_id in track_ids:
        if tr_id != b'':  # an empty snippet only have track_id b''
            idx = where(snippet["track_id"] == tr_id)[0] # get the index of non-empty track id
            if len(idx) < degree:
                flag = True 
    return flag

def contains_other(snippet:ndarray):
    '''
    check if the snippet contains "ANIMAL" (9) and "OTHER" (10) labels
    '''
    flag = False
    animal = where(snippet["label_id"] == 9)[0]
    other = where(snippet["label_id"] == 10)[0]
    flag = any(animal) or any(other) # true when it contains these label
    return flag

def clip(snippet:ndarray):
    '''
    remove points that not in -50<y_cc<50, 0<x_cc<100m
    '''
    idx = where(snippet["y_cc"]>=-50)[0] 
    clip_snip = snippet[idx]
    idx = where(clip_snip["y_cc"]<=50)[0]
    clip_snip = clip_snip[idx]
    idx = where(clip_snip["x_cc"]>=0)[0]
    clip_snip = clip_snip[idx] 
    idx = where(clip_snip["x_cc"]<=100)[0]
    clip_snip = clip_snip[idx]
    return clip_snip


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-d',
        '--dataset',
        type=Path,
        #default="/home/s0001516/thesis/dataset/RadarScenes/",
        default="/home/s0001519/RadarProj/RadarScenes/",
        help='the root path of the dataset'
    )
    parser.add_argument(
        '-l',
        '--length',
        type=float,
        default=5e5,
        help='Acummulation time for frames'
    )
    parser.add_argument(
        '-t',
        '--type',
        type=Path,
        default="train",
        help='train, test or validation'
    )
    arguments = parser.parse_args()
    
    path_to_dataset = os.path.join(arguments.dataset, arguments.type)
    # list all the folders
    list_sequences = os.listdir(path_to_dataset)
    # remove files, only leave folders
    for nm_sequence in list_sequences:
        path_sequence = os.path.join(path_to_dataset, nm_sequence)
        if not os.path.isdir(path_sequence):
            list_sequences.remove(nm_sequence)
    
    # record all the start idx and end idx of the snippet in its own sequence
    list_start_idx = [[] for x in range(len(list_sequences))]
    list_num_future_frames = [[] for x in range(len(list_sequences))]
    # iterate over all sequences
    for seq_idx, nm_sequence in tqdm.tqdm(enumerate(list_sequences)):
        path_sequence = os.path.join(path_to_dataset, nm_sequence)
        if os.path.isdir(path_sequence):
            # Define the *.json file from which data should be loaded
            filename = os.path.join(path_sequence, "scenes.json")
            sequence = Sequence.from_json(filename)
            timestamps = get_timestamps(sequence)
            intervals = list(array(timestamps[1:-1])-array(timestamps[0:-2])) # time invertervals betweens every two scans
            sum = 0
            start_idx = 0
            for idx_interval, interval in enumerate(intervals):
                if sum <= arguments.length:
                    sum += interval
                else:
                    num_future_frames = idx_interval - start_idx
                    # check empty snippet
                    snippet = get_frames(sequence, start_idx, 
                            timestamps, n_next_frames=num_future_frames)
                    # clip snippet
                    snippet = clip(snippet)
                    if (not is_zero_empty(snippet)) and (not is_n_empty(3, snippet)) and (not contains_other(snippet)):
                        list_start_idx[seq_idx].append(start_idx)
                        list_num_future_frames[seq_idx].append(num_future_frames) 
                    # update for the next snippet 
                    start_idx = idx_interval + 1
                    sum = 0

    # write to file
    try:
        with open('static/{}.txt'.format(arguments.type), "w") as f:
            for idx_row, row_start_idx in tqdm.tqdm(enumerate(list_start_idx)):
                row_num_future_frames = list_num_future_frames[idx_row]
                for idx_col, start_idx in enumerate(row_start_idx):
                    num_future_frames = row_num_future_frames[idx_col]
                    f.write('{},{} '.format(start_idx, num_future_frames))
                f.write('\n')
    except IOError:
        print("Unable to open input file")