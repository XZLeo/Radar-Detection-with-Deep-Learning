'''
write bboxes into a text file to aid anchor generation, run it before training
It can be run on laptop, GPU server is not neccessary
'''
from argparse import ArgumentParser
from fileinput import filename
from numpy import array, squeeze, around
from pathlib import Path
from torch.utils.data import DataLoader
import os
import tqdm

from dataset import BasicDataset
from yolo.preprocessing.boxes import get_AABB_snippet
from yolo.preprocessing.coordinateTransfer import coor_transfer



class AnchorDataset(BasicDataset):
    def __init__(self, dataset_path, snip_path) -> None:
        super().__init__(dataset_path, snip_path)
        self.gridmaps = None
        self.boundingboxes = [] 

    def __getitem__(self, index):
        '''
        generate AABB in pixel coordinate
        '''
        snippet, info = super().load_snippet(index, False)
        # generate bounding boxes
        aligned_boxes = array(get_AABB_snippet(snippet))
        # coordinate transfer
        tran_aligned_boxes = coor_transfer(aligned_boxes)
        self.boundingboxes = tran_aligned_boxes
        return self.boundingboxes, info


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-d',
        '--dataset_path',
        type=Path,
        default= "/home/s0001516/thesis/dataset/data",
        help='file path to the train, test or validation set'
    )
    parser.add_argument(
        '-s',
        '--snippet_path',
        type=Path,
        default="/home/s0001516/thesis/src/thesisdlradardetection/static/data.txt",
        help='file path of the start indexes'
    )
    parser.add_argument(
        '-t',
        '--target_path',
        type=Path,
        default="/home/s0001516/thesis/dataset/RadarScenes_GPU/annotation",
        help='file path of the start indexes'
    )
    arguments = parser.parse_args()
    if not os.path.isdir(arguments.target_path):
        os.mkdir(arguments.target_path)
    dataset = AnchorDataset(arguments.dataset_path, arguments.snippet_path)
    generator = DataLoader(dataset)
    max_epochs = 1
    for epoch in range(max_epochs):
        for labels, info in tqdm.tqdm(generator):
            filename = '{}.txt'.format(info[0])
            txt_path = os.path.join(arguments.target_path, filename)
            bboxes = around(squeeze(labels.numpy()[:, :, 1:]), decimals=6)
            if len(bboxes.shape) > 1:
                bboxes1 = bboxes.tolist()
            else:
                bboxes1 = [bboxes.tolist()]
  
            #print(info, bboxes)
            try:
                with open(txt_path, "w") as f:
                    for box in bboxes1:
                        box[0] = int(box[0])
                        for i in box:
                            f.write(str(i)+' ')
                        f.write('\n')
            except IOError:
                print("Unable to open input file")
            #break
                
                
            

        
