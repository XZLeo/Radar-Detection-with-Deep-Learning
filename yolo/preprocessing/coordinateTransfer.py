'''
Author: Ziliang
This script contains helper functions for cooridnate transfer
'''
from numpy import ndarray, zeros

# frame size / meters
LENGTH = 100
# gridmap size / pixel
WIDTH = 608


def coor_transfer(boxes: ndarray)->ndarray:
    '''
    transfer boxes's coordinate from ego-vehicle coordinate to image coordinate
    and also in yolo format, i.e., ratio; it also add an extra colum to make it 
    compatible with YOLO
    param: boxes: n*4 ndarray, each row is a boundin box in ego-vehicle coorindate
    return: tran_boxes: transferred boxes in image coordinate [batch_idx, class, x, y, w, h]
                        relative values
    '''
    try:
        tran_boxes = zeros((boxes.shape[0], boxes.shape[1]+1)) # add an extra
    except IndexError:
        return None
    tran_boxes[:, 0] = 0 # add one colum for line 54
    tran_boxes[:, 1] = boxes[:, 0]          #label
    tran_boxes[:, 2] = (-boxes[:, 2] + LENGTH/2) / LENGTH     #x 
    tran_boxes[:, 3] = (LENGTH - boxes[:, 1]) / LENGTH    #y
    tran_boxes[:, 4] = boxes[:, 4] / LENGTH #w
    tran_boxes[:, 5] = boxes[:, 3] / LENGTH #h
    return tran_boxes 


def coor_transfer_xyxy(boxes: ndarray)->ndarray:
    '''
    for box2cluster
    param: boxes: n*4 ndarray, each row is a boundin box in ego-vehicle coorindate
    return: tran_boxes: transferred boxes in image coordinate [batch_idx, class, x1, y1, x2, y2] 
                        pixel values
    '''
    try:
        tran_boxes = zeros((boxes.shape[0], boxes.shape[1]+1)) # add an extra
    except IndexError:
        print(boxes, boxes.shape)
    tran_boxes[:, 0] = 304 * (1-boxes[:, 2]/50-boxes[:, 4]/100)
    tran_boxes[:, 1] = 608 * (1-boxes[:, 1]/100-boxes[:, 3]/200)
    tran_boxes[:, 2] = 304 * (1-boxes[:, 2]/50+boxes[:, 4]/100)
    tran_boxes[:, 3] = 608 * (1-boxes[:, 1]/100+boxes[:, 3]/200)
    tran_boxes[:, 4] = 0 # confidence place holder
    tran_boxes[:, 5] = boxes[:, 1]   #class label
    return tran_boxes

def reverse_transfer(boxes):
    '''
    transfer boxes's coordinate from image coordinate  to ego-vehicle coordinate
    and also in yolo format, i.e., ratio
    param: boxes: n*4 ndarray, each row is a boundin box in pixel coorindate [x, y, w, h]
    return: tran_boxes: transferred boxes in image coordinate
    '''
    tran_boxes = zeros((boxes.shape[0], boxes.shape[1]+1))
    tran_boxes[:, 0] = None
    tran_boxes[:, 1] = LENGTH - LENGTH/2 * (boxes[:, 1]+boxes[:, 3]) / WIDTH
    tran_boxes[:, 2] = - LENGTH/2*(boxes[:, 0]+boxes[:, 2])/WIDTH+LENGTH/2
    tran_boxes[:, 3] = LENGTH * abs(boxes[:, 1]-boxes[:, 3]) / WIDTH
    tran_boxes[:, 4] = LENGTH * abs(boxes[:, 2]-boxes[:, 0]) / WIDTH
    return tran_boxes