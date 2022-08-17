from os import path
import sys
sys.path.append('..')
from radar_scenes.sequence import Sequence
from numpy import stack, float32, array, zeros, sum, ndarray, linspace
from matplotlib.pyplot import imshow, colorbar, title, grid
import matplotlib.patches as pc
from pytorchyolo.models import load_model

from snippet import clip
import torch
from matplotlib import pyplot as plt
from yolo.detect import detect_image
from yolo.preprocessing.boxes import get_AABB_snippet
from frame import get_timestamps, get_frames
from yolo.preprocessing.gridMap import GridMap
from yolo.preprocessing.coordinateTransfer import coor_transfer
from yolo.preprocessing.boxes import visualize_AABB_cloud


LABEL_NUM = { 0:'CAR',
    1:'PEDESTRIAN',
    2:'PEDESTRIAN_GROUP', 
    3:'TWO_WHEELER',
    4:'LARGE_VEHICLE', 
    5:'STATIC'}
COLOR_SCHEME = { 'CAR':  '#FF6347',
    'PEDESTRIAN': '#4B0082',
    'PEDESTRIAN_GROUP': '#FFD700', 
    'TWO_WHEELER': '#8FBC8F',
    'LARGE_VEHICLE': '#008B8B', 
    'STATIC': '#9400D3'}
GND_LINE = '-'
DET_LINE = '--'


def reverse_transfer(boxes):
    '''
    transfer boxes's coordinate from image coordinate  to ego-vehicle coordinate
    and also in yolo format, i.e., ratio
    param: boxes: n*5 ndarray, [x, y, w, h, class]
    return: tran_boxes: transferred boxes in image coordinate
    '''
    # frame size
    LENGTH = 100
    # print(boxes.shape)
    tran_boxes = zeros((boxes.shape[0], boxes.shape[1]))
    # print(tran_boxes.shape)
    tran_boxes[:, 0] = boxes[:, 4] 
    tran_boxes[:, 1] = 100 - 50 * (boxes[:, 1]+boxes[:, 3]) / 608
    tran_boxes[:, 2] = - 50*(boxes[:, 0]+boxes[:, 2])/608+50
    tran_boxes[:, 3] = 100 * abs(boxes[:, 1]-boxes[:, 3]) / 608
    tran_boxes[:, 4] = 100 * abs(boxes[:, 2]-boxes[:, 0]) / 608
    return tran_boxes


def visualize_AABB_cloud(ax, points: ndarray, truth, detection)->None:
    '''
    Visualize the ground truth and detections on ego-vehicle coordinate
    param truth: 
    '''
    
    col = [0, 0, 0, 1]
    #plot point cloud
    ax.plot(
        points[1, :], #y_cc
        points[0, :], #x_cc
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="#A9A9A9",
        markersize=1
    )
        # plot AABB
    for box in truth:
        c, center_x, center_y, w, h = box # w is always on x axis
        color = COLOR_SCHEME[LABEL_NUM[c]] 
        bottom_left_horizon = center_y - h/2
        bottom_left_vertical = center_x - w/2
        rect = pc.Rectangle((bottom_left_horizon, bottom_left_vertical), h, w,
                            angle=0, fill=False, edgecolor = color,linewidth=1,
                            linestyle=GND_LINE, label='{}'.format('GND '+LABEL_NUM[c]))
        ax.add_patch(rect)
    
    for box in detection:
        c, center_x, center_y, w, h = box # w is always on x axis
        color = COLOR_SCHEME[LABEL_NUM[c]]
        bottom_left_horizon = center_y - h/2
        bottom_left_vertical = center_x - w/2
        rect = pc.Rectangle((bottom_left_horizon, bottom_left_vertical), h, w,
                            angle=0, fill=False, edgecolor = color,linewidth=1, 
                            linestyle=DET_LINE, label='{}'.format('DET '+LABEL_NUM[c]))
        ax.add_patch(rect)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_xaxis()
    ax.set_xlabel('y_cc')
    ax.set_ylabel('x_cc')
    ax.set_xlim(50, -50)
    ax.set_ylim(50, -50)
    plt.legend() #'Car', 'Ped', 'Grp', 'Cyc', 'Tru
    return ax


def image_generator():
    for cur_idx in range(0, 1000, 27):
        radar_data = get_frames(sequence, cur_idx, timestamps, n_next_frames=num_future_frames)
        radar_data = clip(radar_data)
        map = GridMap(radar_data, num_cell=608)
        map.get_max_amplitude_map()
        map.get_Doppler_map(mode=True, skew=True) #'max'
        map.get_Doppler_map(mode=False, skew=True) #'min'
        map.filter_blurry()
        image = stack((map.amp_map, 
                        map.max_doppler_map, map.min_doppler_map), axis=2)  # different from grid_dataset.py!!!
        image = image.astype(float32)

        detections = detect_image(model, image, img_size=608, conf_thres=0.10, nms_thres=0.6)
        boxes = detections[:, [0, 1, 2, 3, 5]]
        
        tran_boxes = reverse_transfer(boxes)
        #print(tran_boxes.shape)
        # gnd bboxes
        aligned_boxes = get_AABB_snippet(radar_data)
        yield (aligned_boxes, tran_boxes, radar_data)


model_path = "yolo/config/yolov3.cfg"
weights_path = "yolo/checkpoints/500ms/yolov3_ckpt_50.pth"
model = load_model(model_path, weights_path).float()
# extract a frame, i.e., 4 continuous scenes from the start time, for DBSCAN
filename = "/home/s0001516/thesis/dataset/RadarScenes/sequence_109/scenes.json"
# Define the *.json file from which data should be loaded
sequence = Sequence.from_json(filename)
timestamps = get_timestamps(sequence)
num_future_frames = 27

fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(1, 1, 1)

for aligned_boxes, tran_boxes, radar_data in image_generator():
    point_cloud = array([radar_data['x_cc'], radar_data['y_cc']])
    ax = visualize_AABB_cloud(ax, point_cloud, tran_boxes, aligned_boxes)
    # visualize the frame

    plt.grid()
    fig.canvas.flush_events()
    plt.show() 
    plt.pause(.1)
    ax.cla()