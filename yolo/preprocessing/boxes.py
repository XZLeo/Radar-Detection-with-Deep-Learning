'''
Generate bounding box as the minimum exterior rectangle of a cluster by PCA
'''
from argparse import ArgumentParser
import os
import  matplotlib.pyplot as plt
import matplotlib.patches as pc
from numpy import argsort, ndarray, max, min, abs, zeros, where,array, cov, dot, transpose, pi, arctan2
from numpy.linalg import inv, eig
from pathlib import Path
from typing import List, Tuple
from logging import getLogger, INFO, basicConfig

from radar_scenes.sequence import Sequence
from frame import get_frames, get_timestamps
from labels import ClassificationLabel

"""Logger for printing."""
_LOG = getLogger(__name__)
basicConfig(level=INFO)

# data type
AABB = Tuple[float, float, float, float]
OBB = Tuple[float, float, float, float, float]

# frame size
LENGTH = 100


def get_AABB(cluster: ndarray)-> AABB:
    '''
    get axis algned bounding boxes from a frame by finding the max, min x and y
    param cluster: numpy array with the first row as x coordinate, second row as y coordinate
    return aligned_box: <x, y, w, h> as YOLO convention
    '''
    max_x = max(cluster[0, :])
    max_y = max(cluster[1, :])
    min_x = min(cluster[0, :])
    min_y = min(cluster[1, :])
    w = abs(max_x-min_x) # w is defined to be always along x axis
    h = abs(max_y-min_y) # becuase of vehicle coordinate system
    x = (max_x + min_x) / 2
    y = (max_y + min_y) / 2
    aligned_box = (x, y, w, h)
    return aligned_box


def get_OBB(cluster: ndarray)-> Tuple[ndarray, OBB, ndarray]:
    '''
    '''
    cluster = transpose(cluster)
    ca = cov(cluster,y = None,rowvar = 0,bias = 1)

    v, vect = eig(ca)
    tvect = transpose(vect)
    #use the inverse of the eigenvectors as a rotation matrix and
    #rotate the points so they align with the x and y axes
    ar = dot(cluster, inv(tvect))
    # get the minimum and maximum x and y
    mina = min(ar,axis=0)
    maxa = max(ar,axis=0)
    width, height = maxa - mina
    diff = (maxa - mina)*0.5
    # the center is just half way between the min and max xy
    center = mina + diff
    #get the 4 corners by subtracting and adding half the bounding boxes height and width to the center
    corners = array([center+[-diff[0],-diff[1]],center+[diff[0],-diff[1]],
                    center+[diff[0],diff[1]],center+[-diff[0],diff[1]],center+[-diff[0],-diff[1]]])
    #use the the eigenvectors as a rotation matrix and
    #rotate the corners and the centerback
    corners = dot(corners,tvect)
    center = dot(center,tvect)
    # corners are in anticlockwise
    # transfer to YOLO format
    if width > height: # the yaw is always between the long ax of the box and positive y
        tempo = width
        width = height
        height = tempo
        mid_ax = array([(corners[0, :] + corners[3, :])/2, (corners[1, :] + corners[2, :])/2])
    else:
        mid_ax = array([(corners[0, :] + corners[1, :])/2, (corners[2, :] + corners[3, :])/2])
    # to eliminate ambiguity, always let the first point
    #  in middle axis at the left, i.e., has smaller x
    mid_ax = mid_ax[argsort(mid_ax[:, 0])]
    yaw = arctan2(mid_ax[1, 1]-mid_ax[0, 1], mid_ax[1, 0]-mid_ax[0, 0]) #radius
    oriented_box = (center[0], center[1], width, height, yaw /pi*180)
    return corners, oriented_box, mid_ax


def visualize_AABB_cloud(points: ndarray, aligned_boxes: List[AABB])->None:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    col = [0, 0, 0, 1]
    #plot point cloud
    ax.plot(
        points[1, :], #y_cc
        points[0, :], #x_cc
        "o",
        markerfacecolor="r",
        markeredgecolor="k",
        markersize=1
    )
    # plot AABB
    for box in aligned_boxes:
        _, center_x, center_y, w, h = box # w is always on x axis
        bottom_left_horizon = center_y - h/2
        bottom_left_vertical = center_x - w/2
        rect = pc.Rectangle((bottom_left_horizon, bottom_left_vertical), h, w,
                            angle=0, fill=False, edgecolor = 'red',linewidth=2)
        ax.add_patch(rect)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_xaxis()
    ax.set_xlabel('y_cc/$m$', fontsize=15)
    ax.set_ylabel('x_cc/$m$', fontsize=15)
    plt.title('Axis Aligned Bounding Boxes')
    return


def get_AABB_snippet(snippet: ndarray):
    '''
    get aligned boxes from a 500ms snippet
    '''
    track_ids = set(snippet["track_id"])
    aligned_boxes = []
    for tr_id in track_ids:
        if tr_id != b'':  # an empty snippet only have track_id b''
            idx = where(snippet["track_id"] == tr_id)[0] # get the index of non-empty track id
            if len(idx) <= 2: # only one point with same tr_id, ignore it
                #print('only one points labeled')
                continue
                #pass
            # more than 2 pionts in a cluster
            points = zeros((2, len(idx)))
            points[0, :] = snippet[idx]["x_cc"].reshape((1,len(idx)))
            points[1, :] = snippet[idx]["y_cc"].reshape((1,len(idx)))
            class_label = snippet[idx[0]]['label_id']
            #if ClassificationLabel.label_to_clabel(class_label) != None:
            mapped_class_label = ClassificationLabel.label_to_clabel(class_label).value
            #if mapped_class_label is not None:
                # generate AABB, abadon animal/other class
            box = get_AABB(points)
            aligned_boxes.append((mapped_class_label, box[0], box[1], box[2], box[3]))
    return aligned_boxes


def test_snippet(snippet: ndarray):
    '''
    test if all the points has the same class label in a cluster
    '''
    track_ids = set(snippet["track_id"])
    aligned_boxes = []
    for tr_id in track_ids:
        if tr_id != b'':  # an empty snippet only have track_id b''
            idx = where(snippet["track_id"] == tr_id)[0] # get the index of non-empty track id
            if len(idx) <= 2: # only one point with same tr_id, ignore it
                #print('only one points labeled')
                continue
                #pass
            # more than 2 pionts in a cluster
            class_label = snippet[idx]['label_id']
            label_set = set()
            for lab in class_label:
                mapped_class_label = ClassificationLabel.label_to_clabel(lab).value
                label_set.add(mapped_class_label)
            aligned_boxes.append(len(label_set)==1) # if not 1, voting is needed 
    return aligned_boxes


def get_OBB_snippet(snippet: ndarray):
    '''
    get oriented boxes from a 500ms snippet
    '''
    len_snip = len(snippet)
    track_ids = set(snippet["track_id"])
    oriented_boxes = []
    list_corners = []
    list_ax = []
    for tr_id in track_ids:
        if tr_id != b'': 
            idx = where(snippet["track_id"] == tr_id)[0] # get the index of non-empty track id
            if len(idx) < 2: # only one point with same tr_id, ignore it (this needs to be synchronized while creating grid maps!)
                continue
            # more than 2 pionts in a cluster
            points = zeros((2, len(idx)))
            points[0, :] = snippet[idx]["x_cc"].reshape((1,len(idx)))
            points[1, :] = snippet[idx]["y_cc"].reshape((1,len(idx)))
            class_label = snippet[idx[0]]['label_id']
            # generate OBB
            corners, oriented_box, mid_ax = get_OBB(points)
            oriented_boxes.append((class_label, oriented_box[0], 
                            oriented_box[1], oriented_box[2], oriented_box[3], oriented_box[4]))
            list_corners.append(corners)
            list_ax.append(mid_ax)
            _LOG.info("OBB{}".format(oriented_box))
    return oriented_boxes, list_corners, list_ax

# def get_OBB_snippet(snippet: ndarray):
#     '''
#     get oriented boxes from a 500ms snippet
#     '''
#     len_snip = len(snippet)
#     track_ids = set(snippet["track_id"])
#     oriented_boxes = []
#     list_corners = []
#     list_ax = []
#     for tr_id in track_ids:
#         if len(tr_id) == 0: 
#             # no tracked objects, all the points are input as one cluster
#             # , returning a huge bouding boxes
#             points = zeros((2, len_snip))
#             points[0, :] = snippet["x_cc"]#.reshape((1,len(idx)))
#             points[1, :] = snippet["y_cc"] #.reshape((1,len(idx)))
#             class_label = snippet[0]['label_id'] # static
#         else:
#             idx = where(snippet["track_id"] == tr_id)[0] # get the index of non-empty track id
#             if len(idx) < 2: # only one point with same tr_id, ignore it (this needs to be synchronized while creating grid maps!)
#                 continue
#             # more than 2 pionts in a cluster
#             points = zeros((2, len(idx)))
#             points[0, :] = snippet[idx]["x_cc"].reshape((1,len(idx)))
#             points[1, :] = snippet[idx]["y_cc"].reshape((1,len(idx)))
#             class_label = snippet[idx[0]]['label_id']
#         # generate OBB
#         corners, oriented_box, mid_ax = get_OBB(points)
#         oriented_boxes.append((class_label, oriented_box[0], 
#                         oriented_box[1], oriented_box[2], oriented_box[3], oriented_box[4]))
#         list_corners.append(corners)
#         list_ax.append(mid_ax)
#         _LOG.info("OBB{}".format(oriented_box))
#     return oriented_boxes, list_corners, list_ax


def visualize_OBB_cloud(points: ndarray, list_corners: List[ndarray],
                        list_ax: List[ndarray])->None:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
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
    # plot OBB
    for corners in list_corners:
        ax.plot(corners[:,1],corners[:,0],'-')  # draw boxes
    # plot middle ax
    # for axis in list_ax:
    #     ax.plot(axis[:, 1], axis[:, 0], 'r-')
    ax.set_aspect('equal', adjustable='box')
    ax.invert_xaxis()
    ax.set_xlabel('y_cc', fontsize=15)
    ax.set_ylabel('x_cc', fontsize=15)
    plt.title('Oriented Bounding Boxes', fontsize=15)
    return


def main()->None:
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
        '--sequence',
        type=int,
        help='sequence number'
    )
    parser.add_argument(
        '-c',
        '--currentIndex',
        type=int,
        help='sequence number'
    )
    parser.add_argument(
        '-n',
        '--numFutureFrames',
        type=int,
        help='number of next frames'
    )
    arguments = parser.parse_args()

    # Define the *.json file from which data should be loaded
    filename = os.path.join(arguments.dataset_path, "sequence_{}".format(arguments.sequence), "scenes.json")
    sequence = Sequence.from_json(filename)
    timestamps = get_timestamps(sequence)
    # which snippet to plot
    radar_data = get_frames(sequence,cur_idx=arguments.currentIndex, timestamps=
                    timestamps,  n_prev_frames=0 , n_next_frames=arguments.numFutureFrames)
    # extract cluster, each track_id should be a cluster
    aligned_boxes = get_AABB_snippet(radar_data)
    oriented_boxes, list_corners, list_ax = get_OBB_snippet(radar_data)
    # visualize the frame
    point_cloud = array([radar_data['x_cc'], radar_data['y_cc']])
    #visualize_AABB_cloud(point_cloud, aligned_boxes)
    visualize_OBB_cloud(point_cloud, list_corners, list_ax)
    plt.show() 
    return


if __name__ == '__main__':
    main()