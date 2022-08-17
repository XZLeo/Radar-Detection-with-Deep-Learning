'''

Get a frame the same as the rada viwer
'''
from time import time
import matplotlib.pyplot as plt
import numpy as np    
from radar_scenes.coordinate_transformation import *
from radar_scenes.sequence import Sequence
import os


def get_timestamps(sequence: Sequence):
    '''
    Create the list of all timesteps
    '''
    timestamps = []
    for idx, scene in enumerate(sequence.scenes()):
        radar_data = scene.radar_data
        timestamps.append(radar_data[0][0]) 
    return timestamps
        

def get_current_scenes(sequence: Sequence, cur_idx, timestamps, n_prev_frames:int, n_next_frames: int): 
    """
    Retrieves the scenes which should be displayed according to the current values of the time slider and the
    spinboxes for the past and future frames.
    Values of the spinboxes are retrieved and from the list of timestamps, the corresponding times are obtained.
    :return: The current frame (type Scene) and a list of other frames (type Scene) which should be displayed.
    """
    # cur_idx = timeline_slider.value()
    cur_timestamp = timestamps[cur_idx]
    current_scene = sequence.get_scene(cur_timestamp)
    other_scenes = []
    for i in range(1, n_prev_frames + 1):  # in the GUI, show previous frames/ show future frames!
        if cur_idx - i < 0:
            break
        t = timestamps[cur_idx - i]
        other_scenes.append(sequence.get_scene(t))
    for i in range(1, n_next_frames + 1):
        if cur_idx + i >= len(timestamps):
            break
        t = timestamps[cur_idx + i]
        other_scenes.append(sequence.get_scene(t))
    return current_scene, other_scenes


def trafo_radar_data_world_to_car(scene, other_scenes) -> np.ndarray:
    """
    Transforms the radar data listed in other_scenes into the same car coordinate system that is used in 'scene'.
    :param scene: Scene. Containing radar data and odometry information of one scene. The odometry information from
    this scene is used to transform the detections from the other timestamps into this scene.
    :param other_scenes: List of Scene items. All detections in these other scenes are transformed
    :return: A numpy array with all radar data from all scenes. The fields "x_cc" and "y_cc" are now relative to the
    current scene.
    """
    if len(other_scenes) == 0:
        return scene.radar_data
    other_radar_data = np.hstack([x.radar_data for x in other_scenes])
    x_cc, y_cc = transform_detections_sequence_to_car(other_radar_data["x_seq"], other_radar_data["y_seq"],
                                                        scene.odometry_data)
    other_radar_data["x_cc"] = x_cc
    other_radar_data["y_cc"] = y_cc
    return np.hstack([scene.radar_data, other_radar_data])


def get_frames(sequence: Sequence, cur_idx, timestamps, n_prev_frames=0 , n_next_frames=0):
    """
    Plot the current frames.
    :param: cur_idx: the frame number to be ploted
    :return: None
    """
    if len(timestamps) == 0 or cur_idx >= len(timestamps):
        return
    cur_timestamp = timestamps[cur_idx]
    current_scene, other_scenes = get_current_scenes(sequence, cur_idx, timestamps, n_prev_frames, n_next_frames)   # 4 sensors together
    radar_data = trafo_radar_data_world_to_car(current_scene, other_scenes) 
    return radar_data


def plot_frames(radar_data: list):
    # extract x, y from the list
    x = radar_data["x_cc"]
    y = radar_data["y_cc"]
    col = [0, 0, 0, 1]
    plt.plot(
        y,
        x,
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=3
    )
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_xaxis()
    plt.show()
    return


if __name__ == '__main__':
    # extract a frame, i.e., 4 continuous scenes from the start time, for DBSCAN
    path_to_dataset = "/home/s0001516/thesis/dataset/RadarScenes"
    # Define the *.json file from which data should be loaded
    filename = os.path.join(path_to_dataset, "data", "sequence_137", "scenes.json")
    sequence = Sequence.from_json(filename)
    timestamps = get_timestamps(sequence)
    cur_idx = 0
    radar_data = get_frames(sequence, cur_idx, timestamps, n_next_frames=3)
    plot_frames(radar_data)
    # change to a frame class, so that boxes can inheritate!
    