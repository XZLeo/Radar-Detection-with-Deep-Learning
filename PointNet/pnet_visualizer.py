"""
This is the Visualizer for Pointnet labels and radar DBSCAN clusters.
It uses the predicted labels from the Pointnet model (txt file retrievedfrom test_radar_semseg.py)
and the cluster functions from radar DBSCAN files
"""
#MODULES
#Python and OS
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from pathlib import Path
#RadarScenes
from PointNet.pnet_dataset import PNETDataset
from radar_scenes.sequence import Sequence
from frame import get_frames, get_timestamps
from labels import ClassificationLabel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def plot_snippet_labels(xyl, NUM_CLASSES, legend_str, plot_filter, plt_obj:plt):
    '''
    Plots the label of each point in the snippet
    '''
    classes = ['CAR', 'PEDESTRIAN', 'PEDESTRIAN_GROUP',  
            'TWO_WHEELER', 'LARGE_VEHICLE', 'STATIC']
    class2label = {cls: i for i, cls in enumerate(classes)}
    seg_classes = class2label
    seg_label_to_cat = {}
    for i, cat in enumerate(seg_classes.keys()):
        seg_label_to_cat[i] = cat
    
    #mpl.rcParams["figure.figsize"]=9,8
    #plt.figure(fig_num)
    unique_labels = set(xyl[:,2])

    count_elems = [0 for _ in range(NUM_CLASSES)]
    for l in range(NUM_CLASSES):
        count_elems[l] += np.sum((xyl[:,2] == l)) 

    colors = [plt_obj.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    mrk = ""
    edge_color = tuple([84/255.0, 83/255.0, 83/255.0, 1])#(r, g, b, 1)
    back_order = 0
    mrk_size = 0
    for k, col in zip(unique_labels, colors):
        if k == 5:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = xyl[:,2] == k
        x1 = xyl[:,0][class_member_mask]
        y1 = xyl[:,1][class_member_mask]
        if k!= 5 and not plot_filter: 
            mrk = "o"
            edge_color = tuple([84/255.0, 83/255.0, 83/255.0, 1])
            back_order = 10
            mrk_size = 8
            edge_width = 0.3
        else: 
            mrk = "." #'x'
            #edge_color = tuple([0/255.0, 102/255.0, 204/255.0, 1])
            edge_color = tuple([169/255.0, 169/255.0, 169/255.0, 1])
            back_order = 1
            mrk_size = 3
            edge_width = 1
        plt_obj.plot(
            x1,
            y1,
            marker=mrk,
            zorder=back_order,
            markeredgewidth = edge_width, 
            fillstyle='none' if k==5 else None,
            alpha=0.95 if k==5 else None, #orig 0.4
            markerfacecolor=tuple(col),
            markeredgecolor=edge_color, #"k"
            markersize=mrk_size,
            linestyle = 'None',
            label='%s (%d)' % (seg_label_to_cat[k], count_elems[int(k)])
        )
    plt_obj.ylabel('Ycc [m]', fontsize=15)
    plt_obj.xlabel('Xcc [m]', fontsize=15)
    #ax.set_facecolor('#eafff5') # color of the back (hex string:)
    plt_obj.tight_layout(h_pad=1.0)
    plt_obj.legend()
    plt_obj.title(legend_str)
    #plt.minorticks_on()
    plt_obj.minorticks_on()
    #plt.grid(which='major',color = 'red', linestyle = '-', linewidth = 0.5)
    #plt.grid(which='minor', linestyle=':', linewidth=0.5, color='black')
    plt_obj.gca().set_aspect('equal', adjustable='box')     
    
def plot_snippet_labels_and_clusters(xylc, NUM_CLASSES, legend_str, plot_filter, plt_obj:plt):
    '''
    Plots the cluster and label from where each point belongs it to
    in each snippet
    '''
    classes = ['CAR', 'PEDESTRIAN', 'PEDESTRIAN_GROUP',  
            'TWO_WHEELER', 'LARGE_VEHICLE', 'STATIC']
    class2label = {cls: i for i, cls in enumerate(classes)}
    seg_classes = class2label
    seg_label_to_cat = {}
    for i, cat in enumerate(seg_classes.keys()):
        seg_label_to_cat[i] = cat
    
    count_elems = [0 for _ in range(NUM_CLASSES)]
    for l in range(NUM_CLASSES):
        count_elems[l] += np.sum((xylc[:,2] == l)) 
    
    cluster_unique_vals = set(xylc[:,3])
    cuv=list(cluster_unique_vals)    
    count_clusters = [0 for _ in range(len(cuv))]
    labels_of_clusters = 5 * np.ones(len(cuv))
    for c, ele in enumerate(cuv):
        count_clusters[c] += np.sum((xylc[:,3] == c))
        if ele != -1:
            index = np.where(xylc[:,3] == c)[0]
            lbl_in_cluster = np.unique(xylc[index,2])
            labels_of_clusters[c] = lbl_in_cluster[0]

    colors = [plt_obj.cm.Spectral(each) for each in np.linspace(0, 1, len(cluster_unique_vals))]
    mrk = ""
    edge_color = tuple([84/255.0, 83/255.0, 83/255.0, 1])#(r, g, b, 1)
    back_order = 0
    mrk_size = 0
    cnt_clus = 0
    for k, col in zip(cluster_unique_vals, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = xylc[:,3] == k
        x1 = xylc[:,0][class_member_mask]
        y1 = xylc[:,1][class_member_mask]
        cluster_legend = ""
        if k!= -1 and not plot_filter: 
            mrk = "o"
            edge_color = tuple([84/255.0, 83/255.0, 83/255.0, 1])
            back_order = 10
            mrk_size = 8
            edge_width = 0.3
            ll = int(labels_of_clusters[int(k)])
            cluster_legend = seg_label_to_cat[ll] + "_" + str(int(k)) + " (" + str(count_clusters[int(k)]) + ")"
        else: 
            mrk = "." #'x'
            edge_color = tuple([169/255.0, 169/255.0, 169/255.0, 1])
            back_order = 1
            mrk_size = 3
            edge_width = 1
            cluster_legend = "STATIC (" + str(count_elems[5]) + ")"
        plt_obj.plot(
            x1,
            y1,
            marker=mrk,
            zorder=back_order,
            markeredgewidth = edge_width, 
            fillstyle='none' if k==-1 else None,
            alpha=0.95 if k==-1 else None, #orig 0.4
            markerfacecolor=tuple(col),
            markeredgecolor=edge_color, #"k"
            markersize=mrk_size,
            linestyle = 'None',
            label='%s' % cluster_legend
        )
        cnt_clus += 1
    plt_obj.ylabel('Ycc [m]', fontsize=15)
    plt_obj.xlabel('Xcc [m]', fontsize=15)
    #ax.set_facecolor('#eafff5') # color of the back (hex string:)
    plt_obj.tight_layout(h_pad=1.0)
    plt_obj.legend()
    plt_obj.title(legend_str)
    #plt.minorticks_on()
    plt_obj.minorticks_on()
    #plt.grid(which='major',color = 'red', linestyle = '-', linewidth = 0.5)
    #plt.grid(which='minor', linestyle=':', linewidth=0.5, color='black')
    plt_obj.gca().set_aspect('equal', adjustable='box')     


def get_pnet_data(test_data_path, test_snip_path, num_point, batch_size, jitter_data):
    '''
    Small method for testing plot_snippet_labels and plot_snippet_labels_and_clusters
    using the pnet_dataset python file
    '''    
    TEST_DATASET = PNETDataset(test_data_path, test_snip_path, num_point, jitter_data=jitter_data)
   
    #getting a subset (in this case, one snippet to plot)  
    subset = True    
    if (subset):                   
        numel_train_subset = 4  
        # Get the sequence from where the snippet belongs to:
        test_index_start = 340 #25-good test with trhe 250 train dataset (change in test_radar...)
        seq_belong = sequence_info(TEST_DATASET, test_index_start, numel_train_subset)
        print("sequence from where the snippet belongs to: ", seq_belong)  
        print()
        
        list_t = list(range(test_index_start, (numel_train_subset + test_index_start), 1))
        TEST_DATASET = torch.utils.data.Subset(TEST_DATASET, list_t) 

    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=batch_size, shuffle=False, 
                                                      pin_memory=True,num_workers=10, drop_last=False)

    #extract the x,y location and the label from the snippet data
    xyl = np.zeros((num_point,3))
    vr = np.zeros(num_point)
    for data, label, _, _ in testDataLoader:
        print("Shape of training data: ", data.shape[0], ". Shape of labels: ", label.shape) 
        print("Length of data: ", len(TEST_DATASET))       
        print()   
        xyl[:,0] = data[:,0].numpy() #x
        xyl[:,1] = data[:,1].numpy() #y
        xyl[:,2] = label.numpy()     #label
        vr = data[:,2].numpy()
        break
    return xyl

def get_raw_data(test_data_path, seq, c_idx, num_fut_fr):
    '''
    Small method for testing plot_snippet_labels and plot_snippet_labels_and_clusters
    using the general dataset python file
    '''   
    # Define the *.json file from which data should be loaded
    filename = os.path.join(test_data_path, "sequence_{}".format(seq), "scenes.json")
    sequence = Sequence.from_json(filename)
    timestamps = get_timestamps(sequence)
    # which snippet to plot
    radar_data = get_frames(sequence,cur_idx=c_idx, timestamps=
                    timestamps,  n_prev_frames=0 , n_next_frames=num_fut_fr)
    point_cloud = np.array([radar_data['y_cc'], radar_data['x_cc']])
    class_label = radar_data['label_id']
    mapped_y = []
    for element in class_label:
        mapped_y.append(ClassificationLabel.label_to_clabel(element).value)
    mapped_y = np.array(mapped_y)
    
    xyl = np.vstack([point_cloud, mapped_y]).T    
    return xyl

    
def sequence_info(TEST_DATASET, test_index_start, numel_train_subset): 
    '''
    Prints the index stated in snippet.py for reference of where
    the snippet is taken from
    ''' 
    nss = TEST_DATASET.num_snip_seq
    seq_belong = []
    carry = nss[0]
    tmp_numel = numel_train_subset
    for i, (cur_nss, nxt_nns) in enumerate(zip(nss, nss[1:])):    
        if test_index_start > carry:
            carry += cur_nss + nxt_nns
        else:
            if cur_nss < tmp_numel:
                seq_belong.append(i+1)                    
                tmp_numel = tmp_numel - cur_nss
            else:
                seq_belong.append(i+1) 
                break
    return seq_belong    
    
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument(
        '-d',
        '--test_dataset_path',
        type=Path,
        default= "/home/s0001519/RadarProj/RadarScenes/train",
        help='file path to the pointnet test set for visualization'
    )
    parser.add_argument(
        '-sn',
        '--test_snippet_path',
        type=Path,
        default="/home/s0001519/RadarProj/venv/zenseact_repo/thesisdlradardetection/static/train.txt",
        help='file path of the start indexes (test)'
    )
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
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
    return parser.parse_args()


def main(args):            
    NUM_CLASSES = 6
    BATCH_SIZE = None
    NUM_POINT = 1000
    TEST_DATA_PATH = args.test_dataset_path
    TEST_SNIP_PATH = args.test_snippet_path 
        
    raw_pnet = 0 #selects between take points directly from pnet dataset or basic dataset
    xyl = [] 
    
    jitter = True
    
    if raw_pnet == 0:
        xyl = get_pnet_data(test_data_path=TEST_DATA_PATH, 
                            test_snip_path=TEST_SNIP_PATH, 
                            num_point=NUM_POINT,batch_size=BATCH_SIZE, jitter_data=jitter) 
    elif raw_pnet == 1:
        xyl = get_raw_data(test_data_path=TEST_DATA_PATH,
                           seq=args.sequence,
                           c_idx=args.currentIndex,
                           num_fut_fr=args.numFutureFrames)     
        
    #Plot GROUND TRUTH labels
    pf = False # plots clusters or labels when False
    plot_snippet_labels(xyl, NUM_CLASSES, 
                        legend_str="Ground Truth", 
                        fig_num=1, 
                        plot_filter=pf,
                        ax_plot=plt.axes())

    #extract the predicted labels from .txt file
    experiment_dir = BASE_DIR + '/Pnet_pytorch/log/sem_seg/' + args.log_dir
    new_label_dir = experiment_dir + '/predicted_labels/'
    new_label_dir = Path(new_label_dir)
    new_label_dir.mkdir(exist_ok=True)
    str_seq_tested = "sequence_labels_"
    filename = os.path.join(new_label_dir, str_seq_tested + '.txt')
    predicted_labels = []
    predicted_labels = open(filename).read().splitlines()
    for i, line in enumerate(predicted_labels):
        predicted_labels[i] = line.split(',')
        predicted_labels[i] = list(map(int, predicted_labels[i]))
    predicted_labels = np.array(predicted_labels)

    #xyl[:,2] = predicted_labels[0,:] #extract one predicted label vector for the one corresponding snippet   

    #Plot PREDICTED labels
    #ax = plt.axes()
    #plot_snippet_labels(xyl, 
    #                    NUM_CLASSES, 
    #                    legend_str="Predicted Labels", 
    #                    fig_num=2, 
    #                    plot_filter=pf,
    #                    ax_plot=ax)
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)