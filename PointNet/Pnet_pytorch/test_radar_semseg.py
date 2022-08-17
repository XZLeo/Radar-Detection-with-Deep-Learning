"""
__author__ = Leonardo Carrera
__description_ = Performs clustering on the RadarScene dataset with a modified
sklearn DBSCAN clustering algorithm for radar using the predicitions from the
semantic segmentation network Pointnet++ 
"""
import time
import argparse
import os
import sys
import torch
import logging
import math
from pathlib import Path
import importlib
from tqdm import tqdm
import provider
import numpy as np
from torch.utils.data import DataLoader, Subset
from PointNet.pnet_dataset import PNETDataset, get_gt_clusters_id
from metrics import BatchStats, Metrics, print_eval_stats
from clusters import ClustersMetrics
from sklearn.metrics import rand_score
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix
from PointNet.pnet_visualizer import plot_snippet_labels_and_clusters, plot_snippet_labels
import matplotlib.pyplot as plt
import matplotlib as mpl
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

#Root directory for accesing the trained model and configuration files
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))

#Names of the classes used in this approach 
classes = ['CAR', 'PEDESTRIAN', 'PEDESTRIAN_GROUP',  
           'TWO_WHEELER', 'LARGE_VEHICLE', 'STATIC']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat
    
def createConfusionMatrix(y_pred, y_true, type_cf):                        
    '''
    Compute confusion matrix (absolute or relative) using two pairs of vectors
    '''    
    cf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5])
    cf = cf_matrix.copy() 
    if type_cf == "perc":
        cf = np.nan_to_num(cf / np.sum(cf, axis=1)[:, np.newaxis])
    return cf    
    
def snippet_subset(test_dataset, idxs_snippets, num_snippets):
    '''
    Prints the index stated in snippet.py for reference of where
    the snippet is taken from
    ''' 
    test_index_start = idxs_snippets
    numel_test_subset = num_snippets
    nss = test_dataset.num_snip_seq
    list_t = list(range(test_index_start, (numel_test_subset + test_index_start), 1))
    new_test_dataset = Subset(test_dataset, list_t)
    #customized string for label in text file       
    seq_belong = []
    carry = nss[0]
    tmp_numel = numel_test_subset
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
    seq_belong = np.array(seq_belong)
    return new_test_dataset, seq_belong

def min_points_per_class(k, config):
    '''
    Takes the k-th class iteration and returns the minimum
    points entered by user in the DBSCAN config file
    '''
    if k == 0:   #car
        return config['mp_car']
    elif k == 1: #pedestrian
        return config['mp_ped']
    elif k == 2: #ped group
        return config['mp_pgrp']
    elif k == 3: #two wheeler (bike)
        return config['mp_bike']
    else: #k == 4: #large vehicle
        return config['mp_lcar']

def e_region(xyv1, xyv2, e_v):
    '''
    Computes the radar distance metric for DBSCAN clustering
    Format:
    x1, y1, vr1, x2, y2, vr2 :
    xyv1[0], xyv1[1], xyv1[2], xyv2[0], xyv2[1], xyv2[2],  
    '''
    epsilon = 0
    delta_x = (xyv1[0] - xyv2[0])**2
    delta_y = (xyv1[1] - xyv2[1])**2
    delta_vr = (xyv1[2] - xyv2[2])**2
    vel = 1/(e_v**2)
    epsilon = math.sqrt(delta_x + delta_y + (vel * delta_vr))
    return epsilon

def save_clusters_of_snippet(clusters_snippet:ClustersMetrics, xy_clus, lbl_clus, gt_clus, count_no_noise):
    '''
    Simple method for storing the predicted points and labels for each cluster found in the snippet into a 
    container of type ClusterMetrics, also, stores the confident scores of each predicted cluster using the 
    Rand Score (The Rand Index computes a similarity measure between two clusterings by considering all pairs 
    of samples and counting pairs that are assigned in the same or different clusters in the predicted and 
    true clusterings)
    '''
    clusters_snippet.add_pred_pnts(xy_clus)
    clusters_snippet.add_pred_labels(lbl_clus)
    tmp_conf_cluster = rand_score(lbl_clus,gt_clus)
    clusters_snippet.add_pred_confs(tmp_conf_cluster,count_no_noise)  
    pass

def save_gt_clusters_of_snippet(clusters_snippet:ClustersMetrics, pt, gt):
    '''
    Saves the ground truth points and labels for each cluster found in the snippet
    '''
    for i in range(len(pt)):
        temp_pt = np.stack((pt[i][0], pt[i][1]), axis=1)
        clusters_snippet.add_gt_pnts(temp_pt)
        clusters_snippet.add_gt_labels(gt[i])
    pass

def read_config(dbscan_config_file):
    '''
    Reads the DBSCAN configure file and retrieves the parameters of 
    mp_car, mp_ped, mp_pgrp, mp_bike, mp_lcar, e_xyv, e_v, and vr_thresh 
    for the radar DBSCAN algorithm
    '''
    config = {}    
    try:
        with open(dbscan_config_file, 'r') as file_obj:
            for line in file_obj:
                if line[0] != '#' and line.strip() !='':
                    key, value = line.split('=')
                    if '.' in value.strip():
                        config[key.strip()] = float(value.strip())
                    else:
                        config[key.strip()] = int(value.strip())
        print("DBSCAN config values: ")
        print(config)
    except:
        print ("Error reading the configuration file.\
            expected lines: param = value \n param = {eps, min_pts}, \
            value = {float, int, int}")
        sys.exit()
    return config

def save_predicted_labels_to_text(new_label_dir, str_seq_tested, pred_label):
    '''
    Saves the predicted class labels per snippet in rows for storing purposes
    '''
    filename = os.path.join(new_label_dir, str_seq_tested + '.txt')
    with open(filename, 'w') as pl_save:
        pos = -1
        for ii in range(len(pred_label)):  #rows                                              
            for j in pred_label[ii,:]: #cols
                pos += 1
                if pos == len(pred_label[ii,:]) - 1:
                    pl_save.write(str(int(j)))
                    pos = -1
                else:
                    pl_save.write(str(int(j)) + ',')
            pl_save.write('\n')
        pl_save.close()
    pass

def debugging_gt_pred(orig_snippet, gt, pred_v):
    '''
    Returns the snippet with grround truth labels or predicted ones
    according to debug_gt flag
    '''
    if args.debug_gt:
        return np.c_[orig_snippet, gt]    
    else:
        return np.c_[orig_snippet, pred_v]
    

#PARSER
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument(
        '-dtt',
        '--test_dataset_path',
        type=Path,
        #default= "/home/s0001519/RadarProj/RadarScenes/test", #local
        default= "/RadarScenes/test", #GPU cluster
        help='file path to the pointnet test set'
    )
    parser.add_argument(
        '-stt',
        '--test_snippet_path',
        type=Path,
        #default="/home/s0001519/RadarProj/venv/zenseact_repo/thesisdlradardetection/static/test.txt", #local
        #default="./static/test.txt", #GPU cluster
        default="/RadarScenes/static/test.txt", #GPU cluster
        help='file path of the start indexes (test)'
    )
    parser.add_argument(
        '-db',
        '--dbscan_config_file',
        type=Path,
        #default="/home/s0001519/RadarProj/venv/zenseact_repo/thesisdlradardetection/PointNet/Pnet_pytorch/config", #Local
        default="./PointNet/Pnet_pytorch/config", #GPU cluster
        help='file path of the start indexes (validation)'
    )
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in testing [default: 4]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4097, help='point number [default: 3097]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--iou_thresh', type=float, default=0.3, help='IoU threshold for testing [default: 0.3]')
    parser.add_argument('--num_snippets', type=int, default=1, help='Selects number of snippets for subset (when debug_size argument is triggered)')
    parser.add_argument('--idxs_snippets', type=int, default=1, help='Select index(indices) from where the subset snippet(s) start (when debug_size argument is triggered)')
    parser.add_argument('--debug_size', action='store_true', help='Selects a portion of the snippets for fast testing')
    parser.add_argument('--debug_gt', action='store_true', help='Selects ground truth labels for debugging instead of predicted labels')
    parser.add_argument('--plot_cluster', action='store_true', help='Plots the predicted clusters with labels. (works with --debug_size)')
    parser.add_argument('--plot_labels', action='store_true', help='Plots the predicted labels. (works with --debug_size)')
    parser.add_argument('--show_save_plots', type=str, default='show', help='Shows the plot(s) or save them in main directory')
    return parser.parse_args()

#MAIN
def main(args):
    np.set_printoptions(suppress=True)
    np.seterr(invalid='ignore')
    def log_string(str):
        logger.info(str)
        print(str)
    
    print()
    print("\033[1;32m *** Testing Process Started, WELCOME! *** \n")  
    print("\033[1;0mPytorch Version: ", torch.__version__)

    '''HYPER PARAMETER'''
    print("GPU processor set(s): ", torch.cuda.get_device_name(int(args.gpu)))
    print("Number of GPUs assigned: ", torch.cuda.device_count())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    '''EXPERIMENT DIR'''
    experiment_dir = ROOT_DIR + '/log/sem_seg/' + args.log_dir
    new_label_dir = experiment_dir + '/predicted_labels/'
    new_label_dir = Path(new_label_dir)
    new_label_dir.mkdir(exist_ok=True)
    
    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/logs/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    print("")
    log_string('PARAMETERS ...')
    log_string(args)
    
    '''DBSCAN'''
    db_config_vals = read_config(args.dbscan_config_file)
    
    '''RADARSCENES DATASET'''
    print()
    print("\033[1;93mLoading Radarscenes dataset... ","\033[1;0m")
    
    num_classes = 6
    batch_size = args.batch_size
    num_point = args.num_point
    test_data_path = args.test_dataset_path
    test_snip_path = args.test_snippet_path 
    test_dataset = PNETDataset(test_data_path, test_snip_path, num_point, jitter_data=False)
        
    #getting a subset (for debugging only)   
    str_seq_tested = ""    
    if (args.debug_size): 
        test_dataset, seq_belong = snippet_subset(test_dataset, args.idxs_snippets, args.num_snippets)
        str_seq_tested = "sequence_labels_"# + '_'.join(map(str, seq_belong))
        print("sequences from where the snippets belong to: ", seq_belong)
    else:
        str_seq_tested = "sequence_labels_"
        print("sequences from where the snippets belong to: ALL DATASET")  
    
    # if batch is larger than the actual dataset (or subset)
    if batch_size > len(test_dataset):
        batch_size = len(test_dataset)

    '''DATALOADER FOR RADARSCENES DATASET'''
    testDataLoader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False, 
                                                 pin_memory=True,num_workers=8,drop_last=False)
            
    #shape viewer
    for data, label, _, _ in testDataLoader:
        print("Data Shape and Length:")
        print("[batch, n_points, features(x/y/vr/RCS)]")
        print("Shape of one training batch: ", data.shape, "\nShape of one label's batch:  ", label.shape)
        #print(data[:,0,:]) #[[4, 2048, 6]] (EXAMPLE)
        break           
    log_string("The number of test snippets is: %d" % len(test_dataset))
    print()
    
    '''MODEL LOADING'''
    model_name = ''
    for model_n in os.listdir(experiment_dir + '/logs'):
        if model_n != 'eval.txt':
            model_name = model_n.split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_classes).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()
        
    print()
    print("\033[1;32m Testing launched... ")   
    print("\033[1;0m ")
    
    with torch.no_grad():        
        num_batches = len(test_dataset)
        iterations = int(np.ceil(num_batches / batch_size))
        total_correct = 0
        total_seen = 0
        total_seen_class = list(np.zeros(num_classes))
        total_correct_class = list(np.zeros(num_classes))
        total_iou_deno_class = list(np.zeros(num_classes))
        total_predicted_class = list(np.zeros(num_classes))
        pred_label = np.array([])
        firstIteration = True
        if (args.plot_cluster) or (args.plot_labels):
            xylc_cluster = []
            xyl_labels = []   
                
        log_string('---- EVALUATION ----')
        cf1_array = np.zeros((6,6))
        start_time = time.time()
        accum_time_pnet = 0.0
        accum_time_dbscan = 0.0
        batch_metrics = []
        gt_labels_total = []
        snippet_gt_clusters_label = []
        for i, (points, target, batch_upsample, trid) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            #print("Inference on batch [%d/%d] ..." % (i + 1, iterations))
            batch_plain_vector = []#np.zeros(BATCH_SIZE * NUM_POINT)
            pred_plain_vector = []#np.zeros(BATCH_SIZE * NUM_POINT)
            total_seen_class_tmp = list(np.zeros(num_classes))
            total_predicted_class_tmp = list(np.zeros(num_classes))
            total_correct_class_tmp = list(np.zeros(num_classes))
            total_iou_deno_class_tmp = list(np.zeros(num_classes))  
            
            trid = trid.data.numpy() #ground truth track_id clusters per snippet       
            
            # ------ Prediction stage from trained model ------
            points = points.data.numpy() #points in the batch
            pts_dbscan = np.delete(points, 3, axis=2) #discarding RCS (for dbscan)
            g_truth = target.data.numpy() 
            points = points[:,:,:-1] #discard range values (for pointnet)
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            start_time_pnet = time.time()
            seg_pred, trans_feat = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, num_classes)
            end_time_pnet = time.time()
            accum_time_pnet += end_time_pnet - start_time_pnet             

            batch_label = target.cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]  #ground truth
            pred_val = np.argmax(pred_val, 2)  #vector of predictions
            # ------ End of prediction stage from trained model ------
            
            #one vector for all the labels in the batch (predicted and grnd trth)
            batch_plain_vector = np.reshape(batch_label, -1)
            pred_plain_vector = np.reshape(pred_val, -1)
            
            #Loops for each snippet in the batch           
            clusters_in_snippet_batch = []
            for j in range(len(batch_upsample)):
                start_time_dbscan = time.time()
                # ------ Getting the original points and their predicted labels -----------
                cur_n_points = 0
                original_pts = np.array([])
                pred_v = pred_val[j,:]
                gt = g_truth[j,:]
                orig_snippet = pts_dbscan[j,:,:]
                if batch_upsample[j]: #if snippet was upsampled
                    orig_snippet = np.c_[orig_snippet, gt]
                    _, indexes = np.unique(orig_snippet, axis=0, return_index=True)
                    orig_snippet = np.delete(orig_snippet,4,1)
                    orig_snippet = debugging_gt_pred(orig_snippet, gt, pred_v)
                    orig_snippet = np.c_[orig_snippet, gt]
                    original_pts = orig_snippet[np.sort(indexes)]
                else: #if snippet was not upsampled
                    orig_snippet = debugging_gt_pred(orig_snippet, gt, pred_v)
                    orig_snippet = np.c_[orig_snippet, gt]
                    original_pts = orig_snippet
                original_pts_gt = np.copy(original_pts)
                original_pts, _ = provider.doppler_static_filter(original_pts, vr_thresh=db_config_vals['vr_thresh'])
                cur_n_points = original_pts.shape[0]
                # ------ End of getting the original points and their predicted labels ------
                                    
                # ------------------------ modified scikit DBSCAN for radar:-------------------------------             
                clustersIDs_in_snippet = -np.ones(cur_n_points)                               
                orig_pred_v = original_pts[:,4]
                orig_gt = original_pts[:,5]
                xyl_temp = np.delete(original_pts,[2,3,5],1) #no vr(2),range(3),gt(5)
                xygt_temp = np.delete(original_pts_gt,[2,3,4],1) #no vr(2),range(3),pred_v(4) 
                xyv_orig_points_dbscan = np.delete(original_pts,[3,4,5],1) # delete range, pred_v and gt
                xy_orig_points = np.delete(xyv_orig_points_dbscan,2,1) # delete vr
                prev_max_clusterID = 0
                #iterate over all classes but static
                for k1 in range(num_classes -1):
                    aux_pred_indices = np.where(orig_pred_v == k1) #indices (per point) of class predicitons
                    if aux_pred_indices[0].shape[0] > 0:
                        xyv = xyv_orig_points_dbscan[aux_pred_indices[0],:] #getting points with the predicted indices
                        min_p = min_points_per_class(k1, db_config_vals)
                        db = DBSCAN(eps=db_config_vals['e_xyv'], 
                                    min_samples=min_p, 
                                    metric=e_region, 
                                    metric_params={'e_v':db_config_vals['e_v']}).fit(xyv)
                        curr_max_clusterID = np.max(db.labels_) + 1      #number of clusters in this for loop run
                        dbscan_total_labels = np.where(db.labels_ == -1, db.labels_, 
                                                        prev_max_clusterID + db.labels_)
                        clustersIDs_in_snippet[aux_pred_indices] = dbscan_total_labels
                        prev_max_clusterID += curr_max_clusterID
                end_time_dbscan = time.time()
                accum_time_dbscan += end_time_dbscan -start_time_dbscan
                # ------------------------ End of modified scikit DBSCAN for radar:-------------------------------
                        
                if (args.plot_cluster) or (args.plot_labels):
                    # storing predicted data for plot
                    # two tabs to all the if to see how the clusters generate
                    xylc_temp = np.c_[xyl_temp, clustersIDs_in_snippet]
                    xylc_cluster.append(xylc_temp.tolist())
                    xyl_labels.append(xyl_temp.tolist())
                
                # -------- Storing predicted and ground truth clusters of the snippet in batch -----------
                try:
                    clusters_snippet = ClustersMetrics(int(np.max(clustersIDs_in_snippet)) + 1) #object to contain all the clusters in snippet
                    # *** prediction clusters ***
                    count_no_noise = 0
                    for cl in np.unique(clustersIDs_in_snippet):
                        if cl != -1:
                            aux_clus_ids = np.where(clustersIDs_in_snippet == cl)
                            xy_clus = xy_orig_points[aux_clus_ids[0],:]
                            lbl_clus = orig_pred_v[aux_clus_ids]
                            gt_clus = orig_gt[aux_clus_ids] # why iterating all the time and with #aux_clus_ids LEOX
                            save_clusters_of_snippet(clusters_snippet, xy_clus, 
                                                    lbl_clus, gt_clus, count_no_noise)
                            count_no_noise += 1     
                    # *** ground truth clusters ***
                    single_trid = trid[j][trid[j] != -2] #discard the "-2 cluster" created in pnet_dataset.py
                    gt_xyl = np.delete(original_pts_gt,[2,3,4],1) #remove vr, range, pred_v
                    gt_pnts, gt_lbls = get_gt_clusters_id(single_trid, gt_xyl)
                    save_gt_clusters_of_snippet(clusters_snippet, gt_pnts, gt_lbls)
                    # Append gt clusters and pred clusters to the computed batch
                    clusters_in_snippet_batch.append(clusters_snippet)    
                except ValueError:  #raised if clusters_snippet is empty.
                    print("Clusters not found in the snippet. Discarding snippet from the batch")
                    pass
                # ------------------------ End of Storing clusters --------------------------------------
                
                if (args.plot_cluster) or (args.plot_labels): #storing ground truth data for plot
                    xylc_temp = np.c_[xygt_temp, single_trid]
                    xylc_cluster.append(xylc_temp.tolist())
                    xyl_labels.append(xygt_temp.tolist())
                
            #concatenate ground-truth-clusters' labels
            for ii in range(len(clusters_in_snippet_batch)):
                snippet_gt_clusters_label.append(clusters_in_snippet_batch[ii].ground_truths_labels)        
                                
            # ------------------- Batch Metrics (TP and FP) ------------------------
            batch_obj = BatchStats(clusters_in_snippet_batch, args.iou_thresh - 1e-6)
            batch_metrics += batch_obj.get_batch_statistics() #output = [array(True_positives), 
                                                              #          array(confidence_per_cluster), 
                                                              #          array(predicted_labels)]
            # ---------------- End of Batch Metrics (TP and FP) --------------------
            
            correct = np.sum((pred_val == batch_label))
            total_correct += correct
            total_seen += (batch_size * num_point)          
            
            if firstIteration:
                pred_label = pred_val
                firstIteration = False
            else:
                pred_label = np.vstack((pred_label,pred_val))
            
            for l in range(num_classes):
                total_seen_class_tmp[l] += np.sum((batch_label == l))
                total_predicted_class_tmp += np.sum((pred_val == l))
                total_correct_class_tmp[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class_tmp[l] += np.sum(((pred_val == l) | (batch_label == l)))
                #total tested dataset
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]      
                        
            cf1 = createConfusionMatrix(pred_plain_vector, batch_plain_vector, type_cf="abs")
            cf1_array += np.array(cf1)
        
        # --------------------------- Total Metrics and Results ---------------------------------------
        gt_labels_total = np.concatenate(snippet_gt_clusters_label)
        
        log_string('*******************************')
        log_string('******** Final Results ********')
        log_string('*******************************')
        log_string('--- PointNet++ predictions: ---')
        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
        iou_per_class_str = '------- IoU --------\n'
        for l in range(num_classes):
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                total_correct_class[l] / (float(total_iou_deno_class[l]) + 1e-6))
        log_string(iou_per_class_str)
        log_string('eval point avg class IoU: %f' % np.mean(IoU))
        log_string('eval whole scene point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
        log_string('eval whole scene point accuracy: %f' % (
                np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))
        log_string('\nConf_matrix (TOTAL absolute): \n%s' % np.array2string(cf1_array, precision=2, separator=','))
        cf_per = np.nan_to_num(cf1_array / np.sum(cf1_array, axis=1)[:, np.newaxis])
        log_string('Conf_matrix (TOTAL Relative): \n%s' % np.array2string(cf_per, precision=4, separator=','))
        log_string('\n')
        
        log_string('--- PointNet++ and DBSCAN pipeline: ---')
        if len(batch_metrics) == 0:  # No detections over whole set.
            print("---- No detections over whole set ----")
            log_string('No AP table (no detections found in set)')
        else:
            true_positives, pred_scores, pred_labels = [
                np.concatenate(x, 0) for x in list(zip(*batch_metrics))]
            
            metrics_obj = Metrics(true_positives, pred_scores, pred_labels, gt_labels_total)
            metrics_output = metrics_obj.ap_per_class()
            table_ap, map_total, f1_total = print_eval_stats(metrics_output, classes, verbose=True)
            log_string('%s' % table_ap)
            log_string('mAP total: %f' % map_total)
            log_string('F1 score total: %f' % f1_total)
        log_string('\n')
        # --------------------------- End of Total Metrics and Results ---------------------------------------  
        
        #predicted labels to txt (comment if wanted)
        save_predicted_labels_to_text(new_label_dir, str_seq_tested, pred_label)
        end_time = time.time()
        log_string('Total time taken (PointNet): %f' % (accum_time_pnet))
        log_string('Total time taken (DBSCAN): %f' % (accum_time_dbscan))
        log_string('Total time taken (PointNet + DBSCAN): %f' % (accum_time_dbscan + accum_time_pnet))
        log_string('Total time taken (ALL): %f' % (end_time - start_time))
        log_string('Done!')
        log_string('\n\n\n\n')
        log_string('  ***********************************  ')    
        log_string('\n\n\n\n')
        
        #for plotting results
        filenames = []
        if (args.plot_cluster) and not (args.plot_labels):
            width, height = 12, 6
            plt.figure(figsize=(width,height)) #width, height
            first_iter = False
            for co, r in enumerate(range(0,len(xylc_cluster),2)):  
                if first_iter == True:
                    plt.figure(figsize=(width,height))
                first_iter = True
                l_str1 = 'PREDICTED CLUSTERS (Snip. N.' + str(co + 1) + ')'
                l_str2 = 'GROUND TRUTH CLUSTERS (Snip. N.' + str(co + 1) + ')'
                xylc_t1 = np.array(xylc_cluster[r])
                xylc_t2 = np.array(xylc_cluster[r+1])
                plt.subplot(1, 2, 2)
                plot_snippet_labels_and_clusters(xylc=xylc_t1, NUM_CLASSES=6, legend_str=l_str1, 
                                                 plot_filter=False, plt_obj=plt)
                plt.subplot(1, 2, 1)
                plot_snippet_labels_and_clusters(xylc=xylc_t2, NUM_CLASSES=6, legend_str=l_str2, 
                                                 plot_filter=False, plt_obj=plt)
                #store the plots
                if (args.show_save_plots) == "save":
                    filename = f'{co}.png'
                    filenames.append(filename)
                    # save frame
                    plt.savefig(filename)
            if (args.show_save_plots) == "show":
                plt.show()
            
        elif (args.plot_labels) and not (args.plot_cluster) and (args.deb):
            width, height = 12, 6
            plt.figure(figsize=(width,height)) #width, height
            first_iter = False
            for co, r in enumerate(range(0,len(xyl_labels),2)):  
                if first_iter == True:
                    plt.figure(figsize=(width,height))
                first_iter = True
                l_str1 = 'PREDICTED LABELS (Snip. N.' + str(co + 1) + ')'
                l_str2 = 'GROUND TRUTH LABELS (Snip. N.' + str(co + 1) + ')'
                xyl_t1 = np.array(xylc_cluster[r])
                xyl_t2 = np.array(xylc_cluster[r+1])
                plt.subplot(1, 2, 2)
                plot_snippet_labels(xyl=xyl_t1, NUM_CLASSES=6, legend_str=l_str1,
                                    plot_filter=False, plt_obj=plt)
                plt.subplot(1, 2, 1)
                plot_snippet_labels(xyl=xyl_t2, NUM_CLASSES=6, legend_str=l_str2,
                                    plot_filter=False, plt_obj=plt)
                #store the plots
                if (args.show_save_plots) == "save":
                    filename = f'{co}.png'
                    filenames.append(filename)
                    # save frame
                    plt.savefig(filename)
            if (args.show_save_plots) == "show":
                plt.show()
            
            
if __name__ == '__main__':
    args = parse_args()
    main(args)