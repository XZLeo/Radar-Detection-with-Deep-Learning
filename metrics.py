from torch import le
import tqdm
import numpy as np
import sys
from numpy import genfromtxt
from clusters import ClustersMetrics
from terminaltables import AsciiTable
from collections import Counter


#for debbuging from csv files only (change for testing in Yolo or Pointnet):
DATA_TST = '/home/s0001519/RadarProj/venv/zenseact_repo/thesisdlradardetection/PointNet/dbscan/data/dataset_test_metrics.csv'
DATA_GT = '/home/s0001519/RadarProj/venv/zenseact_repo/thesisdlradardetection/PointNet/dbscan/data/dataset_gt_metrics.csv'
#until here

CLASS_NAMES = ['CAR', 'PEDESTRIAN', 'PEDESTRIAN_GROUP',
               'TWO_WHEELER', 'LARGE_VEHICLE', 'STATIC']

class Metrics:
    def __init__(self, true_positives, pred_scores, pred_labels, labels):
        self.true_positives = true_positives
        self.pred_scores = pred_scores
        self.pred_labels = pred_labels
        self.labels = labels

    def ap_per_class(self):
        """
        Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
        # Arguments
            tp:    True positives (list).
            conf:  Objectness value from 0-1 (list).
            pred_cls: Predicted object classes (list).
            target_cls: True object classes (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
        tp = self.true_positives
        conf = self.pred_scores
        pred_cls = self.pred_labels
        target_cls = self.labels
        
        # Sort by objectness
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique classes
        unique_classes = np.unique(target_cls) #[0,5]

        # Create Precision-Recall curve and compute AP for each class
        ap, p, r = [], [], []
        for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
            i = pred_cls == c
            n_gt = (target_cls == c).sum()  # Number of ground truth objects
            n_p = i.sum()  # Number of predicted objects

            if n_p == 0 and n_gt == 0:
                continue
            elif n_p == 0 or n_gt == 0:
                ap.append(0)
                r.append(0)
                p.append(0)
            else:
                # Accumulate FPs and TPs
                fpc = (1 - tp[i]).cumsum()
                tpc = (tp[i]).cumsum()

                # Recall
                recall_curve = tpc / (n_gt + 1e-16)
                r.append(recall_curve[-1])

                # Precision
                precision_curve = tpc / (tpc + fpc)
                p.append(precision_curve[-1])

                # AP from recall-precision curve
                ap.append(self.compute_ap(recall_curve, precision_curve))

        # Compute F1 score (harmonic mean of precision and recall)
        p, r, ap = np.array(p), np.array(r), np.array(ap)
        f1 = 2 * p * r / (p + r + 1e-16)

        return p, r, ap, f1, unique_classes.astype("int32")
        
    def compute_ap(self, recall, precision):
        """ Compute the average precision, given the recall and precision curves.
        Code originally from https://github.com/rbgirshick/py-faster-rcnn.
        # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

class BatchStats:
    """
    _summary_ Batch Statistics: computes the true positives, predicted scores (confidence)
    and predicted labels per batch of snippets and their clusters.
    Input: 
        list of clusters (object)
        iou_threshold (0.3 and 0.5 in this thesis)
    """
    
    def __init__(self, clusters_list, iou_threshold):
        self.clusters_list = clusters_list
        self.iou_threshold = iou_threshold
            
    def get_batch_statistics(self):
        """ 
        Compute true positives, predicted scores and predicted labels per batch sample
        """  
        outputs = self.clusters_list
        
        batch_metrics = []
        for sample_i in range(len(self.clusters_list)): #loop for one snippet
                        
            if len(outputs[sample_i].pred_clusters_labels) == 0 and len(outputs[sample_i].ground_truths_labels) == 0:
                continue
            
            output = self.clusters_list[sample_i]        #snippet (output) from snippet batch (outputs)
            pred_scores = output.pred_clusters_confs     #confidence (object score)
            pred_clusters_label = output.pred_clusters_labels  #predicted class labels                  
            pred_clusters_points = output.pred_clusters_points  #predicted cluster points
            gt_clusters_points = output.ground_truths_points   #ground truth annot. points
            gt_clusters_label = output.ground_truths_labels   #ground truth annot clusters  

            # 1 TP value per predicted cluster in snippet
            true_positive_clusters = np.zeros(len(pred_clusters_points)) 
                                    
            annotations = gt_clusters_points
            target_labels = gt_clusters_label if len(annotations) else []
            if len(annotations):
                detected_clusters = []
                target_clusters = annotations #grd trth points

                #for loop for evaluating cluster by cluster in snippet
                for pred_i, (pred_cluster, pred_labels_per_cluster) in enumerate(zip(pred_clusters_points, pred_clusters_label)):
        
                    # If targets are found break
                    # this means that once the detected clusters 
                    # are the same as in annotations it will stop
                    if len(detected_clusters) == len(annotations):
                        break
                    
                    # Ignore if label is not one of the target labels
                    if pred_labels_per_cluster not in target_labels:
                        continue
                    
                    # Filter target_boxes by pred_label so that we only match against boxes of our own label
                    filtered_target_position, filtered_targets = zip(*filter(lambda x: target_labels[x[0]] == pred_labels_per_cluster, enumerate(target_clusters)))
                                        
                    # Find the best matching target for our predicted box
                    ious = []
                    #global_targets_idx = []
                    for ii in range(len(filtered_targets)):
                        iou = pointwise_iou(pred_cluster, filtered_targets[ii])
                        ious.append(iou)                       
                                        
                    best_iou = max(ious)   
                    cluster_filtered_index = ious.index(max(ious))     
                        
                    # Remap the index in the list of filtered targets for that label to the index in the list with all targets.
                    cluster_index = filtered_target_position[cluster_filtered_index]

                    # Check if the iou is above the min treshold and i
                    if best_iou >= self.iou_threshold and cluster_index not in detected_clusters:
                        true_positive_clusters[pred_i] = 1
                        detected_clusters += [cluster_index]
                    
            batch_metrics.append([true_positive_clusters, pred_scores, pred_clusters_label])
        return batch_metrics

    def get_batch_mCOV(self):
        """ 
        Compute true positives, predicted scores and predicted labels per batch sample
        """  
        outputs = self.clusters_list
        
        mmCov = 0
        for sample_i in range(len(self.clusters_list)): #loop over each snippet in a batch
                        
            if len(outputs[sample_i].pred_clusters_labels) == 0 and len(outputs[sample_i].ground_truths_labels) == 0:
                continue
            
            output = self.clusters_list[sample_i]        #snippet (output) from snippet batch (outputs)
            pred_scores = output.pred_clusters_confs     #confidence (object score)
            pred_clusters_label = output.pred_clusters_labels  #predicted class labels                  
            pred_clusters_points = output.pred_clusters_points  #predicted cluster points
            gt_clusters_points = output.ground_truths_points   #ground truth annot. points
            gt_clusters_label = output.ground_truths_labels   #ground truth annot clusters  
                                    
            pred_labels = pred_clusters_label if len(pred_clusters_points) else []
            if len(pred_clusters_points):
                #loop over ground truth cluster
                Cov_sum = 0
                for gt_i, (gt_cluster, gt_labels_per_cluster) in enumerate(zip(gt_clusters_points, gt_clusters_label)):
                            
                    # Ignore if label is not one of the target labels
                    if gt_labels_per_cluster not in pred_labels:
                        continue
                    
                    # Filter pred_boxes by gt_label so that we only match against boxes of our own label
                    _, filtered_preds = zip(*filter(lambda x: pred_labels[x[0]] == gt_labels_per_cluster, enumerate(pred_clusters_points)))
                                        
                    # Find the best matching target for our predicted box
                    best_iou = 0
                    for ii in range(len(filtered_preds)):
                        iou = pointwise_iou(gt_cluster, filtered_preds[ii])
                        best_iou = iou if iou > best_iou else best_iou                                          
                    Cov_sum += best_iou
                mCov = Cov_sum / (gt_i+1)
            else:
                continue       
            mmCov += mCov # total of a batch  
        return mmCov

#using x and y pair of points for both grounnd truth and predictions clusters           
def pointwise_iou(pred_cluster, target_cluster):
    """
    target_cluster is not the class cluter, but only its attribute points 2D ndarray
    """
    num_target = target_cluster.shape[0]
    num_pred = pred_cluster.shape[0]
    # ndarray ==> set of tuple
    
    target_set = set()
    for i in range(num_target):
        x, y = target_cluster[i]
        target_set.add((x, y))
        
    pred_set = set()
    for i in range(num_pred):
        x, y = pred_cluster[i]
        pred_set.add((x, y))
    
    num_intersect = len(target_set.intersection(pred_set))
    num_union = len(target_set.union(pred_set)) + 1e-6  
    return num_intersect / num_union

#printing method
def print_eval_stats(metrics_output, class_names, verbose):
    table_ap = ""
    map_total = 0
    f1_total = 0
    temp_AP = []
    temp_f1 = []
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                if c != 5:
                    ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                    temp_AP.append(AP[i])
                    temp_f1.append(f1[i])
            table_ap = AsciiTable(ap_table).table
        map_total = np.array(temp_AP).mean()
        f1_total = np.array(temp_f1).mean()
    else:
        print("---- mAP not measured (no detections found by model) ----")
        table_ap = "No AP table (no detections found by model)"
    return table_ap, map_total, f1_total    