'''
Class to store the clusters in a snippet
'''
import numpy as np

class ClustersMetrics(object):
    def __init__(self, n_clusters):
        #predicted cluster
        self.pred_clusters_confs = np.zeros((n_clusters,)) #confidence per cluster       
        self.pred_clusters_labels = [] #predicted labels per point and per cluster
        self.pred_clusters_points = [] #predicted points (x,y) per cluster       
        #ground truth clusters
        self.ground_truths_labels = [] #annotated labels per point and per cluster
        self.ground_truths_points = [] #annotated points (x,y) per cluster

    # Prediction methods
    def add_pred_pnts(self, points):
        self.pred_clusters_points.append(points)
        
    def add_pred_labels(self, p_labels):
        label = compress_labels(p_labels)
        self.pred_clusters_labels.append(label)
        
    def add_pred_confs(self, confidence, idx):
        self.pred_clusters_confs[idx] = confidence
    
    #ground truth methods   
    def add_gt_pnts(self, gt_pts):
        self.ground_truths_points.append(gt_pts)
               
    def add_gt_labels(self, gt_lbls):
        label = compress_labels(gt_lbls)
        self.ground_truths_labels.append(label)
               
    
    #normalizing ground truth labels (in case of overlapping by RadarScenes)    
    def ground_truth_label_voting(self):
        gt_clusters_in_snippet=self.ground_truths_labels
        #iterating over the batch
        for i in range(len(gt_clusters_in_snippet)):
            single_gt_cluster_labels = gt_clusters_in_snippet[i]
            
            counter = 0
            num = single_gt_cluster_labels[0]
            for j in single_gt_cluster_labels:
                curr_frequency = single_gt_cluster_labels.count(j)
                if(curr_frequency> counter):
                    counter = curr_frequency
                    num = j 
            
            for k in range(len(single_gt_cluster_labels)):
                self.ground_truths_labels[i][k] = num                  
        pass 
    
#shrink method for PointNet:
def compress_labels(cluster_labels):
    return cluster_labels[0]
