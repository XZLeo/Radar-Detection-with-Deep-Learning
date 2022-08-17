'''
To compare YOLO approach and Pointnet approach, 
Pointwise IOU and mAP takes clusters as the input.
Thus, YOLO's output needs to be transformed to a cluster by 
finding the points in the bboxes
All bbooxes are output of YOLO, i.e., absolute value in the pixel coordinate
'''
from typing import List
from numpy import array, ndarray, where, zeros

from yolo.visualize import *
from clusters import ClustersMetrics


def box_cluster(snip: ndarray, box:ndarray):
    '''
    given a bbox and its snippet, find points in the bbox
    all the coordinates are in the pixel coordinate
    param snip: ndarray with <x, y, class>
          box: <x1, y1, x2, y2> topleft and bottom right
    return cluster: ndarray with <x, y, class>
    '''
    x1, y1, x2, y2 = box
    idx = where(snip["x"]>=x1)[0] 
    cluster = snip[idx]
    idx = where(cluster["x"]<=x2)[0]
    cluster = cluster[idx]
    idx = where(cluster["y"]>=y1)[0]
    cluster = cluster[idx]
    idx = where(cluster["y"]<=y2)[0]
    cluster = cluster[idx]
    return cluster


def boxes_clusters(snippet:ndarray, boxes:ndarray, flg):
      '''
      transfer bounding boxes to clusters 
      param snippet: 500ms frames that is passed to the model for inference
            boxes: model detections or ground truth bboxes
      return list_clst: each element of the list is a cluster, 2D ndarray
      '''
      # transfer the snippet from ego-vehicle 
      # coordinate to the pixel coordinate
      dimension = snippet.shape
      snip = zeros(dimension,
                  dtype=[('x', 'float32'), ('y', 'float32')]) # delete class in the end , ('class', 'uint8')
      snip['x'] = 304 - 304 * snippet['y_cc'] / 50
      snip['y'] = 608 - 608 * snippet['x_cc'] / 100
      #snip['class'] = snippet['label_id']
      
      num_boxes = boxes.shape[0] # number of boxes
      if flg == 'pred':
            # assume box is in <x1, y1, x2, y2, confidence, class>
            pred_boxes = boxes[:, :4] # box geometry, by default it is x1, y1, x2, y2
            pred_scores = boxes[:, 4] # confidence/object score
            pred_labels = boxes[:, -1] # class label

            # loop over boxes
            point_list = []
            for i in range(num_boxes):
                  box = pred_boxes[i, :]
                  cluster = box_cluster(snip, box)
                  point_list.append(cluster)
            return num_boxes, pred_scores, pred_labels, point_list
      elif flg == 'grd':
            # assume box is in [batch_idx, class, x1, y1, x2, y2]
            grd_boxes = boxes[:, 2:] # box geometry, by default it is x1, y1, x2, y2
            grd_labels = boxes[:, 1] # class label

            # loop over boxes
            point_list = []
            for i in range(num_boxes):
                  box = grd_boxes[i, :]
                  cluster = box_cluster(snip, box)
                  point_list.append(cluster)
            return grd_labels, point_list
      else:
            print('Wrong Flag')


class box2clusterMetrices(ClustersMetrics):
      # rewrite
      def add_pred_pnts(self, points:ndarray):
            self.pred_clusters_points = points
            
      def add_pred_labels(self, labels):
            self.pred_clusters_labels = labels
            
      def add_conf(self, confidence:ndarray):
            self.pred_clusters_confs = confidence
      
      #ground truth methods          
      def add_gt_labels(self, gt_lbls:ndarray):
            self.ground_truths_labels = gt_lbls
            
      def add_gt_pnts(self, gt_pts:ndarray):
            self.ground_truths_points = gt_pts 



