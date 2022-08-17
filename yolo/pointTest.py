#! /usr/bin/env python3
'''
For calculating mean-coverage
'''

from __future__ import division

import argparse
import tqdm
import numpy as np

from terminaltables import AsciiTable

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from metrics import BatchStats
from yolo.grid_dataset import AABBDataset
from yolo.utils.models import load_model
from yolo.utils.utils import load_classes, ap_per_class, non_max_suppression, xywh2xyxy, print_environment_info
from yolo.utils.parse_config import parse_data_config
from yolo.utils.box2cluster import boxes_clusters, box2clusterMetrices


def evaluate_model_file(model_path, weights_path, seq_path, snpt_path, class_names, batch_size=8, img_size=608,
                        n_cpu=8, iou_thres=0.5, conf_thres=0.5, nms_thres=0.5, verbose=True):
    """Evaluate model on validation dataset.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param class_names: List of class names
    :type class_names: [str]
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param iou_thres: IOU threshold required to qualify as detected, defaults to 0.5
    :type iou_thres: float, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :param verbose: If True, prints stats of model, defaults to True
    :type verbose: bool, optional
    :return: Returns precision, recall, AP, f1, ap_class
    """
    dataloader = _create_validation_data_loader(
        seq_path=seq_path, 
        snpt_path=snpt_path, 
        batch_size=batch_size,  
        n_cpu=n_cpu)
    model = load_model(model_path, weights_path)
    metrics_output = _evaluate(
        model,
        dataloader,
        class_names,
        img_size,
        iou_thres,
        conf_thres,
        nms_thres,
        verbose)
    return metrics_output


def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")

def _evaluate(model, dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
    """Evaluate model on validation dataset.

    :param model: Model to evaluate
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images with targets
    :type dataloader: DataLoader
    :param class_names: List of class names
    :type class_names: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param iou_thres: IOU threshold required to qualify as detected
    :type iou_thres: float
    :param conf_thres: Object confidence threshold
    :type conf_thres: float
    :param nms_thres: IOU threshold for non-maximum suppression
    :type nms_thres: float
    :param verbose: If True, prints stats of model
    :type verbose: bool
    :return: Returns precision, recall, AP, f1, ap_class
    """
    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    # get snippet here!!
    mmmCov = 0
    for snip, imgs, targets in tqdm.tqdm(dataloader, desc="Validating"): # a batch, snippets are wraped in a tuple
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size # absolute value in pixel coordinate <x1,y1,x2,y2>

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs) # [x1, y1, x2, y2, confidence, class]
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)[0]
        # transfer box to clusters, fit to cluster class
        num_boxes, pred_scores, pred_labels, pred_list = boxes_clusters(snip[0], outputs.numpy(), 'pred')
        bcm = box2clusterMetrices(num_boxes)
        bcm.add_pred_pnts(pred_list)
        bcm.add_pred_labels(pred_labels)
        bcm.add_conf(pred_scores)
        _, grdt_labels, grdt_list = boxes_clusters(snip[0], targets.numpy(), 'grd')
        bcm.add_gt_labels(grdt_labels)
        bcm.add_gt_pnts(grdt_list)

        bcm_batch = [bcm]
        batch_obj = BatchStats(bcm_batch, iou_thres-1e-6)
        mmmCov += batch_obj.get_batch_mCOV()
    #print('sum of mean converage', mmCov, 'number of test samples',len(dataloader))
    mmmCov /= len(dataloader)
    mmmCov /= 1 # batch size
    print('mean average coverage {}'.format(mmmCov))
    return


def _create_validation_data_loader(seq_path, snpt_path, n_cpu, blurry=False, skew=False, batch_size=1):
    """
    Creates a DataLoader for validation.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = AABBDataset(
        seq_path,
        snpt_path,
        blurry=blurry,
        skew=skew)
    dataset.set_eval()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size, # to avoid decouple a batch and put it back
        shuffle=False,
        num_workers=n_cpu,
        collate_fn=dataset.collate_fn,
        pin_memory=True)
    return dataloader


def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Evaluate validation data.")
    parser.add_argument("-m", "--model", type=str, default="yolo/config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="yolo/checkpoints/yolov3_ckpt_640.pth", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-d", "--data", type=str, default="yolo/config/custom_short.data", help="Path to data config file (.data)")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Size of each image batch")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the validation more verbose")
    parser.add_argument("--img_size", type=int, default=608, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # Get data configuration
    data_config = parse_data_config(args.data)
    #train_path = data_config["train"]
    #train_snip_path = data_config["train_snip"]
    valid_path = data_config["valid"]
    valid_snip_path = data_config["valid_snip"]
    class_names = load_classes(data_config["names"]) # List of class names
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    metrics = evaluate_model_file(
        args.model,
        args.weights,
        valid_path,
        valid_snip_path,
        class_names,
        batch_size=args.batch_size,
        n_cpu=args.n_cpu,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        verbose=True)
    # print_eval_stats(metrics, class_names, verbose=True)


if __name__ == "__main__":
    run()
