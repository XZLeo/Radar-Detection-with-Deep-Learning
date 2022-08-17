'''
Adapted from https://github.com/XZLeo/PyTorch-YOLOv3/blob/master/pytorchyolo/train.py
'''
#! /usr/bin/env python3
from __future__ import division

import os
import argparse
import tqdm
import time

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from terminaltables import AsciiTable
from torchsummary import summary

from yolo.grid_dataset import AABBDataset
from yolo.utils.models import load_model
from yolo.utils.logger import Logger
from yolo.utils.utils import to_cpu, load_classes, print_environment_info, provide_determinism, worker_seed_set
from yolo.utils.parse_config import parse_data_config
from yolo.utils.loss import compute_loss
from yolo.pointwiseTest import _evaluate, _create_validation_data_loader


def _create_data_loader(seq_path, snpt_path, blurry, skew, batch_size, n_cpu, aug, rotate, mirror):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = AABBDataset(
        seq_path,
        snpt_path, 
        blurry=blurry,
        skew=skew,
        aug_flg=aug,
        rot_flg=rotate,
        mirr_flg=mirror) #MyRotateTransform
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader


def run():
   
    print_environment_info()

    print(f"Command line arguments: {args}")

    if args.seed != -1:
        provide_determinism(args.seed)

    logger = Logger(args.logdir)  # Tensorboard logger

    # Create output directories if missing
    time_pth = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    os.makedirs("yolo/output", exist_ok=True)
    os.makedirs("yolo/checkpoints", exist_ok=True)
    os.makedirs("yolo/checkpoints/{}".format(time_pth), exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(args.data)
    train_path = data_config["train"]
    train_snip_path = data_config["train_snip"]
    valid_path = data_config["valid"]
    valid_snip_path = data_config["valid_snip"]
    class_names = load_classes(data_config["names"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ############
    # Create model
    # ############
    if args.pretrained_weights:
        model = load_model(args.model, args.pretrained_weights)
    else:
        model = load_model(args.model)

    # Print model
    if args.verbose:
        summary(model, input_size=(
            3, model.hyperparams['height'], model.hyperparams['height']))

    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']

    # #################
    # Create Dataloader
    # #################

    # Load training dataloader
    dataloader = _create_data_loader(
        train_path,
        train_snip_path,
        args.blurry,
        args.skew,
        mini_batch_size,
        args.n_cpu,
        args.augmentation,
        args.rotate,
        args.mirror)
    # Load validation dataloader
    validation_dataloader = _create_validation_data_loader(
        valid_path,
        valid_snip_path,
        args.blurry,
        args.skew,
        mini_batch_size,
        args.n_cpu)

    # ################
    # Create optimizer
    # ################

    params = [p for p in model.parameters() if p.requires_grad]

    if (model.hyperparams['optimizer'] in [None, "adam"]):
        optimizer = optim.Adam(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
        )
    elif (model.hyperparams['optimizer'] == "sgd"):
        optimizer = optim.SGD(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
    # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
    # instead of: 0, 10, 20
    for epoch in range(1, args.epochs+1):

        print("\n---- Training Model ----")

        model.train()  # Set model to training mode
       
        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            if not len(targets):# avoid empty batch
                continue
            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            outputs = model(imgs)
       
            loss, loss_components = compute_loss(outputs, targets, model)

            loss.backward()

            ###############
            # Run optimizer
            ###############

            if batches_done % model.hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model.hyperparams['learning_rate']
                if batches_done < model.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                        # if batches done > 4e5, shirink lr by 0.1;
                        # batches done > 4.5e5, shirink lr by 0.1 again 
                            lr *= value
                # Log the learning rate>
                logger.scalar_summary("train/learning_rate", lr, batches_done)
                # Set learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lr

                # Run optimizer
                optimizer.step()
                # Reset gradients
                optimizer.zero_grad()

            # ############
            # Log progress
            # ############
            if args.verbose:
                print(AsciiTable(
                    [
                        ["Type", "Value"],
                        ["IoU loss", float(loss_components[0])],
                        ["Object loss", float(loss_components[1])],
                        ["Class loss", float(loss_components[2])],
                        ["Loss", float(loss_components[3])],
                        ["Batch loss", to_cpu(loss).item()],
                    ]).table)

            # Tensorboard logging
            tensorboard_log = [
                ("train/iou_loss", float(loss_components[0])),
                ("train/obj_loss", float(loss_components[1])),
                ("train/class_loss", float(loss_components[2])),
                ("train/loss", to_cpu(loss).item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model.seen += imgs.size(0)

        # #############
        # Save progress
        # #############
        # Save model to checkpoint file
        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = f"yolo/checkpoints/{time_pth+'_'+args.checkpoint_path}/yolov3_ckpt_{epoch}.pth"
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            torch.save(model.state_dict(), checkpoint_path)

        # ########
        # Evaluate
        # ########
        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = _evaluate(
                model,
                validation_dataloader,
                class_names,
                img_size=model.hyperparams['height'],
                iou_thres=args.iou_thres,
                conf_thres=args.conf_thres,
                nms_thres=args.nms_thres,
                verbose=args.verbose
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    ("validation/f1", f1.mean())]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description="Trains the YOLO model.")
    parser.add_argument(
        "-m", "--model", type=str,
        default="./yolo/config/yolov3.cfg",
        help="Path to model definition file (.cfg)")
    parser.add_argument(
        "-d", "--data", type=str,
        default="yolo/config/custom_short.data",
        help="Path to data config file (.data)")
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument(
        "-e", "--epochs", type=int,
        default=300, help="Number of epochs")
    parser.add_argument(
        "-v", "--verbose",
        action='store_true',
        help="Makes the training more verbose")
    parser.add_argument(
        "--n_cpu", type=int, default=8,
        help="Number of cpu threads to use during batch generation")
    parser.add_argument(
        "--checkpoint_interval", type=int,
        default=100,
        help="Interval of epochs between saving model weights")
    parser.add_argument(
        "--checkpoint_path", type=str,
        help="the folder to save model weights")
    parser.add_argument(
        "--evaluation_interval", type=int,
        default=1,
        help="Interval of epochs between evaluations on validation set") 
    parser.add_argument(
        "--iou_thres",
        type=float, default=0.5,
        help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument(
        "--conf_thres",
        type=float, default=0.1,
        help="Evaluation: Object confidence threshold")
    parser.add_argument(
        "--nms_thres",
        type=float, default=0.5,
        help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument(
        "--logdir",
        type=str, default="yolo/logs",
        help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument(
        "--seed",
        type=int, default=-1,
        help="Makes results reproducable. Set -1 to disable.")
    parser.add_argument(
        "--blurry",
        action='store_true',
        help="add blurry filter while generating gridMaps")
    parser.add_argument(
        "--skew",
        action='store_true',
        help="skew speed distribution while generating gridMaps")
    parser.add_argument(
        "--number_cell",
        type=int, default=608,
        help="number of cells in the gridmap")
    parser.add_argument(
        '-a',
        '--augmentation',
        action='store_true'
    )
    parser.add_argument(
        '-r',
        '--rotate',
        type=int,
        default=0,
        help='0 do nothing, 1 for rotate the snippet for 30, 2 for rotate randomly'
    )
    parser.add_argument(
        '-mr',
        '--mirror',
        type=str,
        help='flip a snippet horizontally or vertically or both'
    )
    args = parser.parse_args()
    run()
