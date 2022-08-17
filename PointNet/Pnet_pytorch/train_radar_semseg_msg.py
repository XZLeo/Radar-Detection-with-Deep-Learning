"""
Training python file for the trining a pointnet model.
It uses the RadarScenes dataset cropped into train/validation/test files
It also uses the text file with indices from snippet.py 
"""
import argparse
import os
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import copy
#Dataset and Dataloader
from PointNet.pnet_dataset import PNETDataset
from torch.utils.data import DataLoader, Subset
#for tensorboard
from torch.utils.tensorboard import SummaryWriter
#for confusionmatrix
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
#F1 score
from sklearn.metrics import f1_score
from PointNet.pnet_visualizer import plot_snippet_labels
#os.environ["CUDA_LAUNCH_BLOCKING"]="1"

#Root directory for creating/accesing the experiment model and configuration files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

#Names of the classes used in this approach
classes = ['CAR', 'PEDESTRIAN', 'PEDESTRIAN_GROUP',  
           'TWO_WHEELER', 'LARGE_VEHICLE', 'STATIC']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat
    
def data_aug(type_aug, points, target, n_points):
    '''
    Takes the value of type_aug and returns the specified
    type of data augmentation procedure
    '''
    if type_aug == "rotation":
        points = provider.rotate_point_cloud_with_normal(points) #rotation 
    elif type_aug == "jitter":
        points[:,:,] = provider.jitter_point_cloud(points[:,:,]) #Jitter on vr (dynamic) 
    elif type_aug == "scale":
        points = provider.random_scale_point_cloud(points) #scale
    elif type_aug == "shift":
        points = provider.shift_point_cloud(points) #shift #doesnt work rn
    elif type_aug == "dropout":
        points = provider.random_point_dropout(points) #dropout
    else:
        points = points
    return points 
    
    
def batch_size_exit(batch, length_dataset, name):
    '''
    Verifies the consistency of batches and dataset.
    '''
    print("The selected batch size is: ", batch, 
        ",but the number of elements in ", name," dataset is:", length_dataset)
    print("The batch size is larger than the actual dataset!")
    print("Please, select a different batch size or number of subset (if using debug mode)")
    print("EXITING...")
    sys.exit()
    
def inplace_relu(m): 
    '''
    inplace=True means that it will modify the input directly, 
    without allocating any additional output.
    '''
    classname = m.__class__.__name__    
    if classname.find('ReLU') != -1:
        m.inplace=True
        
def generate_graph(optimizer, trainDataLoader, classifier, writer):
    '''
    When called, this method generates a graph of the architecture
    in the tensorboard summary after training a model
    (Suggestion: Use it once with a small dataset once)
    '''
    dev = []
    optimizer.zero_grad()
    snippets_graph = []
    for snippets_graph, label in trainDataLoader:          
        dev = snippets_graph.device
        snippets_graph = snippets_graph.cpu().data.numpy()
        snippets_graph[:, :, :3] = provider.rotate_point_cloud_z(snippets_graph[:, :, :3])
        snippets_graph = torch.Tensor(snippets_graph)
        snippets_graph = snippets_graph.float().to(dev)
        snippets_graph = snippets_graph.transpose(2, 1)
        break  
    clsf = classifier.to(dev)
    writer.add_graph(clsf, snippets_graph) 
    pass
        
def createConfusionMatrix(y_pred, y_true, classes, type_cf):                        
    '''
    Compute confusion matrix (absolute or relative) using two pairs of vectors
    '''   
    cf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5])
    cf = cf_matrix.copy() 
    if type_cf == "abs":
        df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    elif type_cf == "percentage":
        df_cm = pd.DataFrame(np.nan_to_num(cf_matrix / np.sum(cf_matrix, axis=0)[np.newaxis, :]), 
                             index=[i for i in classes],
                             columns=[i for i in classes])
        cf = np.nan_to_num(cf / np.sum(cf, axis=1)[:, np.newaxis])
        
    # Create Heatmap
    my_dpi=250
    plt.figure(figsize=(4.5, 3.5), dpi=my_dpi)     
    plt.subplots_adjust(left=0.35,bottom=0.45)
    ax = sn.heatmap(df_cm, annot=True, annot_kws={"fontsize":6}).get_figure()
    plt.xlabel('Predicted Labels', fontsize = 8) # x-axis label with fontsize 15
    plt.xticks(rotation=45, ha="right")
    plt.ylabel('True Labels', fontsize = 8) # y-axis label with fontsize 15
    plt.tight_layout()
    return ax, cf     

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument(
        '-dt',
        '--train_dataset_path',
        type=Path,
        #default= "/home/s0001519/RadarProj/RadarScenes/train", #Local
        #default= "/RadarScenes/train", #GPU cluster
        default= "/RadarScenes/train_full", #GPU cluster (train + validation)
        help='file path to the pointnet train set'
    )
    parser.add_argument(
        '-st',
        '--train_snippet_path',
        type=Path,
        #default="/home/s0001519/RadarProj/venv/zenseact_repo/thesisdlradardetection/static/train.txt", #Local
        #default="./static/train.txt", #GPU cluster
        default="./static/train_full.txt", #GPU cluster (train + validation)
        help='file path of the start indexes (train)'
    )
    parser.add_argument(
        '-dv',
        '--valid_dataset_path',
        type=Path,
        #default= "/home/s0001519/RadarProj/RadarScenes/validation", #Local
        #default= "/RadarScenes/validation", #GPU cluster
        default= "/RadarScenes/test", #GPU cluster
        help='file path to the pointnet validation set'
    )
    parser.add_argument(
        '-sv',
        '--valid_snippet_path',
        type=Path,
        #default="/home/s0001519/RadarProj/venv/zenseact_repo/thesisdlradardetection/static/validation.txt", #Local
        #default="./static/validation.txt", #GPU cluster
        default="./static/test_500000.0.txt", #GPU cluster
        help='file path of the start indexes (validation)'
    )
    parser.add_argument('--model', type=str, default='pnet2_radar_semseg_msg', help='model name [default: pnet2_radar_semseg_msg]')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 4]')
    parser.add_argument('--epoch', default=30, type=int, help='Epoch to run [default: 30]')
    parser.add_argument('--learning_rate', default=0.0004, type=float, help='Initial learning rate [default: 0.001]') #incrs
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 3097]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.6, help='Decay rate for lr decay [default: 0.6]') #cnst
    parser.add_argument('--debug', action='store_true', help='Selects a portion of the snippets for fast testing') 
    parser.add_argument('--num_snippets_tr', type=int, default=1, help='Selects number of training snippets for subset (when debug argument is triggered)')
    parser.add_argument('--idxs_snippets_tr', type=int, default=0, help='Select training index(indices) from where the subset snippet(s) start (when debug argument is triggered)')
    parser.add_argument('--num_snippets_vl', type=int, default=1, help='Selects number of validadtion snippets for subset (when debug argument is triggered)')
    parser.add_argument('--idxs_snippets_vl', type=int, default=0, help='Select validation index(indices) from where the subset snippet(s) start (when debug argument is triggered)')
    parser.add_argument('--jitter_data', action='store_true', help='Performs jittering on the snippet when calling the dataset')
    parser.add_argument('--plot_labels', action='store_true', help='Plots the predicted labels with labels. (works with --debug)')
    parser.add_argument('--data_aug', type=str, default='None', help='Performs data augmentation techniques on data.')
    return parser.parse_args()

def main(args):
    np.set_printoptions(suppress=True)
    np.seterr(invalid='ignore')
    def log_string(str):
        logger.info(str)
        print(str)
    
    print()
    print("\033[1;32m *** Training Process Started, WELCOME! *** \n")  
    print("\033[1;0mPytorch Version: ", torch.__version__)          
            
    '''HYPER PARAMETER'''    
    print("GPU processor set(s): ", torch.cuda.get_device_name(int(args.gpu)))
    print("Number of GPUs assigned: ", torch.cuda.device_count())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path(os.path.join(ROOT_DIR, 'log'))#'./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)    
    writer = SummaryWriter(os.path.join(experiment_dir, 'tb_trn_runs_' + timestr)) #tensorboard directory
    
    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    print("")
    log_string('PARAMETERS ...')
    log_string(args)      
        
    '''RADARSCENES DATASET'''   
    print()
    print("\033[1;93mLoading Radarscenes dataset... ","\033[1;0m")
    
    NUM_CLASSES = 6
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size
    TRAIN_DATA_PATH = args.train_dataset_path
    TRAIN_SNIP_PATH = args.train_snippet_path    
    VALID_DATA_PATH = args.valid_dataset_path
    VALID_SNIP_PATH = args.valid_snippet_path       
    TRAIN_DATASET = PNETDataset(TRAIN_DATA_PATH, TRAIN_SNIP_PATH, NUM_POINT, jitter_data=args.jitter_data)
    VALID_DATASET = PNETDataset(VALID_DATA_PATH, VALID_SNIP_PATH, NUM_POINT, jitter_data=False)
    
    #getting a subset (for debugging only)     
    if (args.debug): 
        train_index_start = args.idxs_snippets_tr
        valid_index_start = args.idxs_snippets_vl
        numel_train_subset = args.num_snippets_tr#250
        numel_valid_subset = args.num_snippets_vl#30
        list_t = list(range(train_index_start, (numel_train_subset + train_index_start), 1))
        TRAIN_DATASET = Subset(TRAIN_DATASET, list_t) 
        list_v = list(range(valid_index_start, (numel_valid_subset + valid_index_start), 1))
        VALID_DATASET = Subset(VALID_DATASET, list_v)
    # number of batch needs to be greater than the actual dataset
    # otherwise, exit
    if BATCH_SIZE > len(TRAIN_DATASET):
        batch_size_exit(BATCH_SIZE, len(TRAIN_DATASET), name="training")
    if BATCH_SIZE > len(VALID_DATASET):
        batch_size_exit(BATCH_SIZE, len(VALID_DATASET), name="validation")
              
    '''DATALOADER FOR RADARSCENES DATASET'''    
    trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, 
                                                  pin_memory=True,num_workers=8, drop_last=True)
    testDataLoader = DataLoader(VALID_DATASET, batch_size=args.batch_size, shuffle=False, 
                                                 pin_memory=True,num_workers=8,drop_last=True)
    
    # shaper viewer      
    for data, label, _, _ in trainDataLoader:
        data = data[:,:,:-1] #removing range value (this for DBSCAN)
        print("Data Shape and Length:")
        print("[batch, n_points, features(x/y/vr/RCS)]")
        print("Shape of one training batch: ", data.shape, "\nShape of one label's batch:  ", label.shape)
        #print(data[:,0,:]) #[[4, 2048, 6]] (EXAMPLE)
        break    
    log_string("The number of training snippets is: %d" % len(TRAIN_DATASET))
    log_string("The number of test snippets is: %d" % len(VALID_DATASET))    
    print()
    
    #pre-weights (count the number of labels in one snippet)
    labelweights_ini = np.zeros(NUM_CLASSES)
    print("*------ Computing initial label weights ------*")
    for i, (data, label, _, _) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
        tmp, _ = np.histogram(label, range(NUM_CLASSES + 1))
        labelweights_ini += tmp
    labelweights_ini = labelweights_ini.astype(np.float32)
    he_val = np.sqrt(2.0 / np.sum(labelweights_ini))
    he_weights = np.nan_to_num(1. /(labelweights_ini * he_val), nan=0.0, posinf=0.0, neginf=0.0)
    one_over_weights = np.nan_to_num(1. / labelweights_ini, nan=0.0, posinf=0.0, neginf=0.0)
    print("inverse weights: ", one_over_weights)
    print("He weights: ", he_weights)
    labelweights_ini = (labelweights_ini / np.sum(labelweights_ini)) - 1e-6
    labelweights_ini = np.nan_to_num(np.power(np.amax(labelweights_ini) / labelweights_ini, 1 / 1.2))
    weights = torch.Tensor(he_weights).cuda() 
    #weights = torch.Tensor(labelweights_ini).cuda()
    log_string('Initial weights in train dataset: \n%s' % np.array2string(he_weights, precision=2, separator=','))    
    
    '''MODEL LOADING'''
    # if model found in the checkpoint folder:
    MODEL = importlib.import_module(args.model)   
    shutil.copy(os.path.join(ROOT_DIR, ('models/%s.py' % args.model)), str(experiment_dir))
    shutil.copy(os.path.join(ROOT_DIR, 'models/pointnet2_utils.py'), str(experiment_dir))
    
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()    
    classifier.apply(inplace_relu)
            
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            #torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.kaiming_uniform_(m.weight.data) # He initialization (1) (Best so far)
            #torch.nn.init.kaiming_normal_(m.weight.data) # He initialization (2)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
    
    # If a model is found in the checkpoint folder
    # restart the training from there:        
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print()
        log_string('STATUS: Pretrained model found')
        print("\nStarting from epoch: ",start_epoch)
    except:
        print()
        log_string('STATUS: No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)
    
    # Starting the optimizer (SGD or Adam):   
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        ) 
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    ##For tensorboard graph only, not recommended for training. Uncomment next line in debug mode)
    #generate_graph(optimizer, trainDataLoader, classifier, writer)
        
    '''MODEL TRAINING'''
    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size
    global_epoch = 0
    best_iou = 0
    
    #total number of batches    
    num_batches = len(trainDataLoader)
    num_batches_test = len(testDataLoader)
    
    #Plot variables    
    if (args.plot_labels) and (args.debug):
        xyl_plot_total = []
        xyl_plot_total_pred = []
        best_index = []
        count_index_best = 0
    
    print()
    print("\033[1;32m Training launched... ")   
    print("\033[1;0m ")
    
    for epoch in range(start_epoch, args.epoch):
        '''Train on snippet batches'''
        log_string('***** Epoch %d (%d/%s) TRAINING *****' % (global_epoch + 1, epoch + 1, args.epoch)) 
        # FORMULA: lr = lr_0 *(lr_decay ^ (epoch / step_size))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()
        
        # iterating over selcted number of snippets (batch)
        xyl_plot_gt_train = []
        for i, (points, target, _, _) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            #for every mini-batch during the training phase, we typically want to explicitly 
            # set the gradients to zero before starting to do backpropragation (i.e., updating 
            # the Weights and biases) because PyTorch accumulates the gradients on subsequent 
            # backward passes

            points = points.data.numpy()
            points = points[:,:,:-1] #range column deleted (not needed in pointnet)  
            
            if (args.plot_labels) and (args.debug):
                xygt_temp = np.delete(points,[2,3], axis=2) #no vr(2),RCS(3)
                xygt_temp = np.c_[xygt_temp[0,:,:], target[0].data.numpy()]
                xyl_plot_gt_train.append(xygt_temp)                
            
            #Random noise to each feature (vr modified only in dynamic objcts)
            points = data_aug(args.data_aug, points, target, n_points=NUM_POINT)
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            seg_pred, trans_feat = classifier(points)  #pass the points thorugh the model network
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                        
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights) #Here weights for unbalanced data
            loss.backward() # same as: x.grad += dloss/dx (for every parameter x)
            optimizer.step() #updates the value of x using the gradient x.grad (i.e.: x += -lr * x.grad)
            
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))
        
        #Logger for Tensorboard (training)
        accu_train = total_correct / float(total_seen)
        mean_loss_train = loss_sum / num_batches
        writer.add_scalar("Mean_Loss/train", mean_loss_train, epoch)
        writer.add_scalar('Accuracy/train', accu_train, epoch)
        writer.add_scalar('Learning_Rate/train', lr, epoch)
        
        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')
            
        '''Evaluate on chopped scenes'''
        xyl_plot_gt_valid = []
        with torch.no_grad():
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            
            total_predicted_class = [0 for _ in range(NUM_CLASSES)]
            batch_plain_vector = []#np.zeros(BATCH_SIZE * NUM_POINT)
            pred_plain_vector = []#np.zeros(BATCH_SIZE * NUM_POINT)
            f1s = 0
            
            classifier = classifier.train()
            
            log_string('----- EPOCH %03d VALIDATION -----' % (global_epoch + 1))
            for i, (points, target, _, _) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = points[:,:,:-1] #range column deleted (not needed in pointnet)
                
                aux_xy = np.zeros((points.shape[0],NUM_POINT,points.shape[2]))
                aux_target = np.zeros((target.shape[0],NUM_POINT))
                if (args.plot_labels) and (args.debug):
                    aux_xy = copy.deepcopy(points)
                    aux_xy = np.delete(aux_xy,[2,3], axis=2) #no vr(2),RCS(3)
                    aux_target = copy.deepcopy(target.data.numpy())
                
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights) #Here weights!
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp
                
                #Vectors for confusion matrix, and F1 score
                batch_plain_vector = np.reshape(batch_label, -1)
                pred_plain_vector = np.reshape(pred_val, -1)
                f1s = f1_score(batch_plain_vector, pred_plain_vector, average='weighted')
                
                if (args.plot_labels) and (args.debug):
                    xygt_temp = np.c_[aux_xy[0,:,:], aux_target[0]]
                    xyl_temp = np.c_[aux_xy[0,:,:], pred_val[0]]
                    xyl_plot_gt_valid.append(xygt_temp)
                    xyl_plot_total_pred.append(xyl_temp)
                                
                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_predicted_class += np.sum((pred_val == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))           
            
            #Point metrics (not clusters)
            log_string('eval mean loss: %f' % (loss_sum/ float(num_batches_test)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6)))) 
            log_string('eval F1 score: %f' % (f1s))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l] + 1e-6))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / float(num_batches_test)))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
                
                #plot
                if (args.plot_labels) and (args.debug):
                    best_index.append(global_epoch-1)
                    xyl_plot_total.append(xyl_plot_gt_train)
                    xyl_plot_total.append(xyl_plot_gt_valid)
                    
            log_string('Best mIoU: %f' % best_iou)
            
            #For Tensorbord
            eml = loss_sum / float(num_batches_test)
            epa = total_correct / float(total_seen)
            epaca = np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))
            writer.add_scalar("Mean_Loss/validation", eml, epoch)
            writer.add_scalar("Accuracy/validation", epa, epoch)
            
            writer.add_scalar("eval/point_avg_class_IoU", mIoU, epoch)            
            writer.add_scalar("eval/point_avg_class_acc", epaca, epoch)
            writer.add_scalar('eval/F1_score', f1s, epoch)
            
        global_epoch += 1
        
        #For Tensorbord
        ax1, cf1 = createConfusionMatrix(pred_plain_vector, batch_plain_vector, classes, type_cf="abs")
        ax2, cf2 = createConfusionMatrix(pred_plain_vector, batch_plain_vector, classes, type_cf="percentage")
        log_string('\nConf_matrix (absolute): \n%s' % np.array2string(cf1, precision=2, separator=','))
        log_string('Conf_matrix (Percentage): \n%s' % np.array2string(cf2, precision=2, separator=','))
        log_string('\n\n')
        writer.add_figure("Confusion Matrix (Absolute values)", ax1, epoch)
        writer.add_figure("Confusion Matrix (Percentage)", ax2, epoch)        
    writer.close()
    
    if (args.plot_labels) and (args.debug):
        #xyl_plot_total[gt_train/gt_valid/pred_valid][num_snippet]
        ax = plt1.axes()
        plot_snippet_labels(xyl_plot_total[0][0], NUM_CLASSES, legend_str="GT train", 
                            fig_num=0, plot_filter=False, ax_plot=ax)
        plot_snippet_labels(xyl_plot_total[1][0], NUM_CLASSES, legend_str="GT valid", 
                            fig_num=1, plot_filter=False, ax_plot=ax)
        #plot predictions
        count = 2
        #for ll in range(len(xyl_plot_total_pred)):
        for ll in best_index:
            plot_snippet_labels(xyl_plot_total_pred[ll], NUM_CLASSES, legend_str="Pred valid "+str(ll), 
                                    fig_num=count, plot_filter=False, ax_plot=ax)
            count += 1
        plt1.show()
    log_string('\n\n\n\n')
    log_string('  ***********************************  ')    
    log_string('\n\n\n\n')
        

if __name__ == '__main__':
    args = parse_args()
    main(args)        
    