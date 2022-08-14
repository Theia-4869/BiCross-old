import os, errno
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from glob import glob
from PIL import Image
from torchvision import transforms

from BiCross.Loss import *



def create_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def get_total_paths(path, ext):
    return glob(os.path.join(path, '*'+ext))

def get_splitted_dataset(config, split, dataset_name, path_images, path_depths, path_segmentation):
    list_files = [os.path.basename(im) for im in path_images]
    np.random.seed(config['general']['seed'])
    np.random.shuffle(list_files)
    if split == 'train':
        selected_files = list_files[:int(len(list_files)*config['Dataset']['splits']['split_train'])]
    elif split == 'val':
        selected_files = list_files[int(len(list_files)*config['Dataset']['splits']['split_train']):int(len(list_files)*config['Dataset']['splits']['split_train'])+int(len(list_files)*config['Dataset']['splits']['split_val'])]
    else:
        selected_files = list_files[int(len(list_files)*config['Dataset']['splits']['split_train'])+int(len(list_files)*config['Dataset']['splits']['split_val']):]

    path_images = [os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_images'], im[:-4]+config['Dataset']['extensions']['ext_images']) for im in selected_files]
    path_depths = [os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_depths'], im[:-4]+config['Dataset']['extensions']['ext_depths']) for im in selected_files]
    path_segmentation = [os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_segmentations'], im[:-4]+config['Dataset']['extensions']['ext_segmentations']) for im in selected_files]
    return path_images, path_depths, path_segmentation

def get_losses(config):
    def NoneFunction(a, b):
        return 0
    loss_depth = NoneFunction
    loss_uncertainty = NoneFunction
    loss_contrastive = NoneFunction

    if config['general']['loss_depth'] == 'mse':
        loss_depth = nn.MSELoss()
    elif config['general']['loss_depth'] == 'ssi':
        loss_depth = ScaleAndShiftInvariantLoss()
    elif config['general']['loss_depth'] == 'sig':
        loss_depth = SigLoss()

    if config['general']['loss_uncertainty'] == 'l1':
        loss_uncertainty = Uncertainty_L1_Loss()
    elif config['general']['loss_uncertainty'] == 'l2':
        loss_uncertainty = Uncertainty_L2_Loss()
    elif config['general']['loss_uncertainty'] == 'l1log':
        loss_uncertainty = Uncertainty_L1log_Loss()

    loss_contrastive = ContrastiveLoss()
    
    return loss_depth, loss_uncertainty, loss_contrastive

def get_optimizer(config, net):
    names = set([name.split('.')[0] for name, _ in net.named_modules()]) - set(['', 'transformer_encoders'])
    # params_backbone = net.transformer_encoders.parameters()
    params_backbone = net.pretrained.parameters()
    params_scratch = list()
    for name in names:
        params_scratch += list(eval("net."+name).parameters())

    if config['general']['optim'] == 'adam':
        optimizer_backbone = optim.Adam(params_backbone, lr=config['general']['lr_backbone'])
        optimizer_scratch = optim.Adam(params_scratch, lr=config['general']['lr_scratch'])
    elif config['general']['optim'] == 'sgd':
        optimizer_backbone = optim.SGD(params_backbone, lr=config['general']['lr_backbone'], momentum=config['general']['momentum'])
        optimizer_scratch = optim.SGD(params_scratch, lr=config['general']['lr_scratch'], momentum=config['general']['momentum'])
    return optimizer_backbone, optimizer_scratch

def get_schedulers(optimizers):
    return [ReduceLROnPlateau(optimizer) for optimizer in optimizers]
