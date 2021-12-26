import os
import random

import torch
import torch.nn as nn
import numpy as np

import model
from dataload_with_alb  import DiseaseDataset
from torch.utils.data import DataLoader

from warmup_scheduler import GradualWarmupScheduler
from adabelief_pytorch import AdaBelief

import wandb
from config import *
from train import *
from torchsampler import ImbalancedDatasetSampler



def run():
    #########Random seed 고정해주기###########
    random_seed = 0 #3407
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    ###########################################

    batch_size= 4

    ##########################################데이터 로드 하기#################################################
    data_dir = os.path.join(os.getcwd(), "RSNA_COVID_png_512") #train, val 폴더가 들어있는 경로
    classes_name = os.listdir(os.path.join(data_dir, 'train')) #폴더에 들어있는 클래스명
    num_classes =  len(os.listdir(os.path.join(data_dir, 'train'))) #train 폴더 안에 클래스 개수 만큼의 폴더가 있음

    datasets = {x: DiseaseDataset(data_dir=os.path.join(data_dir, x), img_size=512, bit=8, 
                num_classes=num_classes, classes_name=classes_name, data_type='img', mode= x) for x in ['train', 'val']}


    dataloaders = {x: DataLoader(datasets[x], sampler=ImbalancedDatasetSampler(datasets[x]), batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    num_iteration = {x: np.ceil(dataset_sizes[x] / batch_size) for x in ['train', 'val']}
    #############################################################################################################################

    
    net = model.ResNet50(img_channel=1, num_classes=num_classes).to(device) 

    criterion = nn.CrossEntropyLoss() #loss 형태 정해주기

    optimizer_ft = AdaBelief(net.parameters(), lr=1e-3, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = True)
    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_ft, mode='min', factor=0.5, patience=10)

    scheduler_lr = GradualWarmupScheduler(optimizer_ft, multiplier=1, total_epoch=5, after_scheduler=scheduler_lr)

    ########################################################################################
    patience = 6
    
    train(dataloaders, dataset_sizes, num_iteration, net, criterion, optimizer_ft, scheduler_lr, classes_name, patience, num_epoch=5)

if __name__ == '__main__':
    run()