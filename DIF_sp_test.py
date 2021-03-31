#The model is following the tutorial below:
#https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
#
# imports
from __future__ import print_function
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import os
import math
from PIL import Image

from prefetch_generator import BackgroundGenerator

import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from superpixel_module import SPSalientObjDataset
# usage_mode: 1: train, 0: test
# input_mode: superpixels used or not
# set_type: 0: RGB / 1: RGBD

INPUT_TYPE=1 #0=pixel, 1=superpixel
DATASET_TYPE=1 #1=RGBD, 0=RGB
STATE_TRTE=1 #1=Train, 0=Test 

#degub
DEBUG=False
start_VerTrain=1
start_EP=1

#****************************************************************VERSION
if(start_VerTrain==0):
    Ver_Train=1
    while os.path.exists("SP_modelParam_T%d/"%(Ver_Train)) or os.path.exists("SP_VisualResults_T%d/"%(Ver_Train)):
       Ver_Train+=1 #don't cover the previous log
else:
    Ver_Train=start_VerTrain

print("(superpixel) Train Version: %d"%(Ver_Train))

TRAIN_ON=True
epochs=40

TEST_ON=True

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
        

# ###############
state_train_test=1
batch_size=1

##pretraining data
'''
dataset_type = 0
rgb_pretrain_dataset = SPSalientObjDataset(state_train_test,dataset_type,debug=False,sp_update=False,create=True) #Training/RGB dataset
# pretrain_loader = DataLoaderX(rgb_pretrain_dataset, batch_size=batch_size,shuffle=True)

'''
##RGBD training data 
dataset_type = 1
train_dataset = SPSalientObjDataset(state_train_test,dataset_type,debug=False,sp_update=False,create=True) #Training/RGBD dataset
# train_loader = DataLoaderX(train_dataset, batch_size=batch_size,shuffle=True) # Debug used




'''
#testing set
state_train_test=0
for dataset_type in range(1,4):  
  test_dataset = SPSalientObjDataset(state_train_test,dataset_type,debug=False,sp_update=False,create=True) 
  # test_loader = DataLoaderX(test_dataset, batch_size=batch_size)
'''