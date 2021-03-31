'''
network construction
# #https://zhuanlan.zhihu.com/p/64990232
'''
from __future__ import print_function
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
import torchvision
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
# sp pool
from suppixpool_layer import AveSupPixPool, SupPixUnpool
pool = AveSupPixPool()
unpool = SupPixUnpool()

from skimage.segmentation import slic
from skimage import io, color
from slic import RGBD_SLICProcessor
# shared config
from config import Config
cfg=Config()

STAT_TRAIN=1
STAT_TEST=0
DEBUG=False

# ==weights init==
def weight_init(model):
    print("==model parameters initialization==")
    #VGG-16 parameters initialization
    # print( ( models.vgg16().state_dict()['features.0.weight'] ).shape )
    # RGB FCN parameters init
    model.state_dict()['rgb_conv1_1.weight']= models.vgg16().state_dict()['features.0.weight'] 
    model.state_dict()['rgb_conv1_1.bias']= models.vgg16().state_dict()['features.0.bias'] 
    model.state_dict()['rgb_conv1_2.weight']= models.vgg16().state_dict()['features.2.weight'] 
    model.state_dict()['rgb_conv1_2.bias']= models.vgg16().state_dict()['features.2.bias'] 
        
    model.state_dict()['rgb_conv2_1.weight']= models.vgg16().state_dict()['features.5.weight'] 
    model.state_dict()['rgb_conv2_1.bias']= models.vgg16().state_dict()['features.5.bias'] 
    model.state_dict()['rgb_conv2_2.weight']= models.vgg16().state_dict()['features.7.weight'] 
    model.state_dict()['rgb_conv2_2.bias']= models.vgg16().state_dict()['features.7.bias'] 
        
    model.state_dict()['rgb_conv3_1.weight']= models.vgg16().state_dict()['features.10.weight'] 
    model.state_dict()['rgb_conv3_1.bias']= models.vgg16().state_dict()['features.10.bias'] 
    model.state_dict()['rgb_conv3_2.weight']= models.vgg16().state_dict()['features.12.weight'] 
    model.state_dict()['rgb_conv3_2.bias']= models.vgg16().state_dict()['features.12.bias'] 
    model.state_dict()['rgb_conv3_3.weight']= models.vgg16().state_dict()['features.14.weight'] 
    model.state_dict()['rgb_conv3_3.bias']= models.vgg16().state_dict()['features.14.bias'] 
        
    model.state_dict()['rgb_conv4_1.weight']= models.vgg16().state_dict()['features.17.weight'] 
    model.state_dict()['rgb_conv4_1.bias']= models.vgg16().state_dict()['features.17.bias'] 
    model.state_dict()['rgb_conv4_2.weight']= models.vgg16().state_dict()['features.19.weight'] 
    model.state_dict()['rgb_conv4_2.bias']= models.vgg16().state_dict()['features.19.bias'] 
    model.state_dict()['rgb_conv4_3.weight']= models.vgg16().state_dict()['features.21.weight'] 
    model.state_dict()['rgb_conv4_3.bias']= models.vgg16().state_dict()['features.21.bias'] 
        
    model.state_dict()['rgb_conv5_1.weight']= models.vgg16().state_dict()['features.24.weight'] 
    model.state_dict()['rgb_conv5_1.bias']= models.vgg16().state_dict()['features.24.bias'] 
    model.state_dict()['rgb_conv5_2.weight']= models.vgg16().state_dict()['features.26.weight'] 
    model.state_dict()['rgb_conv5_2.bias']= models.vgg16().state_dict()['features.26.bias'] 
    model.state_dict()['rgb_conv5_3.weight']= models.vgg16().state_dict()['features.28.weight'] 
    model.state_dict()['rgb_conv5_3.bias']= models.vgg16().state_dict()['features.28.bias']
    
    #Depth FCN parameters init
    model.state_dict()['depth_conv1_1.weight']= models.vgg16().state_dict()['features.0.weight'] 
    model.state_dict()['depth_conv1_1.bias']= models.vgg16().state_dict()['features.0.bias'] 
    model.state_dict()['depth_conv1_2.weight']= models.vgg16().state_dict()['features.2.weight'] 
    model.state_dict()['depth_conv1_2.bias']= models.vgg16().state_dict()['features.2.bias'] 
        
    model.state_dict()['depth_conv2_1.weight']= models.vgg16().state_dict()['features.5.weight'] 
    model.state_dict()['depth_conv2_1.bias']= models.vgg16().state_dict()['features.5.bias'] 
    model.state_dict()['depth_conv2_2.weight']= models.vgg16().state_dict()['features.7.weight'] 
    model.state_dict()['depth_conv2_2.bias']= models.vgg16().state_dict()['features.7.bias'] 
        
    model.state_dict()['depth_conv3_1.weight']= models.vgg16().state_dict()['features.10.weight'] 
    model.state_dict()['depth_conv3_1.bias']= models.vgg16().state_dict()['features.10.bias'] 
    model.state_dict()['depth_conv3_2.weight']= models.vgg16().state_dict()['features.12.weight'] 
    model.state_dict()['depth_conv3_2.bias']= models.vgg16().state_dict()['features.12.bias'] 
    model.state_dict()['depth_conv3_3.weight']= models.vgg16().state_dict()['features.14.weight'] 
    model.state_dict()['depth_conv3_3.bias']= models.vgg16().state_dict()['features.14.bias'] 
        
    model.state_dict()['depth_conv4_1.weight']= models.vgg16().state_dict()['features.17.weight'] 
    model.state_dict()['depth_conv4_1.bias']= models.vgg16().state_dict()['features.17.bias'] 
    model.state_dict()['depth_conv4_2.weight']= models.vgg16().state_dict()['features.19.weight'] 
    model.state_dict()['depth_conv4_2.bias']= models.vgg16().state_dict()['features.19.bias'] 
    model.state_dict()['depth_conv4_3.weight']= models.vgg16().state_dict()['features.21.weight'] 
    model.state_dict()['depth_conv4_3.bias']= models.vgg16().state_dict()['features.21.bias'] 
        
    model.state_dict()['depth_conv5_1.weight']= models.vgg16().state_dict()['features.24.weight'] 
    model.state_dict()['depth_conv5_1.bias']= models.vgg16().state_dict()['features.24.bias'] 
    model.state_dict()['depth_conv5_2.weight']= models.vgg16().state_dict()['features.26.weight'] 
    model.state_dict()['depth_conv5_2.bias']= models.vgg16().state_dict()['features.26.bias'] 
    model.state_dict()['depth_conv5_3.weight']= models.vgg16().state_dict()['features.28.weight'] 
    model.state_dict()['depth_conv5_3.bias']= models.vgg16().state_dict()['features.28.bias'] 
        
    print("==model parameters initialization end==")
    
'''
networks
'''
# pixel net (base) ===============================================================================
class Net(nn.Module):
    def __init__(self, gpu_device=torch.device("cpu"),start_EP=0):
        super(Net, self).__init__()
        self.vgg_init=True
        self.device=gpu_device
        self.epoch=start_EP
        self.fcn_2d()

    def RGB_fcn_2d(self):    
        # RGB part
        self.rgb_conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.rgb_conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.rgb_pool_1 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.rgb_conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.rgb_pool_2 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.rgb_conv3_2 = nn.Conv2d(256, 256, 3, padding=1)  
        self.rgb_conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.rgb_pool_3 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.rgb_conv4_2 = nn.Conv2d(512, 512, 3, padding=1) 
        self.rgb_conv4_3 = nn.Conv2d(512, 512, 3, padding=1)    
        self.rgb_pool_4 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.rgb_conv5_2 = nn.Conv2d(512, 512, 3, padding=1)   
        self.rgb_conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.rgb_pool_5 =  nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)    

    def depth_fcn_2d(self):    
        # Depth part
        self.depth_conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.depth_conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.depth_pool_1 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.depth_conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.depth_pool_2 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.depth_conv3_2 = nn.Conv2d(256, 256, 3, padding=1)   
        self.depth_conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.depth_pool_3 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.depth_conv4_2 = nn.Conv2d(512, 512, 3, padding=1)   
        self.depth_conv4_3 = nn.Conv2d(512, 512, 3, padding=1)    
        self.depth_pool_4 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.depth_conv5_2 = nn.Conv2d(512, 512, 3, padding=1)  
        self.depth_conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.depth_pool_5 =  nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False) 
    
    def fcn_2d(self):
        self.RGB_fcn_2d()
        self.depth_fcn_2d()
        
        if(self.vgg_init and self.epoch==0):
          weight_init(self)
        if(DEBUG):
          print(self.state_dict())
        
        # short connection convolutions #RGBD #RGB (input channels are different)       
        # self.b1_conv1_rgb = nn.Conv2d(64, 128, 3, padding=1) #no depth features to concat
        self.b1_conv1 = nn.Conv2d(128, 128, 3, padding=1) #has depth features to concat
        self.b1_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.b1_conv3 = nn.Conv2d(128, 1, 1)
        
        # self.b2_conv1_rgb = nn.Conv2d(128, 128, 3, padding=1)
        self.b2_conv1 = nn.Conv2d(256, 128, 3, padding=1)
        self.b2_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.b2_conv3 = nn.Conv2d(128, 1, 1)
        
        # self.b3_conv1_rgb = nn.Conv2d(256, 256, 5, padding=1)
        self.b3_conv1 = nn.Conv2d(512, 256, 5, padding=2)
        self.b3_conv2 = nn.Conv2d(256, 256, 5, padding=2)
        self.b3_conv3 = nn.Conv2d(256, 1, 1)
        
        # self.b4_conv1_rgb = nn.Conv2d(512, 256, 5, padding=1)
        self.b4_conv1 = nn.Conv2d(1024, 256, 5, padding=2)
        self.b4_conv2 = nn.Conv2d(256, 256, 5, padding=2)
        self.b4_conv3 = nn.Conv2d(256, 1, 1)
        
        # self.b5_conv1_rgb = nn.Conv2d(512, 512, 5, padding=1)
        self.b5_conv1 = nn.Conv2d(1024, 512, 5, padding=2)
        self.b5_conv2 = nn.Conv2d(512, 512, 5, padding=2)
        self.b5_conv3 = nn.Conv2d(512, 1, 1)
        
        # self.b6_conv1_rgb = nn.Conv2d(512, 512, 7, padding=1)
        self.b6_conv1 = nn.Conv2d(1024, 512, 7, padding=3)
        self.b6_conv2 = nn.Conv2d(512, 512, 7, padding=3)
        self.b6_conv3 = nn.Conv2d(512, 1, 1)
        
        # upsampling
        self.up_t64=nn.ConvTranspose2d(1, 1, kernel_size=(64, 64), stride=(32, 32), padding=(16, 16))
        self.up_t32=nn.ConvTranspose2d(1, 1, kernel_size=(32, 32), stride=(16, 16), padding=(8, 8))
        self.up_t16=nn.ConvTranspose2d(1, 1, kernel_size=(16, 16), stride=(8, 8), padding=(4, 4)) 
        self.up_t8=nn.ConvTranspose2d(1, 1, kernel_size=(8, 8), stride=(4, 4), padding=(2,  2))
        self.up_t4=nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1,  1))
        
        # sc branches to 1 channel conv
        self.b4_sc=nn.Conv2d(3, 1, 3, padding=1)
        self.b3_sc=nn.Conv2d(3, 1, 3, padding=1)
        self.b2_sc=nn.Conv2d(5, 1, 3, padding=1)
        self.b1_sc=nn.Conv2d(5, 1, 3, padding=1)
        
        # final fusion conv
        self.fusion_conv = nn.Conv2d(6, 1, 1, padding=1)
 
<<<<<<< HEAD
    def forward(self, input_rgb, input_depth, state_trte, rgb_path="", d_path="", d_spx=[]):  #### fcn_2d ####
=======
    def forward(self, input_rgb, input_depth, state_trte, rgb_path="", d_path=""):  #### fcn_2d ####
>>>>>>> dd86c05b6b13e46c37a8b50f8b4b326dda88ba1a
        # Max pooling over a (2, 2) window
        # activation function: Relu or Sigmoid (Later)        
        #sequential: https://www.itread01.com/content/1547079864.html
        #GPUs
        device=self.device
        input_rgb=input_rgb.to(device).float()
        dataset_type=1
        
        ### RGB ##########################################
        rgb_conv1_1 = F.relu(self.rgb_conv1_1(input_rgb), inplace=True).to(device)
        rgb_conv1_2 = F.relu(self.rgb_conv1_2(rgb_conv1_1), inplace=True).to(device)
        rgb_pool1 = self.rgb_pool_1(rgb_conv1_2).to(device)
        
        rgb_conv2_1 = F.relu(self.rgb_conv2_1(rgb_pool1), inplace=True).to(device)
        rgb_conv2_2 = F.relu(self.rgb_conv2_2(rgb_conv2_1), inplace=True).to(device)
        rgb_pool2 = self.rgb_pool_2(rgb_conv2_2).to(device)
        
        rgb_conv3_1 = F.relu(self.rgb_conv3_1(rgb_pool2), inplace=True).to(device)
        rgb_conv3_2 = F.relu(self.rgb_conv3_2(rgb_conv3_1), inplace=True).to(device)
        rgb_conv3_3 = F.relu(self.rgb_conv3_3(rgb_conv3_2), inplace=True).to(device)
        rgb_pool3 = self.rgb_pool_3(rgb_conv3_3).to(device)    
        
        rgb_conv4_1 = F.relu(self.rgb_conv4_1(rgb_pool3), inplace=True).to(device)
        rgb_conv4_2 = F.relu(self.rgb_conv4_2(rgb_conv4_1), inplace=True).to(device)
        rgb_conv4_3 = F.relu(self.rgb_conv4_3(rgb_conv4_2), inplace=True).to(device)
        rgb_pool4 = self.rgb_pool_4(rgb_conv4_3).to(device) 
        
        rgb_conv5_1 = F.relu(self.rgb_conv5_1(rgb_pool4), inplace=True).to(device)
        rgb_conv5_2 = F.relu(self.rgb_conv5_2(rgb_conv5_1), inplace=True).to(device)
        rgb_conv5_3 = F.relu(self.rgb_conv5_3(rgb_conv5_2), inplace=True).to(device)
        rgb_pool5 = self.rgb_pool_2(rgb_conv5_3).to(device) 

        # rgb branches
        b1_rgb = rgb_conv1_2 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu1_2')
        b2_rgb = rgb_conv2_2 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu2_2')
        b3_rgb = rgb_conv3_3 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu3_3')
        b4_rgb = rgb_conv4_3 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu4_3')
        b5_rgb = rgb_conv5_3 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu5_3')
        b6_rgb = rgb_pool5 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_pool5')

        b1_concat = b1_rgb
        b2_concat = b2_rgb
        b3_concat = b3_rgb
        b4_concat = b4_rgb
        b5_concat = b5_rgb
        b6_concat = b6_rgb

        
        if(dataset_type):
            #GPUs
            input_depth=input_depth.to(device).float()
            ### DEPTH ##########################################
            depth_conv1_1 = F.relu(self.depth_conv1_1(input_depth), inplace=True).to(device)
            depth_conv1_2 = F.relu(self.depth_conv1_2(depth_conv1_1), inplace=True).to(device)
            depth_pool1 = self.depth_pool_1(depth_conv1_2).to(device)
            
            depth_conv2_1 = F.relu(self.depth_conv2_1(depth_pool1), inplace=True).to(device)
            depth_conv2_2 = F.relu(self.depth_conv2_2(depth_conv2_1), inplace=True).to(device)
            depth_pool2 = self.depth_pool_2(depth_conv2_2).to(device)    
            
            depth_conv3_1 = F.relu(self.depth_conv3_1(depth_pool2), inplace=True).to(device)
            depth_conv3_2 = F.relu(self.depth_conv3_2(depth_conv3_1), inplace=True).to(device)
            depth_conv3_3 = F.relu(self.depth_conv3_3(depth_conv3_2), inplace=True).to(device)
            depth_pool3 = self.depth_pool_3(depth_conv3_3).to(device)     
            
            depth_conv4_1 = F.relu(self.depth_conv4_1(depth_pool3), inplace=True).to(device)
            depth_conv4_2 = F.relu(self.depth_conv4_2(depth_conv4_1), inplace=True).to(device)
            depth_conv4_3 = F.relu(self.depth_conv4_3(depth_conv4_2), inplace=True).to(device)
            depth_pool4 = self.depth_pool_4(depth_conv4_3).to(device) 
            
            depth_conv5_1 = F.relu(self.depth_conv5_1(depth_pool4), inplace=True).to(device)
            depth_conv5_2 = F.relu(self.depth_conv5_2(depth_conv5_1), inplace=True).to(device)
            depth_conv5_3 = F.relu(self.depth_conv5_3(depth_conv5_2), inplace=True).to(device)
            depth_pool5 = self.depth_pool_2(depth_conv5_3).to(device).to(device) 
    
            # depth branches
            b1_depth=depth_conv1_2 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu1_2')
            b2_depth=depth_conv2_2 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu2_2')
            b3_depth=depth_conv3_3 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu3_3')
            b4_depth=depth_conv4_3 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu4_3')
            b5_depth=depth_conv5_3 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu5_3')
            b6_depth=depth_pool5 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_dpool5')

            ################################
            # short connection: concatenate
            # concat the axis = 1  dim=[0, "1", 2, 3]
            ################################
            b1_concat=torch.cat((b1_rgb,b1_depth),1).to(device)
            b2_concat=torch.cat((b2_rgb,b2_depth),1).to(device)
            b3_concat=torch.cat((b3_rgb,b3_depth),1).to(device)
            b4_concat=torch.cat((b4_rgb,b4_depth),1).to(device)
            b5_concat=torch.cat((b5_rgb,b5_depth),1).to(device)
            b6_concat=torch.cat((b6_rgb,b6_depth),1).to(device)
            
            ################################
            # short connection: to side activation
            ################################
            b1_conv1 = F.relu(self.b1_conv1(b1_concat), inplace=True).to(device)
            b2_conv1 = F.relu(self.b2_conv1(b2_concat), inplace=True).to(device)
            b3_conv1 = F.relu(self.b3_conv1(b3_concat), inplace=True).to(device)
            b4_conv1 = F.relu(self.b4_conv1(b4_concat), inplace=True).to(device)
            b5_conv1 = F.relu(self.b5_conv1(b5_concat), inplace=True).to(device)
            b6_conv1 = F.relu(self.b6_conv1(b6_concat), inplace=True).to(device)
        else:
            b1_conv1 = F.relu(self.b1_conv1_rgb(b1_concat), inplace=True).to(device)
            b2_conv1 = F.relu(self.b2_conv1_rgb(b2_concat), inplace=True).to(device)
            b3_conv1 = F.relu(self.b3_conv1_rgb(b3_concat), inplace=True).to(device)
            b4_conv1 = F.relu(self.b4_conv1_rgb(b4_concat), inplace=True).to(device)
            b5_conv1 = F.relu(self.b5_conv1_rgb(b5_concat), inplace=True).to(device)
            b6_conv1 = F.relu(self.b6_conv1_rgb(b6_concat), inplace=True).to(device)
        
        #branch 1
        b1_conv2 = F.relu(self.b1_conv2(b1_conv1), inplace=True).to(device)
        b1_conv3 = self.b1_conv3(b1_conv2).to(device)
        #branch 2
        b2_conv2 = F.relu(self.b2_conv2(b2_conv1), inplace=True).to(device)
        b2_conv3 = self.b2_conv3(b2_conv2).to(device)
        #branch 3
        b3_conv2 = F.relu(self.b3_conv2(b3_conv1), inplace=True).to(device)
        b3_conv3 = self.b3_conv3(b3_conv2).to(device)
        #branch 4
        b4_conv2 = F.relu(self.b4_conv2(b4_conv1), inplace=True).to(device)
        b4_conv3 = self.b4_conv3(b4_conv2).to(device)
        #branch 5
        b5_conv2 = F.relu(self.b5_conv2(b5_conv1), inplace=True).to(device)
        b5_conv3 = self.b5_conv3(b5_conv2).to(device)
        #branch 6
        b6_conv2 = F.relu(self.b6_conv2(b6_conv1), inplace=True).to(device)
        b6_conv3 = self.b6_conv3(b6_conv2).to(device)
        

            

        ################################
        # short connection: branch results
        ################################        
        # shortconnections to Aside 6
        r6_conv1 = b6_conv3
        # Rside 6
        
        # shortconnections to Aside 5
        r5_conv1 = b5_conv3
        # Rside 5
        
        # shortconnections to Aside 4
        # resize_r4=nn.ConvTranspose2d(1, 1, k * 2, k, k // 2)
        r5_conv1_resize_r4 = self.up_t4(r5_conv1).to(device) #F.interpolate(r5_conv1, [b4_conv3.shape[3],b4_conv3.shape[2]])
        r6_conv1_resize_r4 = self.up_t8(r6_conv1).to(device) #F.interpolate(r6_conv1, [b4_conv3.shape[3],b4_conv3.shape[2]]).to(device)
        #print(r5_conv1_resize_r4.shape) #1, 1, 72, 72
        #print(r6_conv1_resize_r4.shape) #1, 1, 48, 48
        #print(b4_conv3.shape) #1, 1, 76, 76
        r4_concat = torch.cat((b4_conv3, r5_conv1_resize_r4, r6_conv1_resize_r4),1).to(device)
        r4_conv1 = self.b4_sc(r4_concat).to(device)
        # Rside 4
        # shortconnections to Aside 3
        # resize_r3=nn.ConvTranspose2d(1, 1, k * 2, k, k // 2)
        r5_conv1_resize_r3 = self.up_t8(r5_conv1).to(device) #F.interpolate(r5_conv1, [b3_conv3.shape[3],b3_conv3.shape[2]]).to(device)
        r6_conv1_resize_r3 = self.up_t16(r6_conv1).to(device) #F.interpolate(r6_conv1, [b3_conv3.shape[3],b3_conv3.shape[2]]).to(device)
        r3_concat = torch.cat((b3_conv3, r5_conv1_resize_r3, r6_conv1_resize_r3),1).to(device)
        r3_conv1 = self.b3_sc(r3_concat).to(device)
        # Rside 3
        # short connections to Aside 2
        # resize_r2=nn.ConvTranspose2d(1, 1, k * 2, k, k // 2)
        r3_conv1_resize_r2 = self.up_t4(r3_conv1).to(device) #F.interpolate(r3_conv1, [b2_conv3.shape[3],b2_conv3.shape[2]]).to(device)
        r4_conv1_resize_r2 = self.up_t8(r4_conv1).to(device) #F.interpolate(r4_conv1, [b2_conv3.shape[3],b2_conv3.shape[2]]).to(device)
        r5_conv1_resize_r2 = self.up_t16(r5_conv1).to(device) #F.interpolate(r5_conv1, [b2_conv3.shape[3],b2_conv3.shape[2]]).to(device)
        r6_conv1_resize_r2 = self.up_t32(r6_conv1).to(device) #F.interpolate(r6_conv1, [b2_conv3.shape[3],b2_conv3.shape[2]]).to(device)
        r2_concat = torch.cat((b2_conv3, r3_conv1_resize_r2, r4_conv1_resize_r2, r5_conv1_resize_r2, r6_conv1_resize_r2),1).to(device)
        r2_conv1 = self.b2_sc(r2_concat).to(device)
        # Rside 3
        # short connections to Aside 1
        # resize_r1=nn.ConvTranspose2d(1, 1, k * 2, k, k // 2)
        r3_conv1_resize_r1 = self.up_t8(r3_conv1).to(device) #F.interpolate(r3_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r4_conv1_resize_r1 = self.up_t16(r4_conv1).to(device) #F.interpolate(r4_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r5_conv1_resize_r1 = self.up_t32(r5_conv1).to(device) #F.interpolate(r5_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r6_conv1_resize_r1 = self.up_t64(r6_conv1).to(device) #F.interpolate(r6_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r1_concat = torch.cat((b1_conv3, r3_conv1_resize_r1, r4_conv1_resize_r1, r5_conv1_resize_r1, r6_conv1_resize_r1),1).to(device)
        r1_conv1 = self.b1_sc(r1_concat).to(device)
        # Rside 1       
        ################################
        # short connection: FUSION
        ################################        
        img_size=[ b1_conv3.shape[3],b1_conv3.shape[2] ]
        # fuse used branch results
        r1_conv1_resize = r1_conv1 #nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1,  1))(r1_conv1)#F.interpolate(r1_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r2_conv1_resize = self.up_t4(r2_conv1).to(device) #F.interpolate(r2_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r3_conv1_resize = self.up_t8(r3_conv1).to(device) #F.interpolate(r3_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r4_conv1_resize = self.up_t16(r4_conv1).to(device) #F.interpolate(r4_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r5_conv1_resize = self.up_t32(r5_conv1).to(device) #F.interpolate(r5_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r6_conv1_resize = self.up_t64(r6_conv1).to(device) #F.interpolate(r6_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        
        #fusion_concat=torch.cat((r2_conv1_resize,r3_conv1_resize,r4_conv1_resize],1).to(device)
        # STAT_TRAIN = 1 / STAT_TEST = 2
        if(state_trte==STAT_TRAIN): # 6 Channel all in
          fusion_concat=torch.cat((r1_conv1_resize,r2_conv1_resize,r3_conv1_resize,r4_conv1_resize,r5_conv1_resize,r6_conv1_resize),1).to(device)
          fusion_result=self.fusion_conv(fusion_concat).to(device)
        elif(state_trte==STAT_TEST): 
          method=1
          if(method==1): # method 1: 3 channel / 3 zeros
            r1_conv1_zero=torch.zeros(r1_conv1_resize.shape).to(device)
            r5_conv1_zero=torch.zeros(r5_conv1_resize.shape).to(device)
            r6_conv1_zero=torch.zeros(r6_conv1_resize.shape).to(device)
            fusion_concat=torch.cat((r1_conv1_zero,r2_conv1_resize,r3_conv1_resize,r4_conv1_resize,r5_conv1_zero,r6_conv1_zero),1).to(device)
            fusion_result=self.fusion_conv(fusion_concat).to(device)
          elif(method==2): # method 2: 3 channel copy
            fusion_concat=torch.cat((r2_conv1_resize,r2_conv1_resize,r3_conv1_resize,r4_conv1_resize,r3_conv1_resize,r4_conv1_resize),1).to(device)
            fusion_result=self.fusion_conv(fusion_concat).to(device)
          elif(method==3): # method 3: 3b conv
            fusion_concat=torch.cat((r2_conv1_resize,r3_conv1_resize,r4_conv1_resize),1).to(device)
            ## print(self.state_dict()['fusion_conv.weight'].shape) # [1,6,1,1]
            temp_weight=self.state_dict()['fusion_conv.weight']
            temp_weight=temp_weight[:,1:4,:,:] # b2,3,4 params
            ## print(temp_weight.shape)
            temp_bias = self.state_dict()['fusion_conv.bias'] #[:,:,:] #dim =[1,1,1]
            fusion_result=F.conv2d(fusion_concat, weight=temp_weight , bias= temp_bias, padding=1)#self.fusion_conv_inf(fusion_concat).to(device)    
        
        return fusion_result, b1_conv3, b2_conv3, b3_conv3, b4_conv3, b5_conv3, b6_conv3


# pixel net===============================================================================
class Net2(nn.Module):
    def __init__(self, gpu_device=torch.device("cpu"),start_EP=0):
        super(Net2, self).__init__()
        self.vgg_init=True
        self.device=gpu_device
        self.epoch=start_EP
        self.fcn_2d()

    def RGB_fcn_2d(self):    
        # RGB part
        self.rgb_conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.rgb_conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.rgb_pool_1 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.rgb_conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.rgb_pool_2 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.rgb_conv3_2 = nn.Conv2d(256, 256, 3, padding=1)  
        self.rgb_conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.rgb_pool_3 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.rgb_conv4_2 = nn.Conv2d(512, 512, 3, padding=1) 
        self.rgb_conv4_3 = nn.Conv2d(512, 512, 3, padding=1)    
        self.rgb_pool_4 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.rgb_conv5_2 = nn.Conv2d(512, 512, 3, padding=1)   
        self.rgb_conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.rgb_pool_5 =  nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)    

    def depth_fcn_2d(self):    
        # Depth part
        
        self.depth_conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.depth_conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.depth_pool_1 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.depth_conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.depth_pool_2 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.depth_conv3_2 = nn.Conv2d(256, 256, 3, padding=1)   
        self.depth_conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.depth_pool_3 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.depth_conv4_2 = nn.Conv2d(512, 512, 3, padding=1)   
        self.depth_conv4_3 = nn.Conv2d(512, 512, 3, padding=1)    
        self.depth_pool_4 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.depth_conv5_2 = nn.Conv2d(512, 512, 3, padding=1)  
        self.depth_conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.depth_pool_5 =  nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False) 

    
    def fcn_2d(self):
        self.RGB_fcn_2d()
        self.depth_fcn_2d()
        if(self.vgg_init and self.epoch==0):
          weight_init(self)
          if(DEBUG):
            print(self.state_dict())
        # short connection convolutions #RGBD #RGB (input channels are different)       
        # self.b1_conv1_rgb = nn.Conv2d(64, 128, 3, padding=1) #no depth features to concat
        self.b1_conv1 = nn.Conv2d(128, 128, 3, padding=1) #has depth features to concat
        self.b1_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.b1_conv3 = nn.Conv2d(128, 1, 1)
        
        # self.b2_conv1_rgb = nn.Conv2d(128, 128, 3, padding=1)
        self.b2_conv1 = nn.Conv2d(256, 128, 3, padding=1)
        self.b2_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.b2_conv3 = nn.Conv2d(128, 1, 1)
        
        # self.b3_conv1_rgb = nn.Conv2d(256, 256, 5, padding=1)
        self.b3_conv1 = nn.Conv2d(512, 256, 5, padding=2)
        self.b3_conv2 = nn.Conv2d(256, 256, 5, padding=2)
        self.b3_conv3 = nn.Conv2d(256, 1, 1)
        
        # self.b4_conv1_rgb = nn.Conv2d(512, 256, 5, padding=1)
        self.b4_conv1 = nn.Conv2d(1024, 256, 5, padding=2)
        self.b4_conv2 = nn.Conv2d(256, 256, 5, padding=2)
        self.b4_conv3 = nn.Conv2d(256, 1, 1)
        
        # self.b5_conv1_rgb = nn.Conv2d(512, 512, 5, padding=1)
        self.b5_conv1 = nn.Conv2d(1024, 512, 5, padding=2)
        self.b5_conv2 = nn.Conv2d(512, 512, 5, padding=2)
        self.b5_conv3 = nn.Conv2d(512, 1, 1)
        
        # self.b6_conv1_rgb = nn.Conv2d(512, 512, 7, padding=1)
        self.b6_conv1 = nn.Conv2d(1024, 512, 7, padding=3)
        self.b6_conv2 = nn.Conv2d(512, 512, 7, padding=3)
        self.b6_conv3 = nn.Conv2d(512, 1, 1)
        
        self.up_t64=nn.ConvTranspose2d(1, 1, kernel_size=(64, 64), stride=(32, 32), padding=(16, 16))
        self.up_t32=nn.ConvTranspose2d(1, 1, kernel_size=(32, 32), stride=(16, 16), padding=(8, 8))
        self.up_t16=nn.ConvTranspose2d(1, 1, kernel_size=(16, 16), stride=(8, 8), padding=(4, 4)) 
        self.up_t8=nn.ConvTranspose2d(1, 1, kernel_size=(8, 8), stride=(4, 4), padding=(2,  2))
        self.up_t4=nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1,  1))
        
        self.b4_sc=nn.Conv2d(3, 1, 3, padding=1)
        self.b3_sc=nn.Conv2d(3, 1, 3, padding=1)
        self.b2_sc=nn.Conv2d(5, 1, 3, padding=1)
        self.b1_sc=nn.Conv2d(5, 1, 3, padding=1)
        
        self.fusion_conv = nn.Conv2d(6, 1, 1, padding=1)
 
<<<<<<< HEAD
    def forward(self, input_rgb, input_depth, state_trte, rgb_path="", d_path="", d_spx=[]):  #### fcn_2d ####
=======
    def forward(self, input_rgb, input_depth, state_trte, rgb_path="", d_path=""):  #### fcn_2d ####
>>>>>>> dd86c05b6b13e46c37a8b50f8b4b326dda88ba1a
        # Max pooling over a (2, 2) window
        # activation function: Relu or Sigmoid (Later)
        
        #sequential: https://www.itread01.com/content/1547079864.html
        #GPUs
        device=self.device
        input_rgb=input_rgb.to(device).float()
        dataset_type=1
        
        ### RGB ##########################################
        rgb_conv1_1 = F.relu(self.rgb_conv1_1(input_rgb), inplace=True).to(device)
        rgb_conv1_2 = F.relu(self.rgb_conv1_2(rgb_conv1_1), inplace=True).to(device)
        rgb_pool1 = self.rgb_pool_1(rgb_conv1_2).to(device)
        
        rgb_conv2_1 = F.relu(self.rgb_conv2_1(rgb_pool1), inplace=True).to(device)
        rgb_conv2_2 = F.relu(self.rgb_conv2_2(rgb_conv2_1), inplace=True).to(device)
        rgb_pool2 = self.rgb_pool_2(rgb_conv2_2).to(device)
        
        rgb_conv3_1 = F.relu(self.rgb_conv3_1(rgb_pool2), inplace=True).to(device)
        rgb_conv3_2 = F.relu(self.rgb_conv3_2(rgb_conv3_1), inplace=True).to(device)
        rgb_conv3_3 = F.relu(self.rgb_conv3_3(rgb_conv3_2), inplace=True).to(device)
        rgb_pool3 = self.rgb_pool_3(rgb_conv3_3).to(device)    
        
        rgb_conv4_1 = F.relu(self.rgb_conv4_1(rgb_pool3), inplace=True).to(device)
        rgb_conv4_2 = F.relu(self.rgb_conv4_2(rgb_conv4_1), inplace=True).to(device)
        rgb_conv4_3 = F.relu(self.rgb_conv4_3(rgb_conv4_2), inplace=True).to(device)
        rgb_pool4 = self.rgb_pool_4(rgb_conv4_3).to(device) 
        
        rgb_conv5_1 = F.relu(self.rgb_conv5_1(rgb_pool4), inplace=True).to(device)
        rgb_conv5_2 = F.relu(self.rgb_conv5_2(rgb_conv5_1), inplace=True).to(device)
        rgb_conv5_3 = F.relu(self.rgb_conv5_3(rgb_conv5_2), inplace=True).to(device)
        rgb_pool5 = self.rgb_pool_2(rgb_conv5_3).to(device) 

        # rgb branches
        b1_rgb = rgb_conv1_2 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu1_2')
        b2_rgb = rgb_conv2_2 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu2_2')
        b3_rgb = rgb_conv3_3 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu3_3')
        b4_rgb = rgb_conv4_3 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu4_3')
        b5_rgb = rgb_conv5_3 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu5_3')
        b6_rgb = rgb_pool5 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_pool5')

        b1_concat = b1_rgb
        b2_concat = b2_rgb
        b3_concat = b3_rgb
        b4_concat = b4_rgb
        b5_concat = b5_rgb
        b6_concat = b6_rgb

        
        if(dataset_type):
            #GPUs
            input_depth=input_depth.to(device).float()
            ### DEPTH ##########################################
            depth_conv1_1 = F.relu(self.depth_conv1_1(input_depth), inplace=True).to(device)
            depth_conv1_2 = F.relu(self.depth_conv1_2(depth_conv1_1), inplace=True).to(device)
            depth_pool1 = self.depth_pool_1(depth_conv1_2).to(device)
            
            depth_conv2_1 = F.relu(self.depth_conv2_1(depth_pool1), inplace=True).to(device)
            depth_conv2_2 = F.relu(self.depth_conv2_2(depth_conv2_1), inplace=True).to(device)
            depth_pool2 = self.depth_pool_2(depth_conv2_2).to(device)    
            
            depth_conv3_1 = F.relu(self.depth_conv3_1(depth_pool2), inplace=True).to(device)
            depth_conv3_2 = F.relu(self.depth_conv3_2(depth_conv3_1), inplace=True).to(device)
            depth_conv3_3 = F.relu(self.depth_conv3_3(depth_conv3_2), inplace=True).to(device)
            depth_pool3 = self.depth_pool_3(depth_conv3_3).to(device)     
            
            depth_conv4_1 = F.relu(self.depth_conv4_1(depth_pool3), inplace=True).to(device)
            depth_conv4_2 = F.relu(self.depth_conv4_2(depth_conv4_1), inplace=True).to(device)
            depth_conv4_3 = F.relu(self.depth_conv4_3(depth_conv4_2), inplace=True).to(device)
            depth_pool4 = self.depth_pool_4(depth_conv4_3).to(device) 
            
            depth_conv5_1 = F.relu(self.depth_conv5_1(depth_pool4), inplace=True).to(device)
            depth_conv5_2 = F.relu(self.depth_conv5_2(depth_conv5_1), inplace=True).to(device)
            depth_conv5_3 = F.relu(self.depth_conv5_3(depth_conv5_2), inplace=True).to(device)
            depth_pool5 = self.depth_pool_2(depth_conv5_3).to(device).to(device) 
    
            # depth branches
            b1_depth=depth_conv1_2 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu1_2')
            b2_depth=depth_conv2_2 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu2_2')
            b3_depth=depth_conv3_3 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu3_3')
            b4_depth=depth_conv4_3 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu4_3')
            b5_depth=depth_conv5_3 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu5_3')
            b6_depth=depth_pool5 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_dpool5')

            ################################
            # short connection: concatenate
            # concat the axis = 1  dim=[0, "1", 2, 3]
            ################################
            b1_concat=torch.cat((b1_rgb,b1_depth),1).to(device)
            b2_concat=torch.cat((b2_rgb,b2_depth),1).to(device)
            b3_concat=torch.cat((b3_rgb,b3_depth),1).to(device)
            b4_concat=torch.cat((b4_rgb,b4_depth),1).to(device)
            b5_concat=torch.cat((b5_rgb,b5_depth),1).to(device)
            b6_concat=torch.cat((b6_rgb,b6_depth),1).to(device)
            

        ################################
        # short connection: to side activation
        ################################
        if(dataset_type):
            b1_conv1 = F.relu(self.b1_conv1(b1_concat), inplace=True).to(device)
            b2_conv1 = F.relu(self.b2_conv1(b2_concat), inplace=True).to(device)
            b3_conv1 = F.relu(self.b3_conv1(b3_concat), inplace=True).to(device)
            b4_conv1 = F.relu(self.b4_conv1(b4_concat), inplace=True).to(device)
            b5_conv1 = F.relu(self.b5_conv1(b5_concat), inplace=True).to(device)
            b6_conv1 = F.relu(self.b6_conv1(b6_concat), inplace=True).to(device)
        else:
            b1_conv1 = F.relu(self.b1_conv1_rgb(b1_concat), inplace=True).to(device)
            b2_conv1 = F.relu(self.b2_conv1_rgb(b2_concat), inplace=True).to(device)
            b3_conv1 = F.relu(self.b3_conv1_rgb(b3_concat), inplace=True).to(device)
            b4_conv1 = F.relu(self.b4_conv1_rgb(b4_concat), inplace=True).to(device)
            b5_conv1 = F.relu(self.b5_conv1_rgb(b5_concat), inplace=True).to(device)
            b6_conv1 = F.relu(self.b6_conv1_rgb(b6_concat), inplace=True).to(device)
        
        #branch 1
        b1_conv2 = F.relu(self.b1_conv2(b1_conv1), inplace=True).to(device)
        b1_conv3 = self.b1_conv3(b1_conv2).to(device)
        #branch 2
        b2_conv2 = F.relu(self.b2_conv2(b2_conv1), inplace=True).to(device)
        b2_conv3 = self.b2_conv3(b2_conv2).to(device)
        #branch 3
        b3_conv2 = F.relu(self.b3_conv2(b3_conv1), inplace=True).to(device)
        b3_conv3 = self.b3_conv3(b3_conv2).to(device)
        #branch 4
        b4_conv2 = F.relu(self.b4_conv2(b4_conv1), inplace=True).to(device)
        b4_conv3 = self.b4_conv3(b4_conv2).to(device)
        #branch 5
        b5_conv2 = F.relu(self.b5_conv2(b5_conv1), inplace=True).to(device)
        b5_conv3 = self.b5_conv3(b5_conv2).to(device)
        #branch 6
        b6_conv2 = F.relu(self.b6_conv2(b6_conv1), inplace=True).to(device)
        b6_conv3 = self.b6_conv3(b6_conv2).to(device)
        
        ################################
        # short connection: branch results
        ################################        
        # shortconnections to Aside 6
        r6_conv1 = b6_conv3
        # Rside 6
        
        # shortconnections to Aside 5
        r5_conv1 = b5_conv3
        # Rside 5
        
        # shortconnections to Aside 4
        # resize_r4=nn.ConvTranspose2d(1, 1, k * 2, k, k // 2)
        r5_conv1_resize_r4 = self.up_t4(r5_conv1).to(device)#F.interpolate(r5_conv1, [b4_conv3.shape[3],b4_conv3.shape[2]])
        r6_conv1_resize_r4 = self.up_t8(r6_conv1).to(device)#F.interpolate(r6_conv1, [b4_conv3.shape[3],b4_conv3.shape[2]]).to(device)
        #print(r5_conv1_resize_r4.shape) #1, 1, 72, 72
        #print(r6_conv1_resize_r4.shape) #1, 1, 48, 48
        #print(b4_conv3.shape) #1, 1, 76, 76
        r4_concat = torch.cat((b4_conv3, r5_conv1_resize_r4, r6_conv1_resize_r4),1).to(device)
        r4_conv1 = self.b4_sc(r4_concat).to(device)
        # Rside 4
        # shortconnections to Aside 3
        # resize_r3=nn.ConvTranspose2d(1, 1, k * 2, k, k // 2)
        r5_conv1_resize_r3 = self.up_t8(r5_conv1).to(device)#F.interpolate(r5_conv1, [b3_conv3.shape[3],b3_conv3.shape[2]]).to(device)
        r6_conv1_resize_r3 = self.up_t16(r6_conv1).to(device)#F.interpolate(r6_conv1, [b3_conv3.shape[3],b3_conv3.shape[2]]).to(device)
        r3_concat = torch.cat((b3_conv3, r5_conv1_resize_r3, r6_conv1_resize_r3),1).to(device)
        r3_conv1 = self.b3_sc(r3_concat).to(device)
        # Rside 3
        # short connections to Aside 2
        # resize_r2=nn.ConvTranspose2d(1, 1, k * 2, k, k // 2)
        r3_conv1_resize_r2 = self.up_t4(r3_conv1).to(device)#F.interpolate(r3_conv1, [b2_conv3.shape[3],b2_conv3.shape[2]]).to(device)
        r4_conv1_resize_r2 = self.up_t8(r4_conv1).to(device)#F.interpolate(r4_conv1, [b2_conv3.shape[3],b2_conv3.shape[2]]).to(device)
        r5_conv1_resize_r2 = self.up_t16(r5_conv1).to(device)#F.interpolate(r5_conv1, [b2_conv3.shape[3],b2_conv3.shape[2]]).to(device)
        r6_conv1_resize_r2 = self.up_t32(r6_conv1).to(device)#F.interpolate(r6_conv1, [b2_conv3.shape[3],b2_conv3.shape[2]]).to(device)
        r2_concat = torch.cat((b2_conv3, r3_conv1_resize_r2, r4_conv1_resize_r2, r5_conv1_resize_r2, r6_conv1_resize_r2),1).to(device)
        r2_conv1 = self.b2_sc(r2_concat).to(device)
        # Rside 3
        # short connections to Aside 1
        # resize_r1=nn.ConvTranspose2d(1, 1, k * 2, k, k // 2)
        r3_conv1_resize_r1 = self.up_t8(r3_conv1).to(device)#F.interpolate(r3_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r4_conv1_resize_r1 = self.up_t16(r4_conv1).to(device)#F.interpolate(r4_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r5_conv1_resize_r1 = self.up_t32(r5_conv1).to(device)#F.interpolate(r5_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r6_conv1_resize_r1 = self.up_t64(r6_conv1).to(device)#F.interpolate(r6_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r1_concat = torch.cat((b1_conv3, r3_conv1_resize_r1, r4_conv1_resize_r1, r5_conv1_resize_r1, r6_conv1_resize_r1),1).to(device)
        r1_conv1 = self.b1_sc(r1_concat).to(device)
        # Rside 1       
        ################################
        # short connection: FUSION
        ################################        
        img_size=[ b1_conv3.shape[3],b1_conv3.shape[2] ]
        # fuse used branch results
        r1_conv1_resize = r1_conv1 #nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1,  1))(r1_conv1)#F.interpolate(r1_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r2_conv1_resize = self.up_t4(r2_conv1).to(device)#F.interpolate(r2_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r3_conv1_resize = self.up_t8(r3_conv1).to(device)#F.interpolate(r3_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r4_conv1_resize = self.up_t16(r4_conv1).to(device)#F.interpolate(r4_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r5_conv1_resize = self.up_t32(r5_conv1).to(device)#F.interpolate(r5_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r6_conv1_resize = self.up_t64(r6_conv1).to(device)#F.interpolate(r6_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        
        #fusion_concat=torch.cat((r2_conv1_resize,r3_conv1_resize,r4_conv1_resize],1).to(device)
        # STAT_TRAIN = 1 / STAT_TEST = 2
        if(state_trte==STAT_TRAIN): # 6 Channel all in
          fusion_concat=torch.cat((r1_conv1_resize,r2_conv1_resize,r3_conv1_resize,r4_conv1_resize,r5_conv1_resize,r6_conv1_resize),1).to(device)
          fusion_result=self.fusion_conv(fusion_concat).to(device)
        elif(state_trte==STAT_TEST): 
          method=1
          if(method==1): # method 1: 3 channel / 3 zeros
            r1_conv1_zero=torch.zeros(r1_conv1_resize.shape).to(device)
            r5_conv1_zero=torch.zeros(r5_conv1_resize.shape).to(device)
            r6_conv1_zero=torch.zeros(r6_conv1_resize.shape).to(device)
            fusion_concat=torch.cat((r1_conv1_zero,r2_conv1_resize,r3_conv1_resize,r4_conv1_resize,r5_conv1_zero,r6_conv1_zero),1).to(device)
            fusion_result=self.fusion_conv(fusion_concat).to(device)
          elif(method==2): # method 2: 3 channel copy
            fusion_concat=torch.cat((r2_conv1_resize,r2_conv1_resize,r3_conv1_resize,r4_conv1_resize,r3_conv1_resize,r4_conv1_resize),1).to(device)
            fusion_result=self.fusion_conv(fusion_concat).to(device)
          elif(method==3): # method 3: 3b conv
            fusion_concat=torch.cat((r2_conv1_resize,r3_conv1_resize,r4_conv1_resize),1).to(device)
            ## print(self.state_dict()['fusion_conv.weight'].shape) # [1,6,1,1]
            temp_weight=self.state_dict()['fusion_conv.weight']
            temp_weight=temp_weight[:,1:4,:,:] # b2,3,4 params
            ## print(temp_weight.shape)
            temp_bias = self.state_dict()['fusion_conv.bias'] #[:,:,:] #dim =[1,1,1]
            c=F.conv2d(fusion_concat, weight=temp_weight , bias= temp_bias, padding=1)#self.fusion_conv_inf(fusion_concat).to(device)    
            
            
        ########
        # SP pooling
        ########
<<<<<<< HEAD
        '''
=======
>>>>>>> dd86c05b6b13e46c37a8b50f8b4b326dda88ba1a
        # spx = np.zeros((batch_size, xSize, ySize))
        if(dataset_type): # rgbd
          #rgb_path_st=rgb_path.numpy()
          #d_path_st=d_path.numpy()
          d_spx = slic(io.imread(rgb_path))
        else:
          # print("rgb_path",rgb_path)
          # print("d_path",d_path)
          d_spx = RGBD_SLICProcessor(rgb_path,d_path)
          
        # 
<<<<<<< HEAD
        
        # d_spx=sp_map
        spx_trans_compose = transforms.Compose([
            transforms.ToTensor(),
        ])
        '''
        
=======
        spx_trans_compose = transforms.Compose([
            transforms.ToTensor(),
        ])
>>>>>>> dd86c05b6b13e46c37a8b50f8b4b326dda88ba1a
        spx = spx_trans_compose(d_spx).to(device) #torch.from_numpy(spx) 
        # print(spx.shape)#[300, 400] 
        #[1, 1, 642, 642]
        # print()
        # spx = F.interpolate(spx, [fusion_result.shape[3],fusion_result.shape[2]]).to(device)
        pooling_in=F.interpolate(fusion_result, [spx.shape[1],spx.shape[2]]).to(device)
        # print ("INPUT ARRAY  ----------------- \n", X) 
        pld = pool(pooling_in, spx)
        # print ("::pld::",pld.shape)
        # print ("POOLED ARRAY ----------------- \n", pld)
        # print ("Shape of pooled array: ", pld.size())
        
        unpld = unpool(pld, spx)
        fusion_result = unpld
        
        return fusion_result, b1_conv3, b2_conv3, b3_conv3, b4_conv3, b5_conv3, b6_conv3
        
        

        
        
# pixel net===============================================================================
class Net3(nn.Module):
    def __init__(self, gpu_device=torch.device("cpu"),start_EP=0):
        super(Net3, self).__init__()
        self.vgg_init=True
        self.device=gpu_device
        self.epoch=start_EP
        self.fcn_2d()

    def RGB_fcn_2d(self):    
        # RGB part
        self.rgb_conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.rgb_conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.rgb_pool_1 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.rgb_conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.rgb_pool_2 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.rgb_conv3_2 = nn.Conv2d(256, 256, 3, padding=1)  
        self.rgb_conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.rgb_pool_3 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.rgb_conv4_2 = nn.Conv2d(512, 512, 3, padding=1) 
        self.rgb_conv4_3 = nn.Conv2d(512, 512, 3, padding=1)    
        self.rgb_pool_4 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.rgb_conv5_2 = nn.Conv2d(512, 512, 3, padding=1)   
        self.rgb_conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.rgb_pool_5 =  nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)    

    def depth_fcn_2d(self):    
        # Depth part
        
        self.depth_conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.depth_conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.depth_pool_1 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.depth_conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.depth_pool_2 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.depth_conv3_2 = nn.Conv2d(256, 256, 3, padding=1)   
        self.depth_conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.depth_pool_3 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.depth_conv4_2 = nn.Conv2d(512, 512, 3, padding=1)   
        self.depth_conv4_3 = nn.Conv2d(512, 512, 3, padding=1)    
        self.depth_pool_4 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.depth_conv5_2 = nn.Conv2d(512, 512, 3, padding=1)  
        self.depth_conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.depth_pool_5 =  nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False) 

    
    def fcn_2d(self):
        self.RGB_fcn_2d()
        self.depth_fcn_2d()
        if(self.vgg_init and self.epoch==0):
          weight_init(self)
          if(DEBUG):
            print(self.state_dict())
        # short connection convolutions #RGBD #RGB (input channels are different)       
        # self.b1_conv1_rgb = nn.Conv2d(64, 128, 3, padding=1) #no depth features to concat
        self.b1_conv1 = nn.Conv2d(128, 128, 3, padding=1) #has depth features to concat
        self.b1_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.b1_conv3 = nn.Conv2d(128, 1, 1)
        
        # self.b2_conv1_rgb = nn.Conv2d(128, 128, 3, padding=1)
        self.b2_conv1 = nn.Conv2d(256, 128, 3, padding=1)
        self.b2_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.b2_conv3 = nn.Conv2d(128, 1, 1)
        
        # self.b3_conv1_rgb = nn.Conv2d(256, 256, 5, padding=1)
        self.b3_conv1 = nn.Conv2d(512, 256, 5, padding=2)
        self.b3_conv2 = nn.Conv2d(256, 256, 5, padding=2)
        self.b3_conv3 = nn.Conv2d(256, 1, 1)
        
        # self.b4_conv1_rgb = nn.Conv2d(512, 256, 5, padding=1)
        self.b4_conv1 = nn.Conv2d(1024, 256, 5, padding=2)
        self.b4_conv2 = nn.Conv2d(256, 256, 5, padding=2)
        self.b4_conv3 = nn.Conv2d(256, 1, 1)
        
        # self.b5_conv1_rgb = nn.Conv2d(512, 512, 5, padding=1)
        self.b5_conv1 = nn.Conv2d(1024, 512, 5, padding=2)
        self.b5_conv2 = nn.Conv2d(512, 512, 5, padding=2)
        self.b5_conv3 = nn.Conv2d(512, 1, 1)
        
        # self.b6_conv1_rgb = nn.Conv2d(512, 512, 7, padding=1)
        self.b6_conv1 = nn.Conv2d(1024, 512, 7, padding=3)
        self.b6_conv2 = nn.Conv2d(512, 512, 7, padding=3)
        self.b6_conv3 = nn.Conv2d(512, 1, 1)
        
        self.up_t64=nn.ConvTranspose2d(1, 1, kernel_size=(64, 64), stride=(32, 32), padding=(16, 16))
        self.up_t32=nn.ConvTranspose2d(1, 1, kernel_size=(32, 32), stride=(16, 16), padding=(8, 8))
        self.up_t16=nn.ConvTranspose2d(1, 1, kernel_size=(16, 16), stride=(8, 8), padding=(4, 4)) 
        self.up_t8=nn.ConvTranspose2d(1, 1, kernel_size=(8, 8), stride=(4, 4), padding=(2,  2))
        self.up_t4=nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1,  1))
        
        self.b4_sc=nn.Conv2d(3, 1, 3, padding=1)
        self.b3_sc=nn.Conv2d(3, 1, 3, padding=1)
        self.b2_sc=nn.Conv2d(5, 1, 3, padding=1)
        self.b1_sc=nn.Conv2d(5, 1, 3, padding=1)
        
        #self.fusion_conv = nn.Conv2d(6, 1, 1, padding=1)
        self.fusion_conv = nn.Conv1d(6, 1, 1, padding=1)
 
<<<<<<< HEAD
    def forward(self, input_rgb, input_depth, state_trte, rgb_path="", d_path="", d_spx=[]):  #### fcn_2d ####
=======
    def forward(self, input_rgb, input_depth, state_trte, rgb_path="", d_path=""):  #### fcn_2d ####
>>>>>>> dd86c05b6b13e46c37a8b50f8b4b326dda88ba1a
        # Max pooling over a (2, 2) window
        # activation function: Relu or Sigmoid (Later)
        
        #sequential: https://www.itread01.com/content/1547079864.html
        #GPUs
        device=self.device
        input_rgb=input_rgb.to(device).float()
        dataset_type=1
        
        ########
        # SP pooling prepare
        ########
<<<<<<< HEAD
        '''
=======
>>>>>>> dd86c05b6b13e46c37a8b50f8b4b326dda88ba1a
        # spx = np.zeros((batch_size, xSize, ySize))
        if(dataset_type): # rgbd
          #rgb_path_st=rgb_path.numpy()
          #d_path_st=d_path.numpy()
          d_spx = slic(io.imread(rgb_path))
        else:
          # print("rgb_path",rgb_path)
          # print("d_path",d_path)
          d_spx = RGBD_SLICProcessor(rgb_path,d_path)
          
        # to tensor
        spx_trans_compose = transforms.Compose([
            transforms.ToTensor(),
        ])
<<<<<<< HEAD
        '''
=======
>>>>>>> dd86c05b6b13e46c37a8b50f8b4b326dda88ba1a
        spx = spx_trans_compose(d_spx).to(device) #torch.from_numpy(spx) 
        
        
        ### RGB ##########################################
        rgb_conv1_1 = F.relu(self.rgb_conv1_1(input_rgb), inplace=True).to(device)
        rgb_conv1_2 = F.relu(self.rgb_conv1_2(rgb_conv1_1), inplace=True).to(device)
        rgb_pool1 = self.rgb_pool_1(rgb_conv1_2).to(device)
        
        rgb_conv2_1 = F.relu(self.rgb_conv2_1(rgb_pool1), inplace=True).to(device)
        rgb_conv2_2 = F.relu(self.rgb_conv2_2(rgb_conv2_1), inplace=True).to(device)
        rgb_pool2 = self.rgb_pool_2(rgb_conv2_2).to(device)
        
        rgb_conv3_1 = F.relu(self.rgb_conv3_1(rgb_pool2), inplace=True).to(device)
        rgb_conv3_2 = F.relu(self.rgb_conv3_2(rgb_conv3_1), inplace=True).to(device)
        rgb_conv3_3 = F.relu(self.rgb_conv3_3(rgb_conv3_2), inplace=True).to(device)
        rgb_pool3 = self.rgb_pool_3(rgb_conv3_3).to(device)    
        
        rgb_conv4_1 = F.relu(self.rgb_conv4_1(rgb_pool3), inplace=True).to(device)
        rgb_conv4_2 = F.relu(self.rgb_conv4_2(rgb_conv4_1), inplace=True).to(device)
        rgb_conv4_3 = F.relu(self.rgb_conv4_3(rgb_conv4_2), inplace=True).to(device)
        rgb_pool4 = self.rgb_pool_4(rgb_conv4_3).to(device) 
        
        rgb_conv5_1 = F.relu(self.rgb_conv5_1(rgb_pool4), inplace=True).to(device)
        rgb_conv5_2 = F.relu(self.rgb_conv5_2(rgb_conv5_1), inplace=True).to(device)
        rgb_conv5_3 = F.relu(self.rgb_conv5_3(rgb_conv5_2), inplace=True).to(device)
        rgb_pool5 = self.rgb_pool_2(rgb_conv5_3).to(device) 

        # rgb branches
        b1_rgb = rgb_conv1_2 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu1_2')
        b2_rgb = rgb_conv2_2 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu2_2')
        b3_rgb = rgb_conv3_3 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu3_3')
        b4_rgb = rgb_conv4_3 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu4_3')
        b5_rgb = rgb_conv5_3 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu5_3')
        b6_rgb = rgb_pool5 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_pool5')

        b1_concat = b1_rgb
        b2_concat = b2_rgb
        b3_concat = b3_rgb
        b4_concat = b4_rgb
        b5_concat = b5_rgb
        b6_concat = b6_rgb

        
        if(dataset_type):
            #GPUs
            input_depth=input_depth.to(device).float()
            ### DEPTH ##########################################
            depth_conv1_1 = F.relu(self.depth_conv1_1(input_depth), inplace=True).to(device)
            depth_conv1_2 = F.relu(self.depth_conv1_2(depth_conv1_1), inplace=True).to(device)
            depth_pool1 = self.depth_pool_1(depth_conv1_2).to(device)
            
            depth_conv2_1 = F.relu(self.depth_conv2_1(depth_pool1), inplace=True).to(device)
            depth_conv2_2 = F.relu(self.depth_conv2_2(depth_conv2_1), inplace=True).to(device)
            depth_pool2 = self.depth_pool_2(depth_conv2_2).to(device)    
            
            depth_conv3_1 = F.relu(self.depth_conv3_1(depth_pool2), inplace=True).to(device)
            depth_conv3_2 = F.relu(self.depth_conv3_2(depth_conv3_1), inplace=True).to(device)
            depth_conv3_3 = F.relu(self.depth_conv3_3(depth_conv3_2), inplace=True).to(device)
            depth_pool3 = self.depth_pool_3(depth_conv3_3).to(device)     
            
            depth_conv4_1 = F.relu(self.depth_conv4_1(depth_pool3), inplace=True).to(device)
            depth_conv4_2 = F.relu(self.depth_conv4_2(depth_conv4_1), inplace=True).to(device)
            depth_conv4_3 = F.relu(self.depth_conv4_3(depth_conv4_2), inplace=True).to(device)
            depth_pool4 = self.depth_pool_4(depth_conv4_3).to(device) 
            
            depth_conv5_1 = F.relu(self.depth_conv5_1(depth_pool4), inplace=True).to(device)
            depth_conv5_2 = F.relu(self.depth_conv5_2(depth_conv5_1), inplace=True).to(device)
            depth_conv5_3 = F.relu(self.depth_conv5_3(depth_conv5_2), inplace=True).to(device)
            depth_pool5 = self.depth_pool_2(depth_conv5_3).to(device).to(device) 
    
            # depth branches
            b1_depth=depth_conv1_2 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu1_2')
            b2_depth=depth_conv2_2 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu2_2')
            b3_depth=depth_conv3_3 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu3_3')
            b4_depth=depth_conv4_3 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu4_3')
            b5_depth=depth_conv5_3 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu5_3')
            b6_depth=depth_pool5 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_dpool5')

            ################################
            # short connection: concatenate
            # concat the axis = 1  dim=[0, "1", 2, 3]
            ################################
            b1_concat=torch.cat((b1_rgb,b1_depth),1).to(device)
            b2_concat=torch.cat((b2_rgb,b2_depth),1).to(device)
            b3_concat=torch.cat((b3_rgb,b3_depth),1).to(device)
            b4_concat=torch.cat((b4_rgb,b4_depth),1).to(device)
            b5_concat=torch.cat((b5_rgb,b5_depth),1).to(device)
            b6_concat=torch.cat((b6_rgb,b6_depth),1).to(device)
            

        ################################
        # short connection: to side activation
        ################################
        if(dataset_type):
            b1_conv1 = F.relu(self.b1_conv1(b1_concat), inplace=True).to(device)
            b2_conv1 = F.relu(self.b2_conv1(b2_concat), inplace=True).to(device)
            b3_conv1 = F.relu(self.b3_conv1(b3_concat), inplace=True).to(device)
            b4_conv1 = F.relu(self.b4_conv1(b4_concat), inplace=True).to(device)
            b5_conv1 = F.relu(self.b5_conv1(b5_concat), inplace=True).to(device)
            b6_conv1 = F.relu(self.b6_conv1(b6_concat), inplace=True).to(device)
        else:
            b1_conv1 = F.relu(self.b1_conv1_rgb(b1_concat), inplace=True).to(device)
            b2_conv1 = F.relu(self.b2_conv1_rgb(b2_concat), inplace=True).to(device)
            b3_conv1 = F.relu(self.b3_conv1_rgb(b3_concat), inplace=True).to(device)
            b4_conv1 = F.relu(self.b4_conv1_rgb(b4_concat), inplace=True).to(device)
            b5_conv1 = F.relu(self.b5_conv1_rgb(b5_concat), inplace=True).to(device)
            b6_conv1 = F.relu(self.b6_conv1_rgb(b6_concat), inplace=True).to(device)
        
        #branch 1
        b1_conv2 = F.relu(self.b1_conv2(b1_conv1), inplace=True).to(device)
        b1_conv3 = self.b1_conv3(b1_conv2).to(device)
        #branch 2
        b2_conv2 = F.relu(self.b2_conv2(b2_conv1), inplace=True).to(device)
        b2_conv3 = self.b2_conv3(b2_conv2).to(device)
        #branch 3
        b3_conv2 = F.relu(self.b3_conv2(b3_conv1), inplace=True).to(device)
        b3_conv3 = self.b3_conv3(b3_conv2).to(device)
        #branch 4
        b4_conv2 = F.relu(self.b4_conv2(b4_conv1), inplace=True).to(device)
        b4_conv3 = self.b4_conv3(b4_conv2).to(device)
        #branch 5
        b5_conv2 = F.relu(self.b5_conv2(b5_conv1), inplace=True).to(device)
        b5_conv3 = self.b5_conv3(b5_conv2).to(device)
        #branch 6
        b6_conv2 = F.relu(self.b6_conv2(b6_conv1), inplace=True).to(device)
        b6_conv3 = self.b6_conv3(b6_conv2).to(device)
        
        ################################
        # short connection: branch results
        ################################        
        # shortconnections to Aside 6
        r6_conv1 = b6_conv3
        # Rside 6
        
        # shortconnections to Aside 5
        r5_conv1 = b5_conv3
        # Rside 5
        
        # shortconnections to Aside 4
        # resize_r4=nn.ConvTranspose2d(1, 1, k * 2, k, k // 2)
        r5_conv1_resize_r4 = self.up_t4(r5_conv1).to(device)#F.interpolate(r5_conv1, [b4_conv3.shape[3],b4_conv3.shape[2]])
        r6_conv1_resize_r4 = self.up_t8(r6_conv1).to(device)#F.interpolate(r6_conv1, [b4_conv3.shape[3],b4_conv3.shape[2]]).to(device)
        #print(r5_conv1_resize_r4.shape) #1, 1, 72, 72
        #print(r6_conv1_resize_r4.shape) #1, 1, 48, 48
        #print(b4_conv3.shape) #1, 1, 76, 76
        r4_concat = torch.cat((b4_conv3, r5_conv1_resize_r4, r6_conv1_resize_r4),1).to(device)
        r4_conv1 = self.b4_sc(r4_concat).to(device)
        # Rside 4
        # shortconnections to Aside 3
        # resize_r3=nn.ConvTranspose2d(1, 1, k * 2, k, k // 2)
        r5_conv1_resize_r3 = self.up_t8(r5_conv1).to(device)#F.interpolate(r5_conv1, [b3_conv3.shape[3],b3_conv3.shape[2]]).to(device)
        r6_conv1_resize_r3 = self.up_t16(r6_conv1).to(device)#F.interpolate(r6_conv1, [b3_conv3.shape[3],b3_conv3.shape[2]]).to(device)
        r3_concat = torch.cat((b3_conv3, r5_conv1_resize_r3, r6_conv1_resize_r3),1).to(device)
        r3_conv1 = self.b3_sc(r3_concat).to(device)
        # Rside 3
        # short connections to Aside 2
        # resize_r2=nn.ConvTranspose2d(1, 1, k * 2, k, k // 2)
        r3_conv1_resize_r2 = self.up_t4(r3_conv1).to(device)#F.interpolate(r3_conv1, [b2_conv3.shape[3],b2_conv3.shape[2]]).to(device)
        r4_conv1_resize_r2 = self.up_t8(r4_conv1).to(device)#F.interpolate(r4_conv1, [b2_conv3.shape[3],b2_conv3.shape[2]]).to(device)
        r5_conv1_resize_r2 = self.up_t16(r5_conv1).to(device)#F.interpolate(r5_conv1, [b2_conv3.shape[3],b2_conv3.shape[2]]).to(device)
        r6_conv1_resize_r2 = self.up_t32(r6_conv1).to(device)#F.interpolate(r6_conv1, [b2_conv3.shape[3],b2_conv3.shape[2]]).to(device)
        r2_concat = torch.cat((b2_conv3, r3_conv1_resize_r2, r4_conv1_resize_r2, r5_conv1_resize_r2, r6_conv1_resize_r2),1).to(device)
        r2_conv1 = self.b2_sc(r2_concat).to(device)
        # Rside 3
        # short connections to Aside 1
        # resize_r1=nn.ConvTranspose2d(1, 1, k * 2, k, k // 2)
        r3_conv1_resize_r1 = self.up_t8(r3_conv1).to(device)#F.interpolate(r3_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r4_conv1_resize_r1 = self.up_t16(r4_conv1).to(device)#F.interpolate(r4_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r5_conv1_resize_r1 = self.up_t32(r5_conv1).to(device)#F.interpolate(r5_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r6_conv1_resize_r1 = self.up_t64(r6_conv1).to(device)#F.interpolate(r6_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r1_concat = torch.cat((b1_conv3, r3_conv1_resize_r1, r4_conv1_resize_r1, r5_conv1_resize_r1, r6_conv1_resize_r1),1).to(device)
        r1_conv1 = self.b1_sc(r1_concat).to(device)
        # Rside 1       
        ################################
        # short connection: FUSION
        ################################        
        img_size=[ b1_conv3.shape[3],b1_conv3.shape[2] ]
        # fuse used branch results
        r1_conv1_resize = r1_conv1 #nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1,  1))(r1_conv1)#F.interpolate(r1_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r2_conv1_resize = self.up_t4(r2_conv1).to(device)#F.interpolate(r2_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r3_conv1_resize = self.up_t8(r3_conv1).to(device)#F.interpolate(r3_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r4_conv1_resize = self.up_t16(r4_conv1).to(device)#F.interpolate(r4_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r5_conv1_resize = self.up_t32(r5_conv1).to(device)#F.interpolate(r5_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r6_conv1_resize = self.up_t64(r6_conv1).to(device)#F.interpolate(r6_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        
        
        ########
        # SP pooling branches
        ########
        # print(spx.shape)#[300, 400] 
        #[1, 1, 642, 642]
        # print()
        # spx = F.interpolate(spx, [fusion_result.shape[3],fusion_result.shape[2]]).to(device)
        # pooling_in=F.interpolate(fusion_result, [spx.shape[1],spx.shape[2]]).to(device)
        # print ("INPUT ARRAY  ----------------- \n", X) 
        r1_conv1_resize_pldIn=F.interpolate(r1_conv1_resize, [spx.shape[1],spx.shape[2]]).to(device)
        r2_conv1_resize_pldIn=F.interpolate(r2_conv1_resize, [spx.shape[1],spx.shape[2]]).to(device)
        r3_conv1_resize_pldIn=F.interpolate(r3_conv1_resize, [spx.shape[1],spx.shape[2]]).to(device)
        r4_conv1_resize_pldIn=F.interpolate(r4_conv1_resize, [spx.shape[1],spx.shape[2]]).to(device)
        r5_conv1_resize_pldIn=F.interpolate(r5_conv1_resize, [spx.shape[1],spx.shape[2]]).to(device)
        r6_conv1_resize_pldIn=F.interpolate(r6_conv1_resize, [spx.shape[1],spx.shape[2]]).to(device)

        r1_conv1_resize_pld = pool(r1_conv1_resize_pldIn, spx)
        r2_conv1_resize_pld = pool(r2_conv1_resize_pldIn, spx)
        r3_conv1_resize_pld = pool(r3_conv1_resize_pldIn, spx)
        r4_conv1_resize_pld = pool(r4_conv1_resize_pldIn, spx)
        r5_conv1_resize_pld = pool(r5_conv1_resize_pldIn, spx)
        r6_conv1_resize_pld = pool(r6_conv1_resize_pldIn, spx)
        
        # print ("::pld::",pld.shape)
        # print ("POOLED ARRAY ----------------- \n", pld)
        # print ("Shape of pooled array: ", pld.size())        
        
        
        
        #fusion_concat=torch.cat((r2_conv1_resize,r3_conv1_resize,r4_conv1_resize],1).to(device)
        # STAT_TRAIN = 1 / STAT_TEST = 2
        if(state_trte==STAT_TRAIN): # 6 Channel all in
          fusion_concat=torch.cat((r1_conv1_resize_pld,r2_conv1_resize_pld,r3_conv1_resize_pld,
                                   r4_conv1_resize_pld,r5_conv1_resize_pld,r6_conv1_resize_pld),1).to(device)
          fusion_result=self.fusion_conv(fusion_concat).to(device)
        elif(state_trte==STAT_TEST): 
          method=1
          if(method==1): # method 1: 3 channel / 3 zeros
            r1_conv1_zero=torch.zeros(r1_conv1_resize_pld.shape).to(device)
            r5_conv1_zero=torch.zeros(r5_conv1_resize_pld.shape).to(device)
            r6_conv1_zero=torch.zeros(r6_conv1_resize_pld.shape).to(device)
            fusion_concat=torch.cat((r1_conv1_zero,r2_conv1_resize_pld,r3_conv1_resize_pld,r4_conv1_resize_pld,r5_conv1_zero,r6_conv1_zero),1).to(device)
            fusion_result=self.fusion_conv(fusion_concat).to(device)
          elif(method==2): # method 2: 3 channel copy
            fusion_concat=torch.cat((r2_conv1_resize_pld,r2_conv1_resize_pld,r3_conv1_resize_pld,r3_conv1_resize_pld,r4_conv1_resize_pld,r4_conv1_resize)
                                      ,1).to(device)
            fusion_result=self.fusion_conv(fusion_concat).to(device)
          elif(method==3): # method 3: 3b conv
            fusion_concat=torch.cat((r2_conv1_resize_pld,r3_conv1_resize_pld,r4_conv1_resize_pld),1).to(device)
            ## print(self.state_dict()['fusion_conv.weight'].shape) # [1,6,1,1]
            temp_weight=self.state_dict()['fusion_conv.weight']
            temp_weight=temp_weight[:,1:4,:,:] # b2,3,4 params
            ## print(temp_weight.shape)
            temp_bias = self.state_dict()['fusion_conv.bias'] #[:,:,:] #dim =[1,1,1]
            c=F.conv2d(fusion_concat, weight=temp_weight , bias= temp_bias, padding=1)#self.fusion_conv_inf(fusion_concat).to(device)    
            
            
        

        
        unpld = unpool(fusion_result, spx)
        fusion_result = unpld
        
        return fusion_result, b1_conv3, b2_conv3, b3_conv3, b4_conv3, b5_conv3, b6_conv3



# ===============================================================================
# pixel net (interpolation)
class Net_interpolation(nn.Module):
    def __init__(self, gpu_device=torch.device("cpu"),start_EP=0):
        super(Net_interpolation, self).__init__()
        self.vgg_init=True
        self.device=gpu_device
        self.epoch=start_EP
        self.fcn_2d()

    def RGB_fcn_2d(self):    
        # RGB part
        self.rgb_conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.rgb_conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.rgb_pool_1 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.rgb_conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.rgb_pool_2 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.rgb_conv3_2 = nn.Conv2d(256, 256, 3, padding=1)  
        self.rgb_conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.rgb_pool_3 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.rgb_conv4_2 = nn.Conv2d(512, 512, 3, padding=1) 
        self.rgb_conv4_3 = nn.Conv2d(512, 512, 3, padding=1)    
        self.rgb_pool_4 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.rgb_conv5_2 = nn.Conv2d(512, 512, 3, padding=1)   
        self.rgb_conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.rgb_pool_5 =  nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)    

    def depth_fcn_2d(self):    
        # Depth part
        self.depth_conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.depth_conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.depth_pool_1 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.depth_conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.depth_pool_2 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.depth_conv3_2 = nn.Conv2d(256, 256, 3, padding=1)   
        self.depth_conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.depth_pool_3 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.depth_conv4_2 = nn.Conv2d(512, 512, 3, padding=1)   
        self.depth_conv4_3 = nn.Conv2d(512, 512, 3, padding=1)    
        self.depth_pool_4 = nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.depth_conv5_2 = nn.Conv2d(512, 512, 3, padding=1)  
        self.depth_conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.depth_pool_5 =  nn.MaxPool2d(2, 2, dilation=1, ceil_mode=False) 

    
    def fcn_2d(self):
        self.RGB_fcn_2d()
        self.depth_fcn_2d()
        if(self.vgg_init and self.epoch==0):
          weight_init(self)
          if(DEBUG):
              print(self.state_dict())
        
        # short connection convolutions #RGBD #RGB (input channels are different)       
        # self.b1_conv1_rgb = nn.Conv2d(64, 128, 3, padding=1) #no depth features to concat
        self.b1_conv1 = nn.Conv2d(128, 128, 3, padding=1) #has depth features to concat
        self.b1_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.b1_conv3 = nn.Conv2d(128, 1, 1)
        # self.b2_conv1_rgb = nn.Conv2d(128, 128, 3, padding=1)
        self.b2_conv1 = nn.Conv2d(256, 128, 3, padding=1)
        self.b2_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.b2_conv3 = nn.Conv2d(128, 1, 1)
        # self.b3_conv1_rgb = nn.Conv2d(256, 256, 5, padding=1)
        self.b3_conv1 = nn.Conv2d(512, 256, 5, padding=2)
        self.b3_conv2 = nn.Conv2d(256, 256, 5, padding=2)
        self.b3_conv3 = nn.Conv2d(256, 1, 1)
        # self.b4_conv1_rgb = nn.Conv2d(512, 256, 5, padding=1)
        self.b4_conv1 = nn.Conv2d(1024, 256, 5, padding=2)
        self.b4_conv2 = nn.Conv2d(256, 256, 5, padding=2)
        self.b4_conv3 = nn.Conv2d(256, 1, 1)
        # self.b5_conv1_rgb = nn.Conv2d(512, 512, 5, padding=1)
        self.b5_conv1 = nn.Conv2d(1024, 512, 5, padding=2)
        self.b5_conv2 = nn.Conv2d(512, 512, 5, padding=2)
        self.b5_conv3 = nn.Conv2d(512, 1, 1)
        # self.b6_conv1_rgb = nn.Conv2d(512, 512, 7, padding=1)
        self.b6_conv1 = nn.Conv2d(1024, 512, 7, padding=3)
        self.b6_conv2 = nn.Conv2d(512, 512, 7, padding=3)
        self.b6_conv3 = nn.Conv2d(512, 1, 1)
        
        self.b4_sc=nn.Conv2d(3, 1, 3, padding=1)
        self.b3_sc=nn.Conv2d(3, 1, 3, padding=1)
        self.b2_sc=nn.Conv2d(5, 1, 3, padding=1)
        self.b1_sc=nn.Conv2d(5, 1, 3, padding=1)
        
        self.fusion_conv = nn.Conv2d(6, 1, 1, padding=1)
 
<<<<<<< HEAD
    def forward(self, input_rgb, input_depth, state_trte, rgb_path="", d_path="", d_spx=[]):  #### fcn_2d ####
=======
    def forward(self, input_rgb, input_depth, state_trte):  #### fcn_2d ####
>>>>>>> dd86c05b6b13e46c37a8b50f8b4b326dda88ba1a
        # Max pooling over a (2, 2) window
        # activation function: Relu or Sigmoid (Later)
        #sequential: https://www.itread01.com/content/1547079864.html
        #GPUs
        device=self.device
        input_rgb=input_rgb.to(device).float()
        dataset_type=1
        
        ### RGB ##########################################
        rgb_conv1_1 = F.relu(self.rgb_conv1_1(input_rgb), inplace=True).to(device)
        rgb_conv1_2 = F.relu(self.rgb_conv1_2(rgb_conv1_1), inplace=True).to(device)
        rgb_pool1 = self.rgb_pool_1(rgb_conv1_2).to(device)
        
        rgb_conv2_1 = F.relu(self.rgb_conv2_1(rgb_pool1), inplace=True).to(device)
        rgb_conv2_2 = F.relu(self.rgb_conv2_2(rgb_conv2_1), inplace=True).to(device)
        rgb_pool2 = self.rgb_pool_2(rgb_conv2_2).to(device)
        
        rgb_conv3_1 = F.relu(self.rgb_conv3_1(rgb_pool2), inplace=True).to(device)
        rgb_conv3_2 = F.relu(self.rgb_conv3_2(rgb_conv3_1), inplace=True).to(device)
        rgb_conv3_3 = F.relu(self.rgb_conv3_3(rgb_conv3_2), inplace=True).to(device)
        rgb_pool3 = self.rgb_pool_3(rgb_conv3_3).to(device)    
        
        rgb_conv4_1 = F.relu(self.rgb_conv4_1(rgb_pool3), inplace=True).to(device)
        rgb_conv4_2 = F.relu(self.rgb_conv4_2(rgb_conv4_1), inplace=True).to(device)
        rgb_conv4_3 = F.relu(self.rgb_conv4_3(rgb_conv4_2), inplace=True).to(device)
        rgb_pool4 = self.rgb_pool_4(rgb_conv4_3).to(device) 
        
        rgb_conv5_1 = F.relu(self.rgb_conv5_1(rgb_pool4), inplace=True).to(device)
        rgb_conv5_2 = F.relu(self.rgb_conv5_2(rgb_conv5_1), inplace=True).to(device)
        rgb_conv5_3 = F.relu(self.rgb_conv5_3(rgb_conv5_2), inplace=True).to(device)
        rgb_pool5 = self.rgb_pool_2(rgb_conv5_3).to(device) 

        # rgb branches
        b1_rgb = rgb_conv1_2 
        b2_rgb = rgb_conv2_2 
        b3_rgb = rgb_conv3_3 
        b4_rgb = rgb_conv4_3
        b5_rgb = rgb_conv5_3 
        b6_rgb = rgb_pool5 

        b1_concat = b1_rgb
        b2_concat = b2_rgb
        b3_concat = b3_rgb
        b4_concat = b4_rgb
        b5_concat = b5_rgb
        b6_concat = b6_rgb

        
        if(dataset_type):
            #GPUs
            input_depth=input_depth.to(device).float()
            ### DEPTH ##########################################
            depth_conv1_1 = F.relu(self.depth_conv1_1(input_depth), inplace=True).to(device)
            depth_conv1_2 = F.relu(self.depth_conv1_2(depth_conv1_1), inplace=True).to(device)
            depth_pool1 = self.depth_pool_1(depth_conv1_2).to(device)
            
            depth_conv2_1 = F.relu(self.depth_conv2_1(depth_pool1), inplace=True).to(device)
            depth_conv2_2 = F.relu(self.depth_conv2_2(depth_conv2_1), inplace=True).to(device)
            depth_pool2 = self.depth_pool_2(depth_conv2_2).to(device)    
            
            depth_conv3_1 = F.relu(self.depth_conv3_1(depth_pool2), inplace=True).to(device)
            depth_conv3_2 = F.relu(self.depth_conv3_2(depth_conv3_1), inplace=True).to(device)
            depth_conv3_3 = F.relu(self.depth_conv3_3(depth_conv3_2), inplace=True).to(device)
            depth_pool3 = self.depth_pool_3(depth_conv3_3).to(device)     
            
            depth_conv4_1 = F.relu(self.depth_conv4_1(depth_pool3), inplace=True).to(device)
            depth_conv4_2 = F.relu(self.depth_conv4_2(depth_conv4_1), inplace=True).to(device)
            depth_conv4_3 = F.relu(self.depth_conv4_3(depth_conv4_2), inplace=True).to(device)
            depth_pool4 = self.depth_pool_4(depth_conv4_3).to(device) 
            
            depth_conv5_1 = F.relu(self.depth_conv5_1(depth_pool4), inplace=True).to(device)
            depth_conv5_2 = F.relu(self.depth_conv5_2(depth_conv5_1), inplace=True).to(device)
            depth_conv5_3 = F.relu(self.depth_conv5_3(depth_conv5_2), inplace=True).to(device)
            depth_pool5 = self.depth_pool_2(depth_conv5_3).to(device).to(device) 
    
            # depth branches
            b1_depth=depth_conv1_2 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu1_2')
            b2_depth=depth_conv2_2 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu2_2')
            b3_depth=depth_conv3_3 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu3_3')
            b4_depth=depth_conv4_3 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu4_3')
            b5_depth=depth_conv5_3 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu5_3')
            b6_depth=depth_pool5 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_dpool5')

            ################################
            # short connection: concatenate
            # concat the axis = 1  dim=[0, "1", 2, 3]
            ################################
            b1_concat=torch.cat((b1_rgb,b1_depth),1).to(device)
            b2_concat=torch.cat((b2_rgb,b2_depth),1).to(device)
            b3_concat=torch.cat((b3_rgb,b3_depth),1).to(device)
            b4_concat=torch.cat((b4_rgb,b4_depth),1).to(device)
            b5_concat=torch.cat((b5_rgb,b5_depth),1).to(device)
            b6_concat=torch.cat((b6_rgb,b6_depth),1).to(device)
            

        ################################
        # short connection: to side activation
        ################################
        if(dataset_type):
            b1_conv1 = F.relu(self.b1_conv1(b1_concat), inplace=True).to(device)
            b2_conv1 = F.relu(self.b2_conv1(b2_concat), inplace=True).to(device)
            b3_conv1 = F.relu(self.b3_conv1(b3_concat), inplace=True).to(device)
            b4_conv1 = F.relu(self.b4_conv1(b4_concat), inplace=True).to(device)
            b5_conv1 = F.relu(self.b5_conv1(b5_concat), inplace=True).to(device)
            b6_conv1 = F.relu(self.b6_conv1(b6_concat), inplace=True).to(device)
        else:
            b1_conv1 = F.relu(self.b1_conv1_rgb(b1_concat), inplace=True).to(device)
            b2_conv1 = F.relu(self.b2_conv1_rgb(b2_concat), inplace=True).to(device)
            b3_conv1 = F.relu(self.b3_conv1_rgb(b3_concat), inplace=True).to(device)
            b4_conv1 = F.relu(self.b4_conv1_rgb(b4_concat), inplace=True).to(device)
            b5_conv1 = F.relu(self.b5_conv1_rgb(b5_concat), inplace=True).to(device)
            b6_conv1 = F.relu(self.b6_conv1_rgb(b6_concat), inplace=True).to(device)
        
        #branch 1
        b1_conv2 = F.relu(self.b1_conv2(b1_conv1), inplace=True).to(device)
        b1_conv3 = self.b1_conv3(b1_conv2).to(device)
        #branch 2
        b2_conv2 = F.relu(self.b2_conv2(b2_conv1), inplace=True).to(device)
        b2_conv3 = self.b2_conv3(b2_conv2).to(device)
        #branch 3
        b3_conv2 = F.relu(self.b3_conv2(b3_conv1), inplace=True).to(device)
        b3_conv3 = self.b3_conv3(b3_conv2).to(device)
        #branch 4
        b4_conv2 = F.relu(self.b4_conv2(b4_conv1), inplace=True).to(device)
        b4_conv3 = self.b4_conv3(b4_conv2).to(device)
        #branch 5
        b5_conv2 = F.relu(self.b5_conv2(b5_conv1), inplace=True).to(device)
        b5_conv3 = self.b5_conv3(b5_conv2).to(device)
        #branch 6
        b6_conv2 = F.relu(self.b6_conv2(b6_conv1), inplace=True).to(device)
        b6_conv3 = self.b6_conv3(b6_conv2).to(device)
            

        ################################
        # short connection: branch results
        ################################        
        # shortconnections to Aside 6
        r6_conv1 = b6_conv3
        # Rside 6
        
        # shortconnections to Aside 5
        r5_conv1 = b5_conv3
        # Rside 5
        
        # shortconnections to Aside 4
        # resize_r4=nn.ConvTranspose2d(1, 1, k * 2, k, k // 2)
        r5_conv1_resize_r4 = F.interpolate(r5_conv1, [b4_conv3.shape[3],b4_conv3.shape[2]])
        r6_conv1_resize_r4 = F.interpolate(r6_conv1, [b4_conv3.shape[3],b4_conv3.shape[2]]).to(device)
        #print(r5_conv1_resize_r4.shape) #1, 1, 72, 72
        #print(r6_conv1_resize_r4.shape) #1, 1, 48, 48
        #print(b4_conv3.shape) #1, 1, 76, 76
        r4_concat = torch.cat((b4_conv3, r5_conv1_resize_r4, r6_conv1_resize_r4),1).to(device)
        r4_conv1 = self.b4_sc(r4_concat).to(device)
        # Rside 4
        # shortconnections to Aside 3
        # resize_r3=nn.ConvTranspose2d(1, 1, k * 2, k, k // 2)
        r5_conv1_resize_r3 = F.interpolate(r5_conv1, [b3_conv3.shape[3],b3_conv3.shape[2]]).to(device)
        r6_conv1_resize_r3 = F.interpolate(r6_conv1, [b3_conv3.shape[3],b3_conv3.shape[2]]).to(device)
        r3_concat = torch.cat((b3_conv3, r5_conv1_resize_r3, r6_conv1_resize_r3),1).to(device)
        r3_conv1 = self.b3_sc(r3_concat).to(device)
        # Rside 3
        # short connections to Aside 2
        # resize_r2=nn.ConvTranspose2d(1, 1, k * 2, k, k // 2)
        r3_conv1_resize_r2 = F.interpolate(r3_conv1, [b2_conv3.shape[3],b2_conv3.shape[2]]).to(device)
        r4_conv1_resize_r2 = F.interpolate(r4_conv1, [b2_conv3.shape[3],b2_conv3.shape[2]]).to(device)
        r5_conv1_resize_r2 = F.interpolate(r5_conv1, [b2_conv3.shape[3],b2_conv3.shape[2]]).to(device)
        r6_conv1_resize_r2 = F.interpolate(r6_conv1, [b2_conv3.shape[3],b2_conv3.shape[2]]).to(device)
        r2_concat = torch.cat((b2_conv3, r3_conv1_resize_r2, r4_conv1_resize_r2, r5_conv1_resize_r2, r6_conv1_resize_r2),1).to(device)
        r2_conv1 = self.b2_sc(r2_concat).to(device)
        # Rside 3
        # short connections to Aside 1
        # resize_r1=nn.ConvTranspose2d(1, 1, k * 2, k, k // 2)
        r3_conv1_resize_r1 = F.interpolate(r3_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r4_conv1_resize_r1 = F.interpolate(r4_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r5_conv1_resize_r1 = F.interpolate(r5_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r6_conv1_resize_r1 = F.interpolate(r6_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r1_concat = torch.cat((b1_conv3, r3_conv1_resize_r1, r4_conv1_resize_r1, r5_conv1_resize_r1, r6_conv1_resize_r1),1).to(device)
        r1_conv1 = self.b1_sc(r1_concat).to(device)
        # Rside 1       
        ################################
        # short connection: FUSION
        ################################        
        img_size=[ b1_conv3.shape[3],b1_conv3.shape[2] ]
        # fuse used branch results
        r1_conv1_resize = r1_conv1 #nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1,  1))(r1_conv1)#F.interpolate(r1_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r2_conv1_resize = F.interpolate(r2_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r3_conv1_resize = F.interpolate(r3_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r4_conv1_resize = F.interpolate(r4_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r5_conv1_resize = F.interpolate(r5_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r6_conv1_resize = F.interpolate(r6_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        
        #fusion_concat=torch.cat((r2_conv1_resize,r3_conv1_resize,r4_conv1_resize],1).to(device)
        # STAT_TRAIN = 1 / STAT_TEST = 2
        if(state_trte==STAT_TRAIN): # 6 Channel all in
          fusion_concat=torch.cat((r1_conv1_resize,r2_conv1_resize,r3_conv1_resize,r4_conv1_resize,r5_conv1_resize,r6_conv1_resize),1).to(device)
          fusion_result=self.fusion_conv(fusion_concat).to(device)
        elif(state_trte==STAT_TEST): 
          method=1
          if(method==1): # method 1: 3 channel / 3 zeros
            r1_conv1_zero=torch.zeros(r1_conv1_resize.shape).to(device)
            r5_conv1_zero=torch.zeros(r5_conv1_resize.shape).to(device)
            r6_conv1_zero=torch.zeros(r6_conv1_resize.shape).to(device)
            fusion_concat=torch.cat((r1_conv1_zero,r2_conv1_resize,r3_conv1_resize,r4_conv1_resize,r5_conv1_zero,r6_conv1_zero),1).to(device)
            fusion_result=self.fusion_conv(fusion_concat).to(device)
          elif(method==2): # method 2: 3 channel copy
            fusion_concat=torch.cat((r2_conv1_resize,r2_conv1_resize,r3_conv1_resize,r4_conv1_resize,r3_conv1_resize,r4_conv1_resize),1).to(device)
            fusion_result=self.fusion_conv(fusion_concat).to(device)
          elif(method==3): # method 3: 3b conv
            fusion_concat=torch.cat((r2_conv1_resize,r3_conv1_resize,r4_conv1_resize),1).to(device)
            ## print(self.state_dict()['fusion_conv.weight'].shape) # [1,6,1,1]
            temp_weight=self.state_dict()['fusion_conv.weight']
            temp_weight=temp_weight[:,1:4,:,:] # b2,3,4 params
            ## print(temp_weight.shape)
            temp_bias = self.state_dict()['fusion_conv.bias'] #[:,:,:] #dim =[1,1,1]
            fusion_result=F.conv2d(fusion_concat, weight=temp_weight , bias= temp_bias, padding=1)#self.fusion_conv_inf(fusion_concat).to(device)        
        
        return fusion_result, b1_conv3, b2_conv3, b3_conv3, b4_conv3, b5_conv3, b6_conv3


# ===============================================================================
# ===============================================================================
# ===============================================================================
class SPNet(nn.Module):
    def __init__(self, gpu_device=torch.device("cpu"),start_EP=0):
        super(SPNet, self).__init__()
        self.vgg_init=False
        self.device=gpu_device
        self.epoch=start_EP
        self.fcn_1d()

    def RGB_fcn_1d(self):
        # RGB part
        self.rgb_conv1_1 = nn.Conv1d(3, 64, 3, padding=1)
        self.rgb_conv1_2 = nn.Conv1d(64, 64, 3, padding=1)
        self.rgb_pool_1 = nn.MaxPool1d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv2_1 = nn.Conv1d(64, 128, 3, padding=1)
        self.rgb_conv2_2 = nn.Conv1d(128, 128, 3, padding=1)
        self.rgb_pool_2 = nn.MaxPool1d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv3_1 = nn.Conv1d(128, 256, 3, padding=1)
        self.rgb_conv3_2 = nn.Conv1d(256, 256, 3, padding=1)  
        self.rgb_conv3_3 = nn.Conv1d(256, 256, 3, padding=1)
        self.rgb_pool_3 = nn.MaxPool1d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv4_1 = nn.Conv1d(256, 512, 3, padding=1)
        self.rgb_conv4_2 = nn.Conv1d(512, 512, 3, padding=1) 
        self.rgb_conv4_3 = nn.Conv1d(512, 512, 3, padding=1)    
        self.rgb_pool_4 = nn.MaxPool1d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv5_1 = nn.Conv1d(512, 512, 3, padding=1)
        self.rgb_conv5_2 = nn.Conv1d(512, 512, 3, padding=1)   
        self.rgb_conv5_3 = nn.Conv1d(512, 512, 3, padding=1)
        self.rgb_pool_5 =  nn.MaxPool1d(2, 2, dilation=1, ceil_mode=False)    

    def depth_fcn_1d(self):    
        # Depth part
        
        self.depth_conv1_1 = nn.Conv1d(1, 64, 3, padding=1)
        self.depth_conv1_2 = nn.Conv1d(64, 64, 3, padding=1)
        self.depth_pool_1 = nn.MaxPool1d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv2_1 = nn.Conv1d(64, 128, 3, padding=1)
        self.depth_conv2_2 = nn.Conv1d(128, 128, 3, padding=1)
        self.depth_pool_2 = nn.MaxPool1d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv3_1 = nn.Conv1d(128, 256, 3, padding=1)
        self.depth_conv3_2 = nn.Conv1d(256, 256, 3, padding=1)   
        self.depth_conv3_3 = nn.Conv1d(256, 256, 3, padding=1)
        self.depth_pool_3 = nn.MaxPool1d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv4_1 = nn.Conv1d(256, 512, 3, padding=1)
        self.depth_conv4_2 = nn.Conv1d(512, 512, 3, padding=1)   
        self.depth_conv4_3 = nn.Conv1d(512, 512, 3, padding=1)    
        self.depth_pool_4 = nn.MaxPool1d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv5_1 = nn.Conv1d(512, 512, 3, padding=1)
        self.depth_conv5_2 = nn.Conv1d(512, 512, 3, padding=1)  
        self.depth_conv5_3 = nn.Conv1d(512, 512, 3, padding=1)
        self.depth_pool_5 =  nn.MaxPool1d(2, 2, dilation=1, ceil_mode=False) 

    
    def fcn_1d(self):
        self.RGB_fcn_1d()
        self.depth_fcn_1d()
        if(self.vgg_init and self.epoch==0):
          weight_init(self)
        # print(self.state_dict())
        
        # short connection convolutions #RGBD #RGB (input channels are different)       
        
        self.b1_conv1_rgb = nn.Conv1d(64, 128, 3, padding=1) #no depth features to concat
        self.b1_conv1 = nn.Conv1d(128, 128, 3, padding=1) #has depth features to concat
        self.b1_conv2 = nn.Conv1d(128, 128, 3, padding=1)
        self.b1_conv3 = nn.Conv1d(128, 1, 1)
        
        self.b2_conv1_rgb = nn.Conv1d(128, 128, 3, padding=1)
        self.b2_conv1 = nn.Conv1d(256, 128, 3, padding=1)
        self.b2_conv2 = nn.Conv1d(128, 128, 3, padding=1)
        self.b2_conv3 = nn.Conv1d(128, 1, 1)
        
        self.b3_conv1_rgb = nn.Conv1d(256, 256, 5, padding=1)
        self.b3_conv1 = nn.Conv1d(512, 256, 5, padding=2)
        self.b3_conv2 = nn.Conv1d(256, 256, 5, padding=2)
        self.b3_conv3 = nn.Conv1d(256, 1, 1)
        
        self.b4_conv1_rgb = nn.Conv1d(512, 256, 5, padding=1)
        self.b4_conv1 = nn.Conv1d(1024, 256, 5, padding=2)
        self.b4_conv2 = nn.Conv1d(256, 256, 5, padding=2)
        self.b4_conv3 = nn.Conv1d(256, 1, 1)
        
        self.b5_conv1_rgb = nn.Conv1d(512, 512, 5, padding=1)
        self.b5_conv1 = nn.Conv1d(1024, 512, 5, padding=2)
        self.b5_conv2 = nn.Conv1d(512, 512, 5, padding=2)
        self.b5_conv3 = nn.Conv1d(512, 1, 1)
        
        self.b6_conv1_rgb = nn.Conv1d(512, 512, 7, padding=1)
        self.b6_conv1 = nn.Conv1d(1024, 512, 7, padding=3)
        self.b6_conv2 = nn.Conv1d(512, 512, 7, padding=3)
        self.b6_conv3 = nn.Conv1d(512, 1, 1)
        
        self.up_t64=nn.ConvTranspose1d(1, 1, kernel_size=64, stride=32, padding=16)
        self.up_t32=nn.ConvTranspose1d(1, 1, kernel_size=32, stride=16, padding=8)
        self.up_t16=nn.ConvTranspose1d(1, 1, kernel_size=16, stride=8, padding=4) 
        self.up_t8=nn.ConvTranspose1d(1, 1, kernel_size=8, stride=4, padding=2)
        self.up_t4=nn.ConvTranspose1d(1, 1, kernel_size=4, stride=2, padding=1)
        
        self.b4_sc=nn.Conv1d(3, 1, 3, padding=1)
        self.b3_sc=nn.Conv1d(3, 1, 3, padding=1)
        self.b2_sc=nn.Conv1d(5, 1, 3, padding=1)
        self.b1_sc=nn.Conv1d(5, 1, 3, padding=1)
        
        self.fusion_conv = nn.Conv1d(6, 1, 1, padding=1)
        # self.fusion_conv_inf = nn.Conv1d(3, 1, 1)
        # self.out_sig=nn.Sigmoid()

<<<<<<< HEAD
    def forward(self, input_rgb, input_depth,dataset_type, rgb_path="", d_path="", d_spx=[]): #### fcn_1d ####
=======
    def forward(self, input_rgb, input_depth,dataset_type): #### fcn_1d ####
>>>>>>> dd86c05b6b13e46c37a8b50f8b4b326dda88ba1a
        # Max pooling over a (2, 2) window
        # activation function: Relu or Sigmoid (Later)
        #sequential: https://www.itread01.com/content/1547079864.html
        #GPUs
        device=self.device
        input_rgb=input_rgb.to(device).float()
        dataset_type=1
        
        ### RGB ##########################################
        rgb_conv1_1 = F.relu(self.rgb_conv1_1(input_rgb), inplace=True).to(device)
        rgb_conv1_2 = F.relu(self.rgb_conv1_2(rgb_conv1_1), inplace=True).to(device)
        rgb_pool1 = self.rgb_pool_1(rgb_conv1_2).to(device)
        
        rgb_conv2_1 = F.relu(self.rgb_conv2_1(rgb_pool1), inplace=True).to(device)
        rgb_conv2_2 = F.relu(self.rgb_conv2_2(rgb_conv2_1), inplace=True).to(device)
        rgb_pool2 = self.rgb_pool_2(rgb_conv2_2).to(device)
        
        rgb_conv3_1 = F.relu(self.rgb_conv3_1(rgb_pool2), inplace=True).to(device)
        rgb_conv3_2 = F.relu(self.rgb_conv3_2(rgb_conv3_1), inplace=True).to(device)
        rgb_conv3_3 = F.relu(self.rgb_conv3_3(rgb_conv3_2), inplace=True).to(device)
        rgb_pool3 = self.rgb_pool_3(rgb_conv3_3).to(device)    
        
        rgb_conv4_1 = F.relu(self.rgb_conv4_1(rgb_pool3), inplace=True).to(device)
        rgb_conv4_2 = F.relu(self.rgb_conv4_2(rgb_conv4_1), inplace=True).to(device)
        rgb_conv4_3 = F.relu(self.rgb_conv4_3(rgb_conv4_2), inplace=True).to(device)
        rgb_pool4 = self.rgb_pool_4(rgb_conv4_3).to(device) 
        
        rgb_conv5_1 = F.relu(self.rgb_conv5_1(rgb_pool4), inplace=True).to(device)
        rgb_conv5_2 = F.relu(self.rgb_conv5_2(rgb_conv5_1), inplace=True).to(device)
        rgb_conv5_3 = F.relu(self.rgb_conv5_3(rgb_conv5_2), inplace=True).to(device)
        rgb_pool5 = self.rgb_pool_2(rgb_conv5_3).to(device) 

        # rgb branches
        b1_rgb = rgb_conv1_2 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu1_2')
        b2_rgb = rgb_conv2_2 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu2_2')
        b3_rgb = rgb_conv3_3 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu3_3')
        b4_rgb = rgb_conv4_3 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu4_3')
        b5_rgb = rgb_conv5_3 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu5_3')
        b6_rgb = rgb_pool5 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_pool5')

        b1_concat = b1_rgb
        b2_concat = b2_rgb
        b3_concat = b3_rgb
        b4_concat = b4_rgb
        b5_concat = b5_rgb
        b6_concat = b6_rgb

        
        if(dataset_type):
            #GPUs
            input_depth=input_depth.to(device).float()
            ### DEPTH ##########################################
            depth_conv1_1 = F.relu(self.depth_conv1_1(input_depth), inplace=True).to(device)
            depth_conv1_2 = F.relu(self.depth_conv1_2(depth_conv1_1), inplace=True).to(device)
            depth_pool1 = self.depth_pool_1(depth_conv1_2).to(device)
            
            depth_conv2_1 = F.relu(self.depth_conv2_1(depth_pool1), inplace=True).to(device)
            depth_conv2_2 = F.relu(self.depth_conv2_2(depth_conv2_1), inplace=True).to(device)
            depth_pool2 = self.depth_pool_2(depth_conv2_2).to(device)    
            
            depth_conv3_1 = F.relu(self.depth_conv3_1(depth_pool2), inplace=True).to(device)
            depth_conv3_2 = F.relu(self.depth_conv3_2(depth_conv3_1), inplace=True).to(device)
            depth_conv3_3 = F.relu(self.depth_conv3_3(depth_conv3_2), inplace=True).to(device)
            depth_pool3 = self.depth_pool_3(depth_conv3_3).to(device)     
            
            depth_conv4_1 = F.relu(self.depth_conv4_1(depth_pool3), inplace=True).to(device)
            depth_conv4_2 = F.relu(self.depth_conv4_2(depth_conv4_1), inplace=True).to(device)
            depth_conv4_3 = F.relu(self.depth_conv4_3(depth_conv4_2), inplace=True).to(device)
            depth_pool4 = self.depth_pool_4(depth_conv4_3).to(device) 
            
            depth_conv5_1 = F.relu(self.depth_conv5_1(depth_pool4), inplace=True).to(device)
            depth_conv5_2 = F.relu(self.depth_conv5_2(depth_conv5_1), inplace=True).to(device)
            depth_conv5_3 = F.relu(self.depth_conv5_3(depth_conv5_2), inplace=True).to(device)
            depth_pool5 = self.depth_pool_2(depth_conv5_3).to(device).to(device) 
    
            # depth branches
            b1_depth=depth_conv1_2 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu1_2')
            b2_depth=depth_conv2_2 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu2_2')
            b3_depth=depth_conv3_3 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu3_3')
            b4_depth=depth_conv4_3 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu4_3')
            b5_depth=depth_conv5_3 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu5_3')
            b6_depth=depth_pool5 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_dpool5')

            ################################
            # short connection: concatenate
            # concat the axis = 1  dim=[0, "1", 2, 3]
            ################################
            b1_concat=torch.cat((b1_rgb,b1_depth),1).to(device)
            b2_concat=torch.cat((b2_rgb,b2_depth),1).to(device)
            b3_concat=torch.cat((b3_rgb,b3_depth),1).to(device)
            b4_concat=torch.cat((b4_rgb,b4_depth),1).to(device)
            b5_concat=torch.cat((b5_rgb,b5_depth),1).to(device)
            b6_concat=torch.cat((b6_rgb,b6_depth),1).to(device)
            

        ################################
        # short connection: to side activation
        ################################
        if(dataset_type):
            b1_conv1 = F.relu(self.b1_conv1(b1_concat), inplace=True).to(device)
            b2_conv1 = F.relu(self.b2_conv1(b2_concat), inplace=True).to(device)
            b3_conv1 = F.relu(self.b3_conv1(b3_concat), inplace=True).to(device)
            b4_conv1 = F.relu(self.b4_conv1(b4_concat), inplace=True).to(device)
            b5_conv1 = F.relu(self.b5_conv1(b5_concat), inplace=True).to(device)
            b6_conv1 = F.relu(self.b6_conv1(b6_concat), inplace=True).to(device)
        else:
            b1_conv1 = F.relu(self.b1_conv1_rgb(b1_concat), inplace=True).to(device)
            b2_conv1 = F.relu(self.b2_conv1_rgb(b2_concat), inplace=True).to(device)
            b3_conv1 = F.relu(self.b3_conv1_rgb(b3_concat), inplace=True).to(device)
            b4_conv1 = F.relu(self.b4_conv1_rgb(b4_concat), inplace=True).to(device)
            b5_conv1 = F.relu(self.b5_conv1_rgb(b5_concat), inplace=True).to(device)
            b6_conv1 = F.relu(self.b6_conv1_rgb(b6_concat), inplace=True).to(device)
        
        #branch 1
        b1_conv2 = F.relu(self.b1_conv2(b1_conv1), inplace=True).to(device)
        b1_conv3 = self.b1_conv3(b1_conv2).to(device)
        #branch 2
        b2_conv2 = F.relu(self.b2_conv2(b2_conv1), inplace=True).to(device)
        b2_conv3 = self.b2_conv3(b2_conv2).to(device)
        #branch 3
        b3_conv2 = F.relu(self.b3_conv2(b3_conv1), inplace=True).to(device)
        b3_conv3 = self.b3_conv3(b3_conv2).to(device)
        #branch 4
        b4_conv2 = F.relu(self.b4_conv2(b4_conv1), inplace=True).to(device)
        b4_conv3 = self.b4_conv3(b4_conv2).to(device)
        #branch 5
        b5_conv2 = F.relu(self.b5_conv2(b5_conv1), inplace=True).to(device)
        b5_conv3 = self.b5_conv3(b5_conv2).to(device)
        #branch 6
        b6_conv2 = F.relu(self.b6_conv2(b6_conv1), inplace=True).to(device)
        b6_conv3 = self.b6_conv3(b6_conv2).to(device)
            

        ################################
        # short connection: branch results
        ################################        
        # shortconnections to Aside 6
        r6_conv1 = b6_conv3
        # Rside 6
        
        # shortconnections to Aside 5
        r5_conv1 = b5_conv3
        # Rside 5
        
        # shortconnections to Aside 4
        # print(r6_conv1.shape)
        # print(r5_conv1.shape)
        
        r5_conv1_resize_r4 = self.up_t4(r5_conv1).to(device)#F.interpolate(r5_conv1, [b4_conv3.shape[3],b4_conv3.shape[2]])
        r6_conv1_resize_r4 = self.up_t8(r6_conv1).to(device)#F.interpolate(r6_conv1, [b4_conv3.shape[3],b4_conv3.shape[2]]).to(device)
        #print(r5_conv1_resize_r4.shape) #1, 1, 72, 72
        #print(r6_conv1_resize_r4.shape) #1, 1, 48, 48
        #print(b4_conv3.shape) #1, 1, 76, 76
        r4_concat = torch.cat((b4_conv3, r5_conv1_resize_r4, r6_conv1_resize_r4),1).to(device)
        r4_conv1 = self.b4_sc(r4_concat).to(device)
        # Rside 4
        # shortconnections to Aside 3
        r5_conv1_resize_r3 = self.up_t8(r5_conv1).to(device)#F.interpolate(r5_conv1, [b3_conv3.shape[3],b3_conv3.shape[2]]).to(device)
        r6_conv1_resize_r3 = self.up_t16(r6_conv1).to(device)#F.interpolate(r6_conv1, [b3_conv3.shape[3],b3_conv3.shape[2]]).to(device)
        r3_concat = torch.cat((b3_conv3, r5_conv1_resize_r3, r6_conv1_resize_r3),1).to(device)
        r3_conv1 = self.b3_sc(r3_concat).to(device)
        # Rside 3
        # short connections to Aside 2
        r3_conv1_resize_r2 = self.up_t4(r3_conv1).to(device)#F.interpolate(r3_conv1, [b2_conv3.shape[3],b2_conv3.shape[2]]).to(device)
        r4_conv1_resize_r2 = self.up_t8(r4_conv1).to(device)#F.interpolate(r4_conv1, [b2_conv3.shape[3],b2_conv3.shape[2]]).to(device)
        r5_conv1_resize_r2 = self.up_t16(r5_conv1).to(device)#F.interpolate(r5_conv1, [b2_conv3.shape[3],b2_conv3.shape[2]]).to(device)
        r6_conv1_resize_r2 = self.up_t32(r6_conv1).to(device)#F.interpolate(r6_conv1, [b2_conv3.shape[3],b2_conv3.shape[2]]).to(device)
        r2_concat = torch.cat((b2_conv3, r3_conv1_resize_r2, r4_conv1_resize_r2, r5_conv1_resize_r2, r6_conv1_resize_r2),1).to(device)
        r2_conv1 = self.b2_sc(r2_concat).to(device)
        # Rside 3
        # short connections to Aside 1
        r3_conv1_resize_r1 = self.up_t8(r3_conv1).to(device)#F.interpolate(r3_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r4_conv1_resize_r1 = self.up_t16(r4_conv1).to(device)#F.interpolate(r4_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r5_conv1_resize_r1 = self.up_t32(r5_conv1).to(device)#F.interpolate(r5_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r6_conv1_resize_r1 = self.up_t64(r6_conv1).to(device)#F.interpolate(r6_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r1_concat = torch.cat((b1_conv3, r3_conv1_resize_r1, r4_conv1_resize_r1, r5_conv1_resize_r1, r6_conv1_resize_r1),1).to(device)
        r1_conv1 = self.b1_sc(r1_concat).to(device)
        # Rside 1       
        ################################
        # short connection: FUSION
        ################################        
        # img_size=[ b1_conv3.shape[3]*b1_conv3.shape[2] ]
        # fuse used branch results
        r1_conv1_resize = r1_conv1 
        r2_conv1_resize = self.up_t4(r2_conv1).to(device)#F.interpolate(r2_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r3_conv1_resize = self.up_t8(r3_conv1).to(device)#F.interpolate(r3_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r4_conv1_resize = self.up_t16(r4_conv1).to(device)#F.interpolate(r4_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r5_conv1_resize = self.up_t32(r5_conv1).to(device)#F.interpolate(r5_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        r6_conv1_resize = self.up_t64(r6_conv1).to(device)#F.interpolate(r6_conv1, [b1_conv3.shape[3],b1_conv3.shape[2]]).to(device)
        
        #fusion_concat=torch.cat((r2_conv1_resize,r3_conv1_resize,r4_conv1_resize],1).to(device)
        fusion_concat=torch.cat((r1_conv1_resize,r2_conv1_resize,r3_conv1_resize,r4_conv1_resize,r5_conv1_resize,r6_conv1_resize),1).to(device)
        fusion_result=self.fusion_conv(fusion_concat).to(device)
        

        return fusion_result, b1_conv3, b2_conv3, b3_conv3, b4_conv3, b5_conv3, b6_conv3
##=============================SP ver====================================================================================================
# SPNet_interpolation
class SPNet_interpolation(nn.Module):
    def __init__(self, gpu_device=torch.device("cpu"),start_EP=0):
        super(SPNet2, self).__init__()
        self.vgg_init=True
        self.device=gpu_device
        self.epoch=start_EP
        self.fcn_1d()

    def RGB_fcn_1d(self):
        # RGB part
        self.rgb_conv1_1 = nn.Conv1d(3, 64, 3, padding=1)
        self.rgb_conv1_2 = nn.Conv1d(64, 64, 3, padding=1)
        self.rgb_pool_1 = nn.MaxPool1d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv2_1 = nn.Conv1d(64, 128, 3, padding=1)
        self.rgb_conv2_2 = nn.Conv1d(128, 128, 3, padding=1)
        self.rgb_pool_2 = nn.MaxPool1d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv3_1 = nn.Conv1d(128, 256, 3, padding=1)
        self.rgb_conv3_2 = nn.Conv1d(256, 256, 3, padding=1)  
        self.rgb_conv3_3 = nn.Conv1d(256, 256, 3, padding=1)
        self.rgb_pool_3 = nn.MaxPool1d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv4_1 = nn.Conv1d(256, 512, 3, padding=1)
        self.rgb_conv4_2 = nn.Conv1d(512, 512, 3, padding=1) 
        self.rgb_conv4_3 = nn.Conv1d(512, 512, 3, padding=1)    
        self.rgb_pool_4 = nn.MaxPool1d(2, 2, dilation=1, ceil_mode=False)
        
        self.rgb_conv5_1 = nn.Conv1d(512, 512, 3, padding=1)
        self.rgb_conv5_2 = nn.Conv1d(512, 512, 3, padding=1)   
        self.rgb_conv5_3 = nn.Conv1d(512, 512, 3, padding=1)
        self.rgb_pool_5 =  nn.MaxPool1d(2, 2, dilation=1, ceil_mode=False)    

    def depth_fcn_1d(self):    
        # Depth part
        self.depth_conv1_1 = nn.Conv1d(1, 64, 3, padding=1)
        self.depth_conv1_2 = nn.Conv1d(64, 64, 3, padding=1)
        self.depth_pool_1 = nn.MaxPool1d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv2_1 = nn.Conv1d(64, 128, 3, padding=1)
        self.depth_conv2_2 = nn.Conv1d(128, 128, 3, padding=1)
        self.depth_pool_2 = nn.MaxPool1d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv3_1 = nn.Conv1d(128, 256, 3, padding=1)
        self.depth_conv3_2 = nn.Conv1d(256, 256, 3, padding=1)   
        self.depth_conv3_3 = nn.Conv1d(256, 256, 3, padding=1)
        self.depth_pool_3 = nn.MaxPool1d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv4_1 = nn.Conv1d(256, 512, 3, padding=1)
        self.depth_conv4_2 = nn.Conv1d(512, 512, 3, padding=1)   
        self.depth_conv4_3 = nn.Conv1d(512, 512, 3, padding=1)    
        self.depth_pool_4 = nn.MaxPool1d(2, 2, dilation=1, ceil_mode=False)
        
        self.depth_conv5_1 = nn.Conv1d(512, 512, 3, padding=1)
        self.depth_conv5_2 = nn.Conv1d(512, 512, 3, padding=1)  
        self.depth_conv5_3 = nn.Conv1d(512, 512, 3, padding=1)
        self.depth_pool_5 =  nn.MaxPool1d(2, 2, dilation=1, ceil_mode=False) 

    
    def fcn_1d(self):
        self.RGB_fcn_1d()
        self.depth_fcn_1d()
        if(self.vgg_init and self.epoch==0):
          weight_init(self)
        # print(self.state_dict())
        
        # short connection convolutions #RGBD #RGB (input channels are different)       
        
        self.b1_conv1_rgb = nn.Conv1d(64, 128, 3, padding=1) #no depth features to concat
        self.b1_conv1 = nn.Conv1d(128, 128, 3, padding=1) #has depth features to concat
        self.b1_conv2 = nn.Conv1d(128, 128, 3, padding=1)
        self.b1_conv3 = nn.Conv1d(128, 1, 1)
        
        self.b2_conv1_rgb = nn.Conv1d(128, 128, 3, padding=1)
        self.b2_conv1 = nn.Conv1d(256, 128, 3, padding=1)
        self.b2_conv2 = nn.Conv1d(128, 128, 3, padding=1)
        self.b2_conv3 = nn.Conv1d(128, 1, 1)
        
        self.b3_conv1_rgb = nn.Conv1d(256, 256, 5, padding=1)
        self.b3_conv1 = nn.Conv1d(512, 256, 5, padding=2)
        self.b3_conv2 = nn.Conv1d(256, 256, 5, padding=2)
        self.b3_conv3 = nn.Conv1d(256, 1, 1)
        
        self.b4_conv1_rgb = nn.Conv1d(512, 256, 5, padding=1)
        self.b4_conv1 = nn.Conv1d(1024, 256, 5, padding=2)
        self.b4_conv2 = nn.Conv1d(256, 256, 5, padding=2)
        self.b4_conv3 = nn.Conv1d(256, 1, 1)
        
        self.b5_conv1_rgb = nn.Conv1d(512, 512, 5, padding=1)
        self.b5_conv1 = nn.Conv1d(1024, 512, 5, padding=2)
        self.b5_conv2 = nn.Conv1d(512, 512, 5, padding=2)
        self.b5_conv3 = nn.Conv1d(512, 1, 1)
        
        self.b6_conv1_rgb = nn.Conv1d(512, 512, 7, padding=1)
        self.b6_conv1 = nn.Conv1d(1024, 512, 7, padding=3)
        self.b6_conv2 = nn.Conv1d(512, 512, 7, padding=3)
        self.b6_conv3 = nn.Conv1d(512, 1, 1)
        '''
        self.up_t64=nn.ConvTranspose1d(1, 1, kernel_size=64, stride=32, padding=16)
        self.up_t32=nn.ConvTranspose1d(1, 1, kernel_size=32, stride=16, padding=8)
        self.up_t16=nn.ConvTranspose1d(1, 1, kernel_size=16, stride=8, padding=4) 
        self.up_t8=nn.ConvTranspose1d(1, 1, kernel_size=8, stride=4, padding=2)
        self.up_t4=nn.ConvTranspose1d(1, 1, kernel_size=4, stride=2, padding=1)
        '''
        
        self.b4_sc=nn.Conv1d(3, 1, 3, padding=1)
        self.b3_sc=nn.Conv1d(3, 1, 3, padding=1)
        self.b2_sc=nn.Conv1d(5, 1, 3, padding=1)
        self.b1_sc=nn.Conv1d(5, 1, 3, padding=1)
        
        self.fusion_conv = nn.Conv1d(6, 1, 1, padding=1)
        # self.fusion_conv_inf = nn.Conv1d(3, 1, 1)
        # self.out_sig=nn.Sigmoid()

    def forward(self, input_rgb, input_depth): #### fcn_1d ####
        # Max pooling over a (2, 2) window
        # activation function: Relu or Sigmoid (Later)
        #sequential: https://www.itread01.com/content/1547079864.html
        #GPUs
        device=self.device
        input_rgb=input_rgb.to(device).float()
        dataset_type=1
        
        ### RGB ##########################################
        rgb_conv1_1 = F.relu(self.rgb_conv1_1(input_rgb), inplace=True).to(device)
        rgb_conv1_2 = F.relu(self.rgb_conv1_2(rgb_conv1_1), inplace=True).to(device)
        rgb_pool1 = self.rgb_pool_1(rgb_conv1_2).to(device)
        
        rgb_conv2_1 = F.relu(self.rgb_conv2_1(rgb_pool1), inplace=True).to(device)
        rgb_conv2_2 = F.relu(self.rgb_conv2_2(rgb_conv2_1), inplace=True).to(device)
        rgb_pool2 = self.rgb_pool_2(rgb_conv2_2).to(device)
        
        rgb_conv3_1 = F.relu(self.rgb_conv3_1(rgb_pool2), inplace=True).to(device)
        rgb_conv3_2 = F.relu(self.rgb_conv3_2(rgb_conv3_1), inplace=True).to(device)
        rgb_conv3_3 = F.relu(self.rgb_conv3_3(rgb_conv3_2), inplace=True).to(device)
        rgb_pool3 = self.rgb_pool_3(rgb_conv3_3).to(device)    
        
        rgb_conv4_1 = F.relu(self.rgb_conv4_1(rgb_pool3), inplace=True).to(device)
        rgb_conv4_2 = F.relu(self.rgb_conv4_2(rgb_conv4_1), inplace=True).to(device)
        rgb_conv4_3 = F.relu(self.rgb_conv4_3(rgb_conv4_2), inplace=True).to(device)
        rgb_pool4 = self.rgb_pool_4(rgb_conv4_3).to(device) 
        
        rgb_conv5_1 = F.relu(self.rgb_conv5_1(rgb_pool4), inplace=True).to(device)
        rgb_conv5_2 = F.relu(self.rgb_conv5_2(rgb_conv5_1), inplace=True).to(device)
        rgb_conv5_3 = F.relu(self.rgb_conv5_3(rgb_conv5_2), inplace=True).to(device)
        rgb_pool5 = self.rgb_pool_2(rgb_conv5_3).to(device) 

        # rgb branches
        b1_rgb = rgb_conv1_2 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu1_2')
        b2_rgb = rgb_conv2_2 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu2_2')
        b3_rgb = rgb_conv3_3 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu3_3')
        b4_rgb = rgb_conv4_3 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu4_3')
        b5_rgb = rgb_conv5_3 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_relu5_3')
        b6_rgb = rgb_pool5 #list( dict(self.rgb_fcn.named_children()).keys()).index('fcn_pool5')

        b1_concat = b1_rgb
        b2_concat = b2_rgb
        b3_concat = b3_rgb
        b4_concat = b4_rgb
        b5_concat = b5_rgb
        b6_concat = b6_rgb

        
        if(dataset_type):
            #GPUs
            input_depth=input_depth.to(device).float()
            ### DEPTH ##########################################
            depth_conv1_1 = F.relu(self.depth_conv1_1(input_depth), inplace=True).to(device)
            depth_conv1_2 = F.relu(self.depth_conv1_2(depth_conv1_1), inplace=True).to(device)
            depth_pool1 = self.depth_pool_1(depth_conv1_2).to(device)
            
            depth_conv2_1 = F.relu(self.depth_conv2_1(depth_pool1), inplace=True).to(device)
            depth_conv2_2 = F.relu(self.depth_conv2_2(depth_conv2_1), inplace=True).to(device)
            depth_pool2 = self.depth_pool_2(depth_conv2_2).to(device)    
            
            depth_conv3_1 = F.relu(self.depth_conv3_1(depth_pool2), inplace=True).to(device)
            depth_conv3_2 = F.relu(self.depth_conv3_2(depth_conv3_1), inplace=True).to(device)
            depth_conv3_3 = F.relu(self.depth_conv3_3(depth_conv3_2), inplace=True).to(device)
            depth_pool3 = self.depth_pool_3(depth_conv3_3).to(device)     
            
            depth_conv4_1 = F.relu(self.depth_conv4_1(depth_pool3), inplace=True).to(device)
            depth_conv4_2 = F.relu(self.depth_conv4_2(depth_conv4_1), inplace=True).to(device)
            depth_conv4_3 = F.relu(self.depth_conv4_3(depth_conv4_2), inplace=True).to(device)
            depth_pool4 = self.depth_pool_4(depth_conv4_3).to(device) 
            
            depth_conv5_1 = F.relu(self.depth_conv5_1(depth_pool4), inplace=True).to(device)
            depth_conv5_2 = F.relu(self.depth_conv5_2(depth_conv5_1), inplace=True).to(device)
            depth_conv5_3 = F.relu(self.depth_conv5_3(depth_conv5_2), inplace=True).to(device)
            depth_pool5 = self.depth_pool_2(depth_conv5_3).to(device).to(device) 
    
            # depth branches
            b1_depth=depth_conv1_2 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu1_2')
            b2_depth=depth_conv2_2 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu2_2')
            b3_depth=depth_conv3_3 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu3_3')
            b4_depth=depth_conv4_3 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu4_3')
            b5_depth=depth_conv5_3 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_drelu5_3')
            b6_depth=depth_pool5 #list( dict(self.depth_fcn.named_children()).keys()).index('fcn_dpool5')

            ################################
            # short connection: concatenate
            # concat the axis = 1  dim=[0, "1", 2, 3]
            ################################
            b1_concat=torch.cat((b1_rgb,b1_depth),1).to(device)
            b2_concat=torch.cat((b2_rgb,b2_depth),1).to(device)
            b3_concat=torch.cat((b3_rgb,b3_depth),1).to(device)
            b4_concat=torch.cat((b4_rgb,b4_depth),1).to(device)
            b5_concat=torch.cat((b5_rgb,b5_depth),1).to(device)
            b6_concat=torch.cat((b6_rgb,b6_depth),1).to(device)
            

        ################################
        # short connection: to side activation
        ################################
        if(dataset_type):
            b1_conv1 = F.relu(self.b1_conv1(b1_concat), inplace=True).to(device)
            b2_conv1 = F.relu(self.b2_conv1(b2_concat), inplace=True).to(device)
            b3_conv1 = F.relu(self.b3_conv1(b3_concat), inplace=True).to(device)
            b4_conv1 = F.relu(self.b4_conv1(b4_concat), inplace=True).to(device)
            b5_conv1 = F.relu(self.b5_conv1(b5_concat), inplace=True).to(device)
            b6_conv1 = F.relu(self.b6_conv1(b6_concat), inplace=True).to(device)
        else:
            b1_conv1 = F.relu(self.b1_conv1_rgb(b1_concat), inplace=True).to(device)
            b2_conv1 = F.relu(self.b2_conv1_rgb(b2_concat), inplace=True).to(device)
            b3_conv1 = F.relu(self.b3_conv1_rgb(b3_concat), inplace=True).to(device)
            b4_conv1 = F.relu(self.b4_conv1_rgb(b4_concat), inplace=True).to(device)
            b5_conv1 = F.relu(self.b5_conv1_rgb(b5_concat), inplace=True).to(device)
            b6_conv1 = F.relu(self.b6_conv1_rgb(b6_concat), inplace=True).to(device)
        
        #branch 1
        b1_conv2 = F.relu(self.b1_conv2(b1_conv1), inplace=True).to(device)
        b1_conv3 = self.b1_conv3(b1_conv2).to(device)
        #branch 2
        b2_conv2 = F.relu(self.b2_conv2(b2_conv1), inplace=True).to(device)
        b2_conv3 = self.b2_conv3(b2_conv2).to(device)
        #branch 3
        b3_conv2 = F.relu(self.b3_conv2(b3_conv1), inplace=True).to(device)
        b3_conv3 = self.b3_conv3(b3_conv2).to(device)
        #branch 4
        b4_conv2 = F.relu(self.b4_conv2(b4_conv1), inplace=True).to(device)
        b4_conv3 = self.b4_conv3(b4_conv2).to(device)
        #branch 5
        b5_conv2 = F.relu(self.b5_conv2(b5_conv1), inplace=True).to(device)
        b5_conv3 = self.b5_conv3(b5_conv2).to(device)
        #branch 6
        b6_conv2 = F.relu(self.b6_conv2(b6_conv1), inplace=True).to(device)
        b6_conv3 = self.b6_conv3(b6_conv2).to(device)
            

        ################################
        # short connection: branch results
        ################################        
        # shortconnections to Aside 6
        r6_conv1 = b6_conv3
        # Rside 6
        
        # shortconnections to Aside 5
        r5_conv1 = b5_conv3
        # Rside 5
        
        # shortconnections to Aside 4
        # print(r6_conv1.shape)
        # print(r5_conv1.shape)
        
        r5_conv1_resize_r4 = F.interpolate(r5_conv1, [b4_conv3.shape[2]])
        r6_conv1_resize_r4 = F.interpolate(r6_conv1, [b4_conv3.shape[2]]).to(device)
        #print(r5_conv1_resize_r4.shape) #1, 1, 72, 72
        #print(r6_conv1_resize_r4.shape) #1, 1, 48, 48
        #print(b4_conv3.shape) #1, 1, 76, 76
        r4_concat = torch.cat((b4_conv3, r5_conv1_resize_r4, r6_conv1_resize_r4),1).to(device)
        r4_conv1 = self.b4_sc(r4_concat).to(device)
        # Rside 4
        # shortconnections to Aside 3
        r5_conv1_resize_r3 = F.interpolate(r5_conv1, [b3_conv3.shape[2]]).to(device)
        r6_conv1_resize_r3 = F.interpolate(r6_conv1, [b3_conv3.shape[2]]).to(device)
        r3_concat = torch.cat((b3_conv3, r5_conv1_resize_r3, r6_conv1_resize_r3),1).to(device)
        r3_conv1 = self.b3_sc(r3_concat).to(device)
        # Rside 3
        # short connections to Aside 2
        r3_conv1_resize_r2 = F.interpolate(r3_conv1, [b2_conv3.shape[2]]).to(device)
        r4_conv1_resize_r2 = F.interpolate(r4_conv1, [b2_conv3.shape[2]]).to(device)
        r5_conv1_resize_r2 = F.interpolate(r5_conv1, [b2_conv3.shape[2]]).to(device)
        r6_conv1_resize_r2 = F.interpolate(r6_conv1, [b2_conv3.shape[2]]).to(device)
        r2_concat = torch.cat((b2_conv3, r3_conv1_resize_r2, r4_conv1_resize_r2, r5_conv1_resize_r2, r6_conv1_resize_r2),1).to(device)
        r2_conv1 = self.b2_sc(r2_concat).to(device)
        # Rside 3
        # short connections to Aside 1
        r3_conv1_resize_r1 = F.interpolate(r3_conv1, [b1_conv3.shape[2]]).to(device)
        r4_conv1_resize_r1 = F.interpolate(r4_conv1, [b1_conv3.shape[2]]).to(device)
        r5_conv1_resize_r1 = F.interpolate(r5_conv1, [b1_conv3.shape[2]]).to(device)
        r6_conv1_resize_r1 = F.interpolate(r6_conv1, [b1_conv3.shape[2]]).to(device)
        r1_concat = torch.cat((b1_conv3, r3_conv1_resize_r1, r4_conv1_resize_r1, r5_conv1_resize_r1, r6_conv1_resize_r1),1).to(device)
        r1_conv1 = self.b1_sc(r1_concat).to(device)
        # Rside 1       
        ################################
        # short connection: FUSION
        ################################        
        # img_size=[ b1_conv3.shape[3]*b1_conv3.shape[2] ]
        # fuse used branch results
        r1_conv1_resize = r1_conv1 
        r2_conv1_resize = F.interpolate(r2_conv1, [b1_conv3.shape[2]]).to(device)
        r3_conv1_resize = F.interpolate(r3_conv1, [b1_conv3.shape[2]]).to(device)
        r4_conv1_resize = F.interpolate(r4_conv1, [b1_conv3.shape[2]]).to(device)
        r5_conv1_resize = F.interpolate(r5_conv1, [b1_conv3.shape[2]]).to(device)
        r6_conv1_resize = F.interpolate(r6_conv1, [b1_conv3.shape[2]]).to(device)
        
        #fusion_concat=torch.cat((r2_conv1_resize,r3_conv1_resize,r4_conv1_resize],1).to(device)
        fusion_concat=torch.cat((r1_conv1_resize,r2_conv1_resize,r3_conv1_resize,r4_conv1_resize,r5_conv1_resize,r6_conv1_resize),1).to(device)
        fusion_result=self.fusion_conv(fusion_concat).to(device)
        

        return fusion_result, b1_conv3, b2_conv3, b3_conv3, b4_conv3, b5_conv3, b6_conv3


# ===============================================================================
# ===============================================================================
# ===============================================================================
# =============================================================================== 
#cfg
# vgg choice
base = {'dss': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}
# extend vgg choice --- follow the paper, you can change it
extra = {'dss': [(64, 128, 3, [8, 16, 32, 64]), (128, 128, 3, [4, 8, 16, 32]), (256, 256, 5, [8, 16]),
                 (512, 256, 5, [4, 8]), (512, 512, 5, []), (512, 512, 7, [])]}

extra_p = {'dss': [(128, 128, 3, [8, 16, 32, 64]), (256, 128, 3, [4, 8, 16, 32]), (512, 256, 5, [8, 16]),
                 (1024, 256, 5, [4, 8]), (1024, 512, 5, []), (1024, 512, 7, [])]}
# parallel
base_rgb = {'dss': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}
base_depth = {'dss': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}
#short connection
connect = {'dss': [[2, 3, 4, 5], [2, 3, 4, 5], [4, 5], [4, 5], [], []]}

'''
functions
'''
# vgg16
def vgg(cfg, i=3, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers

# feature map before sigmoid: build the connection and deconvolution
class ConcatLayer(nn.Module):
    def __init__(self, list_k, k, scale=True):
        super(ConcatLayer, self).__init__()
        l, up, self.scale = len(list_k), [], scale
        # print(list_k)
        for i in range(l):
            up.append(nn.ConvTranspose2d(1, 1, list_k[i], list_k[i] // 2, list_k[i] // 4))
        self.upconv = nn.ModuleList(up)
        self.conv = nn.Conv2d(l + 1, 1, 1, 1)
        self.deconv = nn.ConvTranspose2d(1, 1, k * 2, k, k // 2) if scale else None

    def forward(self, x, list_x):
        elem_x = [x]
        for i, elem in enumerate(list_x):
            elem_x.append(self.upconv[i](elem))
        if self.scale:
            out = self.deconv(self.conv(torch.cat(elem_x, dim=1)))
        else:
            out = self.conv(torch.cat(elem_x, dim=1))
        return out

# extend vgg: side outputs
class FeatLayer(nn.Module):
    def __init__(self, in_channel, channel, k):
        super(FeatLayer, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(in_channel, channel, k, 1, k // 2), nn.ReLU(inplace=True),
                                  nn.Conv2d(channel, channel, k, 1, k // 2), nn.ReLU(inplace=True),
                                  nn.Conv2d(channel, 1, 1, 1))
    def forward(self, x):
        return self.main(x)

# fusion features
class FusionLayer(nn.Module):
    def __init__(self, nums=6):
        super(FusionLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(nums))
        self.nums = nums
        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.weights, 1 / self.nums)

    def forward(self, x):
        for i in range(self.nums):
            out = self.weights[i] * x[i] if i == 0 else out + self.weights[i] * x[i]
        return out


# extra part
def extra_layer(vgg, cfg):
    feat_layers, concat_layers, scale = [], [], 1
    for k, v in enumerate(cfg):
        # side output (paper: figure 3)
        feat_layers += [FeatLayer(v[0], v[1], v[2])]
        # feature map before sigmoid
        concat_layers += [ConcatLayer(v[3], scale, k != 0)]
        scale *= 2
    return vgg, feat_layers, concat_layers

def extra_2layers(cfg):
    feat_layers, concat_layers, scale = [], [], 1
    for k, v in enumerate(cfg):
        # side output (paper: figure 3)
        feat_layers += [FeatLayer(v[0], v[1], v[2])]
        # feature map before sigmoid
        concat_layers += [ConcatLayer(v[3], scale, k != 0)]
        scale *= 2
    return feat_layers, concat_layers

def extra_featlayer(cfg):
    feat_layers, scale = [], 1
    for k, v in enumerate(cfg):
        # side output (paper: figure 3)
        feat_layers += [FeatLayer(v[0], v[1], v[2])]
    return feat_layers, concat_layers

def extra_concatlayer(cfg):
    concat_layers, scale = [], 1
    for k, v in enumerate(cfg):
        # feature map before sigmoid
        concat_layers += [ConcatLayer(v[3], scale, k != 0)]
        scale *= 2
    return feat_layers, concat_layers
    
'''
networks
'''
# ===============================================================================
# =============================================================================== 
# DSS network
# Note: if you use other backbone network, please change extract
class DSS_parallel(nn.Module):
    # vgg(base_rgb['dss'], 3), vgg(base_depth['dss'], 1), *extra_2layers(extra_p['dss']), connect['dss'] 
    def __init__(self,device=torch.device("cpu"), extract=[3, 8, 15, 22, 29], v2=True):
        super(DSS_parallel, self).__init__()
        self.device=device
        self.extract = extract
        self.connect = connect['dss']      
        feat_layers, concat_layers=extra_2layers(extra_p['dss'])
        self.base_rgb = nn.ModuleList(vgg(base_rgb['dss'], 3))
        self.base_depth = nn.ModuleList(vgg(base_depth['dss'], 1))
        
        self.feat= nn.ModuleList(feat_layers)
        self.comb = nn.ModuleList(concat_layers)
        self.pool = nn.MaxPool2d(2, 2)  #nn.AvgPool2d(3, 1, 1)
        self.v2 = v2
        if v2: self.fuse = FusionLayer()

    def forward(self, intput_rgb, input_depth, label=None):
        device=self.device
        prob, back, y, num = list(), list(), list(), 0
        x_rgb=intput_rgb.to(device)
        x_depth=input_depth.to(device)
        for k in range(len(self.base_rgb)):
            #x = self.base[k](x)
            x_rgb = self.base_rgb[k](x_rgb)
            x_depth = self.base_depth[k](x_depth)
            if k in self.extract:
              x=torch.cat((x_rgb,x_depth), dim=1)
              y.append(self.feat[num](x))  # b1 - b5
              num += 1
        # side output
        y.append(self.feat[num](self.pool(x))) #b6
        # print(y)
        
        for i, k in enumerate(range(len(y))):
            back.append(self.comb[i](y[i], [y[j] for j in self.connect[i]]))
        
        # fusion map
        if self.v2:
            # version2: learning fusion
            back.append(self.fuse(back))
        else:
            # version1: mean fusion
            back.append(torch.cat(back, dim=1).mean(dim=1, keepdim=True))
        # add sigmoid
        for i in back: prob.append(torch.sigmoid(i))

        
        return prob[6], prob[0], prob[1], prob[2], prob[3], prob[4], prob[5]

# weight init
def xavier(param):
    init.xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)   



# ===============================================================================
# =============================================================================== 

'''
# construction test
if __name__ == '__main__':
    #DSS parallel
    dssnet = DSS_parallel()#DSS_parallel( vgg(base_rgb['dss'], 3), vgg(base_depth['dss'], 1), *extra_2layers(extra_p['dss']), connect['dss'] ) 
    seq=torch.nn.Sequential(*list(dssnet.children())[:])
    print(seq)
'''
