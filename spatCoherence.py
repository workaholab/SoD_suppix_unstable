#################################
# defined Loss function
# #################################
# https://discuss.pytorch.org/t/equivalent-of-tensorflows-sigmoid-cross-entropy-with-logits-in-pytorch/1985/7
# https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelSoftMarginLoss.html
from __future__ import print_function
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from visualization import visual_tensor

class Spco_Loss(nn.Module):
    def __init__(self, device, weight=[1.0, 1.0, 0.5]):
        # --------------------------------------------
        # Initialization
        # --------------------------------------------
        super(Spco_Loss, self).__init__()
        self.weight = weight
        self.cross_entropy_with_logits = nn.BCEWithLogitsLoss()
        self.device=device
        
        #weighting param
        self.sumRGBf=9600000
        self.sumDf=4800000
        
        #inputs => get from forward
        # fuse_result, b1_result, b2_result, b3_result, b4_result, b5_result, b6_result , gt

    def forward(self, epoch, fuse_result, b1_result, b2_result, b3_result, b4_result, b5_result, b6_result, gts, rgbs, depths):
        # --------------------------------------------
        # Define forward pass
        # --------------------------------------------
        # 

        # fusion
        fuse_losses = self.model_fusion_loss(fuse_result,gts)
        # branches
        # loss_branch = self.model_fusion_loss(b1_result, b2_result, b3_result, b4_result, b5_result, b6_result, gt)
        loss_branch_b1 = self.model_side_loss(b1_result, gts)
        loss_branch_b2 = self.model_side_loss(b2_result, gts)
        loss_branch_b3 = self.model_side_loss(b3_result, gts)
        #print(b4_result.shape)
        #print(gts.shape)
        # print(torch.max(gts),"/",torch.min(gts))
        loss_branch_b4 = self.model_side_loss(b4_result, gts)
        loss_branch_b5 = self.model_side_loss(b5_result, gts)
        loss_branch_b6 = self.model_side_loss(b6_result, gts)
        side_losses=loss_branch_b1+loss_branch_b2+loss_branch_b3+loss_branch_b4+loss_branch_b5+loss_branch_b6
        
        # spatial coherence
        if(len(rgbs.shape)==len(depths.shape) and epoch>30):
          loss_sp = self.model_neighbor_loss(fuse_result.float(), rgbs.float(), depths.float(), gts.float())
        else:
          loss_sp=0
        
        final_loss=fuse_losses+side_losses+loss_sp
        

        return final_loss
        
    def model_side_loss(self, b_side_out, ground_truth):
        # input data resize to original image size
        ground_truth=torch.squeeze(ground_truth).to(self.device)
        image_size=ground_truth.shape
        # print("ground_truth",torch.max(ground_truth))
        # print("image_size",image_size)
        
        b_side_out = F.interpolate(b_side_out,image_size).to(self.device)
        # sigmoid_b_side_out=torch.sigmoid(b_side_out).to(self.device) # already been sigmoid
        sigmoid_b_side_out=torch.squeeze(b_side_out).to(self.device) 
        
        #ground_truth.requires_grad=True
        losses =self.cross_entropy_with_logits(sigmoid_b_side_out, ground_truth).to(self.device) 
        # self.cross_entropy_with_logits(sigmoid_b_side_out,ground_truth)
        losses=torch.mean(losses)
        # print("model_side_loss", losses)
        
        return losses
        
    # RuntimeError: CUDA error: device-side assert triggered.   => F.bce generate the error        
    def model_fusion_loss(self, b_fuse_out, ground_truth):
        # input data resize to original image size
        ground_truth=torch.squeeze(ground_truth).to(self.device)
        image_size=ground_truth.shape
        
        b_fuse_out = F.interpolate(b_fuse_out,image_size).to(self.device)
        # sigmoid_b_fuse_out=torch.sigmoid(b_fuse_out).to(self.device) # already been sigmoid
        
        sigmoid_b_fuse_out=torch.squeeze(b_fuse_out).to(self.device)
        #ground_truth.requires_grad=True
        '''
        print("ground_truth.shape",ground_truth.shape)
        print("sigmoid_b_fuse_out.shape",sigmoid_b_fuse_out.shape)
        print(torch.max(sigmoid_b_fuse_out), torch.min(sigmoid_b_fuse_out))
        print(torch.max(ground_truth), torch.min(ground_truth))
        '''
        losses = self.cross_entropy_with_logits(sigmoid_b_fuse_out, ground_truth).to(self.device)  #self.cross_entropy_with_logits(sigmoid_b_fuse_out,ground_truth) #sigmoid_cross_entropy_with_logits
        losses=torch.mean(losses)
        # print("sigmoid_b_fuse_out", losses)
        
        return losses
 
    def model_neighbor_loss(self, b_fuse_out, rgbs, depths, gts):
        # pytorch tensor
        # (N, C, H, W) => Here is (B, C, W ,H) (Because of PIL trasnformation, confirmed)
        # tensorflow 
        # (N, H, W, C)
        
        # groundtruth
        ground_truth_flatten=torch.squeeze(gts,0).to(self.device)
        image_size=torch.squeeze(gts).shape 
        image_height=image_size[1]
        image_width=image_size[0]
        # image_size=[image_height,image_width]
        
        # make sure the resolution of all data is equal
        b_fuse_out = F.interpolate(b_fuse_out,image_size).to(self.device)
        rgbs = F.interpolate(rgbs,image_size).to(self.device)
        depths = F.interpolate(depths,image_size).to(self.device)
        # flatten
        b_fuse_out_flatten=torch.squeeze(b_fuse_out, 0) 
        rgbs_flatten = torch.cat((rgbs, depths), 1).to(self.device)
        prediction_flatten = b_fuse_out_flatten

        #size
        '''
        print("# image size: ",image_size)        
        print("# b_fuse_out: ",b_fuse_out.shape )
        print("# rgbs: ",rgbs.shape )
        print("# GTS: ",gts.shape )
        print("# depths: ",depths.shape )
        '''
                
        ###  => Here is (B, C, W ,H) (Because of PIL trasnformation, confirmed)
        #8 directs
        ## mono direct
        prediction_down = b_fuse_out_flatten[:,:,1:image_height].to(self.device)
        prediction_up = b_fuse_out_flatten[:,:,0:image_height-1].to(self.device)
        prediction_left = b_fuse_out_flatten[:,0:image_width-1,:].to(self.device)
        prediction_right = b_fuse_out_flatten[:,1:image_width,:].to(self.device)
        ## bi-direct
        prediction_down_left = b_fuse_out_flatten[:,0:image_width-1,1:image_height].to(self.device)
        prediction_down_right = b_fuse_out_flatten[:,1:image_width,1:image_height].to(self.device)
        prediction_up_left = b_fuse_out_flatten[:,0:image_width-1,0:image_height-1].to(self.device)
        prediction_up_right = b_fuse_out_flatten[:,1:image_width,0:image_height-1].to(self.device)
        
        # 1-count
        prediction_down_count = torch.ones(prediction_down.shape,device=self.device)
        prediction_up_count = torch.ones(prediction_up.shape,device=self.device)
        prediction_left_count = torch.ones(prediction_left.shape,device=self.device)
        prediction_right_count = torch.ones(prediction_right.shape,device=self.device)
        prediction_down_left_count = torch.ones(prediction_down_left.shape,device=self.device)
        prediction_down_right_count = torch.ones(prediction_down_right.shape,device=self.device)
        prediction_up_left_count = torch.ones(prediction_up_left.shape,device=self.device)
        prediction_up_right_count = torch.ones(prediction_up_right.shape,device=self.device)
        
        # zero count
        concat_row = torch.zeros([1, image_width, 1], device=self.device)
        concat_column = torch.zeros([1, 1, image_height], device=self.device)
        concat_column_minus = torch.zeros([1, 1, image_height-1], device=self.device)

        #padding the unfill space
        prediction_down = torch.cat([prediction_down, concat_row], 2).to(self.device)
        prediction_up = torch.cat([concat_row, prediction_up], 2).to(self.device)
        prediction_left = torch.cat([concat_column, prediction_left], 1).to(self.device)
        prediction_right = torch.cat([prediction_right, concat_column], 1).to(self.device)
        # print("## pass, phase 1 ##") 
        prediction_down_left = torch.cat([concat_column_minus, prediction_down_left], 1).to(self.device)
        prediction_down_left = torch.cat([prediction_down_left, concat_row], 2).to(self.device)
        prediction_down_right = torch.cat([prediction_down_right, concat_column_minus], 1).to(self.device)
        prediction_down_right = torch.cat([prediction_down_right, concat_row], 2).to(self.device)
        prediction_up_left = torch.cat([concat_column_minus, prediction_up_left], 1).to(self.device)
        prediction_up_left = torch.cat([concat_row, prediction_up_left], 2).to(self.device)
        prediction_up_right = torch.cat([prediction_up_right, concat_column_minus], 1).to(self.device)
        prediction_up_right = torch.cat([concat_row, prediction_up_right], 2).to(self.device)  
        # print("## pass, phase 2 ##") 
        # padding count 
        prediction_down_count = torch.cat([prediction_down_count, concat_row], 2).to(self.device)
        prediction_up_count = torch.cat([concat_row, prediction_up_count], 2).to(self.device)
        prediction_left_count = torch.cat([concat_column, prediction_left_count], 1).to(self.device)
        prediction_right_count = torch.cat([prediction_right_count, concat_column], 1).to(self.device)
        # print("## pass, phase 3 ##") 
        #print(concat_column_minus.shape)
        #print(prediction_down_left.shape)
        prediction_down_left_count = torch.cat([concat_column_minus, prediction_down_left_count], 1).to(self.device)
        prediction_down_left_count = torch.cat([prediction_down_left_count, concat_row], 2).to(self.device)
        prediction_down_right_count = torch.cat([prediction_down_right_count, concat_column_minus], 1).to(self.device)
        prediction_down_right_count = torch.cat([prediction_down_right_count, concat_row], 2).to(self.device)
        prediction_up_left_count = torch.cat([concat_column_minus, prediction_up_left_count], 1).to(self.device)
        prediction_up_left_count = torch.cat([concat_row, prediction_up_left_count], 2).to(self.device)
        prediction_up_right_count = torch.cat([prediction_up_right_count, concat_column_minus], 1).to(self.device)
        prediction_up_right_count = torch.cat([concat_row, prediction_up_right_count], 2).to(self.device)  
        # print("## pass, phase 4 ##")
        
        # before 2018/4/21 & 2018/4/23
        logit_down = prediction_flatten * prediction_down_count
        logit_up = prediction_flatten * prediction_up_count
        logit_left = prediction_flatten * prediction_left_count
        logit_right = prediction_flatten * prediction_right_count
        logit_down_left = prediction_flatten * prediction_down_left_count
        logit_down_right = prediction_flatten * prediction_down_right_count
        logit_up_left = prediction_flatten * prediction_up_left_count
        logit_up_right = prediction_flatten * prediction_up_right_count
        
        prediction_count = prediction_down_count + prediction_up_count + prediction_left_count + prediction_right_count + prediction_down_left_count + prediction_down_right_count + prediction_up_left_count + prediction_up_right_count
        # preprocessing of RGBs
        # print("## pass, Logit ##")
        
        # 4-d
        # print("## RGB, not original size but 640x640 ##") 
        rgb_down = rgbs_flatten[:,:,:,1:image_height].to(self.device)
        rgb_up = rgbs_flatten[:,:,:,0:image_height-1].to(self.device)
        rgb_left = rgbs_flatten[:,:,0:image_width-1,:].to(self.device)
        rgb_right = rgbs_flatten[:,:,1:image_width,:].to(self.device)
        # other 4-d / 8-d
        rgb_down_left = rgbs_flatten[:,:,0:image_width-1,1:image_height].to(self.device)
        rgb_down_right = rgbs_flatten[:,:,1:image_width,1:image_height].to(self.device)
        rgb_up_left = rgbs_flatten[:,:,0:image_width-1,0:image_height-1].to(self.device)
        rgb_up_right = rgbs_flatten[:,:,1:image_width,0:image_height-1].to(self.device)
        # print("## pass, RGB phase 1 ##")
        
        rgb_down_count = torch.ones(rgb_down.shape, device=self.device)
        rgb_up_count = torch.ones(rgb_up.shape, device=self.device)
        rgb_left_count = torch.ones(rgb_left.shape, device=self.device)
        rgb_right_count = torch.ones(rgb_right.shape, device=self.device)
        
        rgb_down_left_count = torch.ones(rgb_down_left.shape, device=self.device)
        rgb_down_right_count = torch.ones(rgb_down_right.shape, device=self.device)
        rgb_up_left_count = torch.ones(rgb_up_left.shape, device=self.device)
        rgb_up_right_count = torch.ones(rgb_up_right.shape, device=self.device)
        
        concat_row_rgb = torch.zeros([1, 4, image_width, 1], device=self.device)
        concat_column_rgb = torch.zeros([1, 4, 1, image_height], device=self.device)
        concat_column_minus_rgb = torch.zeros([1, 4, 1, image_height-1], device=self.device)
        # print("## pass, RGB phase 2 ##")
        # print("concat_row_rgb: ", concat_row_rgb.shape)
        
        rgb_down = torch.cat([rgb_down, concat_row_rgb], 3).to(self.device)
        rgb_up = torch.cat([concat_row_rgb, rgb_up], 3).to(self.device)
        rgb_left = torch.cat([concat_column_rgb, rgb_left], 2).to(self.device)
        rgb_right = torch.cat([rgb_right, concat_column_rgb], 2).to(self.device)
        
        rgb_down_left = torch.cat([concat_column_minus_rgb, rgb_down_left], 2).to(self.device)
        rgb_down_left = torch.cat([rgb_down_left, concat_row_rgb], 3).to(self.device)
        rgb_down_right = torch.cat([rgb_down_right, concat_column_minus_rgb], 2).to(self.device)
        rgb_down_right = torch.cat([rgb_down_right, concat_row_rgb], 3).to(self.device)
        rgb_up_left = torch.cat([concat_column_minus_rgb, rgb_up_left], 2).to(self.device)
        rgb_up_left = torch.cat([concat_row_rgb, rgb_up_left], 3).to(self.device)
        rgb_up_right = torch.cat([rgb_up_right, concat_column_minus_rgb], 2).to(self.device)
        rgb_up_right = torch.cat([concat_row_rgb, rgb_up_right], 3).to(self.device)
        # print("## pass, RGB phase 3 ##")
        #count
        rgb_down_count = torch.cat([rgb_down_count, concat_row_rgb], 3).to(self.device)
        rgb_up_count = torch.cat([concat_row_rgb, rgb_up_count], 3).to(self.device)
        rgb_left_count = torch.cat([concat_column_rgb, rgb_left_count], 2).to(self.device)
        rgb_right_count = torch.cat([rgb_right_count, concat_column_rgb], 2).to(self.device)
        
        rgb_down_left_count = torch.cat([concat_column_minus_rgb, rgb_down_left_count], 2).to(self.device)
        rgb_down_left_count = torch.cat([rgb_down_left_count, concat_row_rgb], 3).to(self.device)
        rgb_down_right_count = torch.cat([rgb_down_right_count, concat_column_minus_rgb], 2).to(self.device)
        rgb_down_right_count = torch.cat([rgb_down_right_count, concat_row_rgb], 3).to(self.device)
        rgb_up_left_count = torch.cat([concat_column_minus_rgb, rgb_up_left_count], 2).to(self.device)
        rgb_up_left_count = torch.cat([concat_row_rgb, rgb_up_left_count], 3).to(self.device)
        rgb_up_right_count = torch.cat([rgb_up_right_count, concat_column_minus_rgb], 2).to(self.device)
        rgb_up_right_count = torch.cat([concat_row_rgb, rgb_up_right_count], 3).to(self.device)
        # print("## pass, RGB phase 4 ##")

        rgb_orig_down = rgbs_flatten * rgb_down_count
        rgb_orig_up = rgbs_flatten * rgb_up_count
        rgb_orig_left = rgbs_flatten * rgb_left_count
        rgb_orig_right = rgbs_flatten * rgb_right_count
        rgb_orig_down_left = rgbs_flatten * rgb_down_left_count
        rgb_orig_down_right = rgbs_flatten * rgb_down_right_count
        rgb_orig_up_left = rgbs_flatten * rgb_up_left_count
        rgb_orig_up_right = rgbs_flatten * rgb_up_right_count
        
        
        # 2018/8/1 added, processing of gts
        # print(ground_truth_flatten.shape)
        '''
        ## mono direct
        prediction_down = b_fuse_out_flatten[:,:,1:image_height].to(self.device)
        prediction_up = b_fuse_out_flatten[:,:,0:image_height-1].to(self.device)
        prediction_left = b_fuse_out_flatten[:,0:image_width-1,:].to(self.device)
        prediction_right = b_fuse_out_flatten[:,1:image_width,:].to(self.device)
        ## bi-direct
        prediction_down_left = b_fuse_out_flatten[:,0:image_width-1,1:image_height].to(self.device)
        prediction_down_right = b_fuse_out_flatten[:,1:image_width,1:image_height].to(self.device)
        prediction_up_left = b_fuse_out_flatten[:,0:image_width-1,0:image_height-1].to(self.device)
        prediction_up_right = b_fuse_out_flatten[:,1:image_width,0:image_height-1].to(self.device)
        '''
        gt_down = ground_truth_flatten[:,:,1:image_height].to(self.device)
        gt_up = ground_truth_flatten[:,:,0:image_height-1].to(self.device)
        gt_left = ground_truth_flatten[:,0:image_width-1,:].to(self.device)
        gt_right = ground_truth_flatten[:,1:image_width,:].to(self.device)
        gt_down_left = ground_truth_flatten[:,0:image_width-1,1:image_height].to(self.device)
        gt_down_right = ground_truth_flatten[:,1:image_width,1:image_height].to(self.device)
        gt_up_left = ground_truth_flatten[:,0:image_width-1,0:image_height-1].to(self.device)
        gt_up_right = ground_truth_flatten[:,1:image_width,0:image_height-1].to(self.device)
        
        '''
        prediction_down_count = torch.ones(prediction_down.shape,device=self.device)
        prediction_up_count = torch.ones(prediction_up.shape,device=self.device)
        prediction_left_count = torch.ones(prediction_left.shape,device=self.device)
        prediction_right_count = torch.ones(prediction_right.shape,device=self.device)
        prediction_down_left_count = torch.ones(prediction_down_left.shape,device=self.device)
        prediction_down_right_count = torch.ones(prediction_down_right.shape,device=self.device)
        prediction_up_left_count = torch.ones(prediction_up_left.shape,device=self.device)
        prediction_up_right_count = torch.ones(prediction_up_right.shape,device=self.device)
        '''
        gt_down_count = torch.ones(gt_down.shape,device=self.device)
        gt_up_count = torch.ones(gt_up.shape,device=self.device)
        gt_left_count = torch.ones(gt_left.shape,device=self.device)
        gt_right_count = torch.ones(gt_right.shape,device=self.device)
        gt_down_left_count = torch.ones(gt_down_left.shape,device=self.device)
        gt_down_right_count = torch.ones(gt_down_right.shape,device=self.device)
        gt_up_left_count = torch.ones(gt_up_left.shape,device=self.device)
        gt_up_right_count = torch.ones(gt_up_right.shape,device=self.device)
        # print("## GT phase 1 pass ##")
        # reuse prediction partss
        #concat_row = torch.zeros([1, 1, image_width], torch.float32)
        #concat_column = torch.zeros([1,image_height,1], torch.float32)
        #concat_column_minus = torch.zeros([1,image_height-1,1], torch.float32)
        
        '''
        #padding the unfill space
        prediction_down = torch.cat([prediction_down, concat_row], 2).to(self.device)
        prediction_up = torch.cat([concat_row, prediction_up], 2).to(self.device)
        prediction_left = torch.cat([concat_column, prediction_left], 1).to(self.device)
        prediction_right = torch.cat([prediction_right, concat_column], 1).to(self.device)
        # print("## pass, phase 1 ##") 
        prediction_down_left = torch.cat([concat_column_minus, prediction_down_left], 1).to(self.device)
        prediction_down_left = torch.cat([prediction_down_left, concat_row], 2).to(self.device)
        prediction_down_right = torch.cat([prediction_down_right, concat_column_minus], 1).to(self.device)
        prediction_down_right = torch.cat([prediction_down_right, concat_row], 2).to(self.device)
        prediction_up_left = torch.cat([concat_column_minus, prediction_up_left], 1).to(self.device)
        prediction_up_left = torch.cat([concat_row, prediction_up_left], 2).to(self.device)
        prediction_up_right = torch.cat([prediction_up_right, concat_column_minus], 1).to(self.device)
        prediction_up_right = torch.cat([concat_row, prediction_up_right], 2).to(self.device)  
        # print("## pass, phase 2 ##") 
        # padding count 
        prediction_down_count = torch.cat([prediction_down_count, concat_row], 2).to(self.device)
        prediction_up_count = torch.cat([concat_row, prediction_up_count], 2).to(self.device)
        prediction_left_count = torch.cat([concat_column, prediction_left_count], 1).to(self.device)
        prediction_right_count = torch.cat([prediction_right_count, concat_column], 1).to(self.device)
        # print("## pass, phase 3 ##") 
        #print(concat_column_minus.shape)
        #print(prediction_down_left.shape)
        prediction_down_left_count = torch.cat([concat_column_minus, prediction_down_left_count], 1).to(self.device)
        prediction_down_left_count = torch.cat([prediction_down_left_count, concat_row], 2).to(self.device)
        prediction_down_right_count = torch.cat([prediction_down_right_count, concat_column_minus], 1).to(self.device)
        prediction_down_right_count = torch.cat([prediction_down_right_count, concat_row], 2).to(self.device)
        prediction_up_left_count = torch.cat([concat_column_minus, prediction_up_left_count], 1).to(self.device)
        prediction_up_left_count = torch.cat([concat_row, prediction_up_left_count], 2).to(self.device)
        prediction_up_right_count = torch.cat([prediction_up_right_count, concat_column_minus], 1).to(self.device)
        prediction_up_right_count = torch.cat([concat_row, prediction_up_right_count], 2).to(self.device) 
        '''
        
        gt_down = torch.cat([gt_down, concat_row], 2)
        gt_up = torch.cat([concat_row,gt_up], 2)
        gt_left = torch.cat([concat_column, gt_left], 1)
        gt_right = torch.cat([gt_right, concat_column], 1)
        
        gt_down_left = torch.cat([concat_column_minus, gt_down_left], 1)
        gt_down_right = torch.cat([gt_down_right, concat_column_minus], 1)
        gt_up_left = torch.cat([concat_column_minus, gt_up_left], 1)
        gt_up_right = torch.cat([gt_up_right, concat_column_minus], 1)
        gt_down_left = torch.cat([gt_down_left, concat_row], 2)
        gt_down_right = torch.cat([gt_down_right, concat_row], 2)
        gt_up_left = torch.cat([concat_row, gt_up_left], 2)
        gt_up_right = torch.cat([concat_row, gt_up_right], 2)
        # print("## GT phase 2-1 pass ##")        
        gt_down_count = torch.cat([gt_down_count, concat_row], 2)
        gt_up_count = torch.cat([concat_row, gt_up_count], 2)
        gt_left_count = torch.cat([concat_column, gt_left_count], 1)
        gt_right_count = torch.cat([gt_right_count, concat_column], 1)
        gt_down_left_count = torch.cat([concat_column_minus, gt_down_left_count], 1)
        gt_down_right_count = torch.cat([gt_down_right_count, concat_column_minus], 1)
        gt_up_left_count = torch.cat([concat_column_minus, gt_up_left_count], 1)
        gt_up_right_count = torch.cat([gt_up_right_count, concat_column_minus], 1)
        gt_down_left_count = torch.cat([gt_down_left_count, concat_row], 2)
        gt_down_right_count = torch.cat([gt_down_right_count, concat_row], 2)
        gt_up_left_count = torch.cat([concat_row, gt_up_left_count], 2)
        gt_up_right_count = torch.cat([concat_row, gt_up_right_count], 2)
        # print("## GT phase 2-2 pass ##")
        
                
        gt_orig_down = ground_truth_flatten * gt_down_count
        gt_orig_up =  ground_truth_flatten * gt_up_count
        gt_orig_left =  ground_truth_flatten * gt_left_count
        gt_orig_right = ground_truth_flatten * gt_right_count
        gt_orig_down_left =  ground_truth_flatten * gt_down_left_count
        gt_orig_down_right = ground_truth_flatten * gt_down_right_count
        gt_orig_up_left = ground_truth_flatten * gt_up_left_count
        gt_orig_up_right =  ground_truth_flatten * gt_up_right_count
        # print("## GT phase 3 pass ##")
        
        # before 2018/4/21 & 2018/4/23
        '''losses_down = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_down, logits=logit_down)
        losses_up = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_up, logits=logit_up)
        losses_left = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_left, logits=logit_left)
        losses_right = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_right, logits=logit_right)
        losses_down_left = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_down_left, logits=logit_down_left)
        losses_down_right = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_down_right, logits=logit_down_right)
        losses_up_left = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_up_left, logits=logit_up_left)
        losses_up_right = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_up_right, logits=logit_up_right)
        
        losses = (losses_down+losses_up+losses_left+losses_right+losses_down_left+losses_down_right+losses_up_left+losses_up_right) / prediction_count'''
        
        # 2018/4/22
        '''prediction_mean = (prediction_down + prediction_up + prediction_left + prediction_right + prediction_down_left + prediction_down_right + prediction_up_left + prediction_up_right) / prediction_count
        losses = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_mean, logits=prediction_flatten)'''
        
        # 2018/4/23
        # 2018/7/18 updated
        # 2018/8/20 updated
        down_power = torch.pow(rgb_down-rgb_orig_down,2)
        up_power = torch.pow(rgb_up-rgb_orig_up,2)
        left_power = torch.pow(rgb_left-rgb_orig_left,2)
        right_power = torch.pow(rgb_right-rgb_orig_right,2)
        down_left_power = torch.pow(rgb_down_left-rgb_orig_down_left,2)
        down_right_power = torch.pow(rgb_down_right-rgb_orig_down_right,2)
        up_left_power = torch.pow(rgb_up_left-rgb_orig_up_left,2)
        up_right_power = torch.pow(rgb_up_right-rgb_orig_up_right,2)
        '''
        print("## power pass ##")
        
        print(down_power.shape)
        print("self.sumRGBf*(down_power[:,:,:,0]+down_power[:,:,:,1]+down_power[:,:,:,2]) + self.sumDf*down_power[:,:,:,3]")
        '''
        down_sum = self.sumRGBf*(down_power[:,0,:,:]+down_power[:,1,:,:]+down_power[:,2,:,:]) + self.sumDf*down_power[:,3,:,:]
        up_sum = self.sumRGBf*(up_power[:,0,:,:]+up_power[:,1,:,:]+up_power[:,2,:,:]) + self.sumDf*up_power[:,3,:,:]
        left_sum = self.sumRGBf*(left_power[:,0,:,:]+left_power[:,1,:,:]+left_power[:,2,:,:]) + self.sumDf*left_power[:,3,:,:]
        right_sum = self.sumRGBf*(right_power[:,0,:,:]+right_power[:,1,:,:]+right_power[:,2,:,:]) + self.sumDf*right_power[:,3,:,:]
        down_left_sum = self.sumRGBf*(down_left_power[:,0,:,:]+down_left_power[:,1,:,:]+down_left_power[:,2,:,:]) + self.sumDf*down_left_power[:,3,:,:]
        down_right_sum = self.sumRGBf*(down_right_power[:,0,:,:]+down_right_power[:,1,:,:]+down_right_power[:,2,:,:]) + self.sumDf*down_right_power[:,3,:,:]
        up_left_sum = self.sumRGBf*(up_left_power[:,0,:,:]+up_left_power[:,1,:,:]+up_left_power[:,2,:,:]) + self.sumDf*up_left_power[:,3,:,:]
        up_right_sum = self.sumRGBf*(up_right_power[:,0,:,:]+up_right_power[:,1,:,:]+up_right_power[:,2,:,:]) + self.sumDf*up_right_power[:,3,:,:]
        '''
        print(down_sum.shape)
        print("## sumRGBf pass ##")

        print(prediction_down.shape)
        print(logit_down.shape)
        print(gt_down.shape)
        print("torch.exp(-down_sum /(128) ) * torch.abs(prediction_down-logit_down) * torch.abs(-1+prediction_down-gt_down)")
        print(torch.exp(-down_sum /(128) ).shape)
        print(torch.abs(prediction_down-logit_down).shape)
        print(torch.abs(-1+prediction_down-gt_down).shape)
        '''
        
        losses_down = torch.exp(-down_sum /(128) ) * torch.abs(prediction_down-logit_down) * torch.abs(-1+prediction_down-gt_down)
        losses_up = torch.exp(-up_sum /(128))*torch.abs(prediction_up-logit_up)*torch.abs(-1+prediction_up-gt_up)
        losses_left = torch.exp(-left_sum /(128))*torch.abs(prediction_left-logit_left)*torch.abs(-1+prediction_left-gt_left)
        losses_right = torch.exp(-right_sum /(128))*torch.abs(prediction_right-logit_right)*torch.abs(-1+prediction_right-gt_right)
        losses_down_left = torch.exp(-down_left_sum /(128))*torch.abs(prediction_down_left-logit_down_left)*torch.abs(-1+prediction_down_left-gt_down_left)
        losses_down_right = torch.exp(-down_right_sum /(128))*torch.abs(prediction_down_right-logit_down_right)*torch.abs(-1+prediction_down_right-gt_down_right)
        losses_up_left = torch.exp(-up_left_sum /(128))*torch.abs(prediction_up_left-logit_up_left)*torch.abs(-1+prediction_up_left-gt_up_left)
        losses_up_right = torch.exp(-up_right_sum /(128))*torch.abs(prediction_up_right-logit_up_right)*torch.abs(-1+prediction_up_right-gt_up_right)
        # print("## loss =1 pass ##")
        losses = (losses_down+losses_up+losses_left+losses_right+losses_down_left+losses_down_right+losses_up_left+losses_up_right) / prediction_count
        
        losses = torch.mean(losses)
        
        # print("## loss =2 pass ##")
        # losses = torch.cast(losses, tf.float64)
        
        return losses
        
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
class SP_Spco_Loss(nn.Module):
    def __init__(self, device, weight=[1.0, 1.0, 0.5]):
        # --------------------------------------------
        # Initialization
        # --------------------------------------------
        super(SP_Spco_Loss, self).__init__()
        self.weight = weight
        self.cross_entropy_with_logits = nn.BCEWithLogitsLoss()
        self.device=device
        
        #weighting param
        self.sumRGBf=9600000
        self.sumDf=4800000
        
        #inputs => get from forward
        # fuse_result, b1_result, b2_result, b3_result, b4_result, b5_result, b6_result , gt

    def forward(self, epoch, fuse_result, b1_result, b2_result, b3_result, b4_result, b5_result, b6_result, gts, rgbs, depths,result2d_out):
        # --------------------------------------------
        # Define forward pass
        # --------------------------------------------
        # Transform targets to one-hot vector
        
        #####################################
        # example code of custom 
        #####################################
        
        # fusion
        fuse_losses = self.model_fusion_loss(fuse_result,gts)
        # branches
        # loss_branch = self.model_fusion_loss(b1_result, b2_result, b3_result, b4_result, b5_result, b6_result, gt)
        loss_branch_b1 = self.model_side_loss(b1_result, gts)
        loss_branch_b2 = self.model_side_loss(b2_result, gts)
        loss_branch_b3 = self.model_side_loss(b3_result, gts)
        #print(b4_result.shape)
        #print(gts.shape)
        # print(torch.max(gts),"/",torch.min(gts))
        loss_branch_b4 = self.model_side_loss(b4_result, gts)
        loss_branch_b5 = self.model_side_loss(b5_result, gts)
        loss_branch_b6 = self.model_side_loss(b6_result, gts)
        side_losses=loss_branch_b1+loss_branch_b2+loss_branch_b3+loss_branch_b4+loss_branch_b5+loss_branch_b6
        
        # spatial coherence
        '''
        if(len(rgbs.shape)==len(depths.shape) and epoch>30):
          loss_sp = self.model_neighbor_loss(fuse_result.float(), rgbs.float(), depths.float(), gts.float()) # result2d_out / fuse_result.float()
        else:
          loss_sp=0
        '''
        
        final_loss=fuse_losses+side_losses # + loss_sp
        

        return final_loss
        
    def model_side_loss(self, b_side_out, ground_truth):
        """
        # back to original size
        b_side_out = torch.image.resize_images(b_side_out, [torch.shape(ground_truth)[1], torch.shape(ground_truth)[2]])
    
        prediction_flatten = torch.squeeze(b_side_out, axis=(3,))
        # first, cast and normalize the ground truth
        prediction_flatten = torch.cast(prediction_flatten, torch.float64, name='predict_flat_cast')
        #print (prediction_flatten.shape)
        ground_truth_flatten = torch.cast(ground_truth / 255.0, torch.float64, name='gt_flat_cast')
        ground_truth_flatten = torch.squeeze(ground_truth_flatten, axis=(3,))
        # then, calculate the cross entropy between the gt and the fused saliency map
        losses = torch.nn.sigmoid_cross_entropy_with_logits(labels=ground_truth_flatten, logits=prediction_flatten)
        '''
        # assert not torch.math.is_nan(losses), 'model_side_loss with losses = NaN at sigmoid cross entropy function'    
        torch.print(torch.math.is_nan(losses))
        assert_op = torch.Assert(torch.math.is_nan(losses), [losses])
        with torch.control_dependencies([assert_op]):
            print('model_side_loss with losses = NaN at sigmoid cross entropy function')
        '''
    
        losses = torch.reduce_mean(losses, axis=(1, 2))
        
        print(torch.math.is_nan(losses))
        print('model_fusion_loss with losses  = NaN at sigmoid cross entropy function')

            
        return losses
        """

        ground_truth=torch.squeeze(ground_truth).to(self.device)
        image_size=ground_truth.shape
        # print("ground_truth",torch.max(ground_truth))
        # print("image_size",image_size)
        
        b_side_out = F.interpolate(b_side_out,image_size).to(self.device)
        # sigmoid_b_side_out=torch.sigmoid(b_side_out).to(self.device) # already been sigmoid
        
        sigmoid_b_side_out=torch.squeeze(b_side_out).to(self.device) 
        
        #ground_truth.requires_grad=True
        
        losses = torch.sum(torch.abs(sigmoid_b_side_out-ground_truth))
        #self.cross_entropy_with_logits(sigmoid_b_side_out, ground_truth).to(self.device) # self.cross_entropy_with_logits(sigmoid_b_side_out,ground_truth)
        losses=torch.mean(losses)
        
        # print("model_side_loss", losses)
        
        return losses
        
    # RuntimeError: CUDA error: device-side assert triggered.   => F.bce generate the error        
    def model_fusion_loss(self, b_fuse_out, ground_truth):
        """
        # back to original size
        b_fuse_out = torch.image.resize_images(b_fuse_out, [torch.shape(ground_truth)[1], torch.shape(ground_truth)[2]])
    
        prediction_flatten = torch.squeeze(b_fuse_out, axis=(3,))
        # first, cast and normalize the ground truth
        prediction_flatten = torch.cast(prediction_flatten, torch.float64, name='predict_flat_cast')
        ground_truth_flatten = torch.cast(ground_truth / 255.0, torch.float64, name='gt_flat_cast')
        ground_truth_flatten = torch.squeeze(ground_truth_flatten, axis=(3,))
        # then, calculate the cross entropy between the gt and the fused saliency map
        losses = torch.nn.sigmoid_cross_entropy_with_logits(labels=ground_truth_flatten, logits=prediction_flatten)
        
        # assert not torch.math.is_nan(losses), 'model_fusion_loss with losses  = NaN at sigmoid cross entropy function'
        '''
        assert_op = torch.Assert(torch.math.is_nan(losses), [losses])
        with torch.control_dependencies([assert_op]):
            print(torch.math.is_nan(losses))
            print('model_fusion_loss with losses  = NaN at sigmoid cross entropy function')
        '''
    
        losses = torch.reduce_mean(losses, axis=(1, 2))
        
        print(torch.math.is_nan(losses))
        print('model_fusion_loss with losses  = NaN at sigmoid cross entropy function')
    
        return losses
        """
        ground_truth=torch.squeeze(ground_truth).to(self.device)
        image_size=ground_truth.shape
        
        b_fuse_out = F.interpolate(b_fuse_out,image_size).to(self.device)
        # sigmoid_b_fuse_out=torch.sigmoid(b_fuse_out).to(self.device) # already been sigmoid
        
        sigmoid_b_fuse_out=torch.squeeze(b_fuse_out).to(self.device)

        #ground_truth.requires_grad=True
        
        '''
        print("ground_truth.shape",ground_truth.shape)
        print("sigmoid_b_fuse_out.shape",sigmoid_b_fuse_out.shape)
        print(torch.max(sigmoid_b_fuse_out), torch.min(sigmoid_b_fuse_out))
        print(torch.max(ground_truth), torch.min(ground_truth))
        '''
        losses = torch.sum(torch.abs(sigmoid_b_fuse_out-ground_truth))
        #self.cross_entropy_with_logits(sigmoid_b_fuse_out, ground_truth).to(self.device)  #self.cross_entropy_with_logits(sigmoid_b_fuse_out,ground_truth) #sigmoid_cross_entropy_with_logits
        losses=torch.mean(losses)
        
        # print("sigmoid_b_fuse_out", losses)
        
        return losses
 
    def model_neighbor_loss(self, b_fuse_out, rgbs, depths, gts):
        # groundtruth
        ground_truth_flatten=torch.squeeze(gts,0).to(self.device)
        image_size=torch.squeeze(gts).shape 
        #size
        image_height=image_size[1]
        image_width=image_size[0]
        image_size=[image_width,image_height]
        '''        
        print("## ground_truth ##") 
        print(ground_truth.shape)
        print("## image_size ##") 
        print(image_size)
        '''
        b_fuse_out = F.interpolate(b_fuse_out,image_size).to(self.device)
        b_fuse_out_flatten=torch.squeeze(b_fuse_out, 0) 
        '''
        # norm and depth # !! norm !!
        print("## b_fuse_out_flatten ##") 
        print(b_fuse_out_flatten.shape) 
        
        print("## rgbs/depth ##") 
        print(rgbs.shape)  
        print(depths.shape)
        '''
        rgbs_flatten = torch.cat((rgbs, depths), 1).to(self.device)
        prediction_flatten = b_fuse_out_flatten
        '''
        print("## prediction_flatten ##") 
        print(prediction_flatten.shape)  
        '''
        
        ###  => Here is (B, C, W ,H) (Because of PIL trasnformation, confirmed)
        #8 directs
        ## mono direct
        prediction_down = b_fuse_out_flatten[:,:,1:image_height].to(self.device)
        prediction_up = b_fuse_out_flatten[:,:,0:image_height-1].to(self.device)
        prediction_left = b_fuse_out_flatten[:,0:image_width-1,:].to(self.device)
        prediction_right = b_fuse_out_flatten[:,1:image_width,:].to(self.device)
        ## bi-direct
        prediction_down_left = b_fuse_out_flatten[:,0:image_width-1,1:image_height].to(self.device)
        prediction_down_right = b_fuse_out_flatten[:,1:image_width,1:image_height].to(self.device)
        prediction_up_left = b_fuse_out_flatten[:,0:image_width-1,0:image_height-1].to(self.device)
        prediction_up_right = b_fuse_out_flatten[:,1:image_width,0:image_height-1].to(self.device)
        
        # 1-count
        prediction_down_count = torch.ones(prediction_down.shape,device=self.device)
        prediction_up_count = torch.ones(prediction_up.shape,device=self.device)
        prediction_left_count = torch.ones(prediction_left.shape,device=self.device)
        prediction_right_count = torch.ones(prediction_right.shape,device=self.device)
        prediction_down_left_count = torch.ones(prediction_down_left.shape,device=self.device)
        prediction_down_right_count = torch.ones(prediction_down_right.shape,device=self.device)
        prediction_up_left_count = torch.ones(prediction_up_left.shape,device=self.device)
        prediction_up_right_count = torch.ones(prediction_up_right.shape,device=self.device)
        
        # zero count
        concat_row = torch.zeros([1, image_width, 1], device=self.device)
        concat_column = torch.zeros([1, 1, image_height], device=self.device)
        concat_column_minus = torch.zeros([1, 1, image_height-1], device=self.device)

        #padding the unfill space
        prediction_down = torch.cat([prediction_down, concat_row], 2).to(self.device)
        prediction_up = torch.cat([concat_row, prediction_up], 2).to(self.device)
        prediction_left = torch.cat([concat_column, prediction_left], 1).to(self.device)
        prediction_right = torch.cat([prediction_right, concat_column], 1).to(self.device)
        # print("## pass, phase 1 ##") 
        prediction_down_left = torch.cat([concat_column_minus, prediction_down_left], 1).to(self.device)
        prediction_down_left = torch.cat([prediction_down_left, concat_row], 2).to(self.device)
        prediction_down_right = torch.cat([prediction_down_right, concat_column_minus], 1).to(self.device)
        prediction_down_right = torch.cat([prediction_down_right, concat_row], 2).to(self.device)
        prediction_up_left = torch.cat([concat_column_minus, prediction_up_left], 1).to(self.device)
        prediction_up_left = torch.cat([concat_row, prediction_up_left], 2).to(self.device)
        prediction_up_right = torch.cat([prediction_up_right, concat_column_minus], 1).to(self.device)
        prediction_up_right = torch.cat([concat_row, prediction_up_right], 2).to(self.device)  
        # print("## pass, phase 2 ##") 
        # padding count 
        prediction_down_count = torch.cat([prediction_down_count, concat_row], 2).to(self.device)
        prediction_up_count = torch.cat([concat_row, prediction_up_count], 2).to(self.device)
        prediction_left_count = torch.cat([concat_column, prediction_left_count], 1).to(self.device)
        prediction_right_count = torch.cat([prediction_right_count, concat_column], 1).to(self.device)
        # print("## pass, phase 3 ##") 
        #print(concat_column_minus.shape)
        #print(prediction_down_left.shape)
        prediction_down_left_count = torch.cat([concat_column_minus, prediction_down_left_count], 1).to(self.device)
        prediction_down_left_count = torch.cat([prediction_down_left_count, concat_row], 2).to(self.device)
        prediction_down_right_count = torch.cat([prediction_down_right_count, concat_column_minus], 1).to(self.device)
        prediction_down_right_count = torch.cat([prediction_down_right_count, concat_row], 2).to(self.device)
        prediction_up_left_count = torch.cat([concat_column_minus, prediction_up_left_count], 1).to(self.device)
        prediction_up_left_count = torch.cat([concat_row, prediction_up_left_count], 2).to(self.device)
        prediction_up_right_count = torch.cat([prediction_up_right_count, concat_column_minus], 1).to(self.device)
        prediction_up_right_count = torch.cat([concat_row, prediction_up_right_count], 2).to(self.device)  
        # print("## pass, phase 4 ##")
        
        # before 2018/4/21 & 2018/4/23
        logit_down = prediction_flatten * prediction_down_count
        logit_up = prediction_flatten * prediction_up_count
        logit_left = prediction_flatten * prediction_left_count
        logit_right = prediction_flatten * prediction_right_count
        logit_down_left = prediction_flatten * prediction_down_left_count
        logit_down_right = prediction_flatten * prediction_down_right_count
        logit_up_left = prediction_flatten * prediction_up_left_count
        logit_up_right = prediction_flatten * prediction_up_right_count
        
        prediction_count = prediction_down_count + prediction_up_count + prediction_left_count + prediction_right_count + prediction_down_left_count + prediction_down_right_count + prediction_up_left_count + prediction_up_right_count
        # preprocessing of RGBs
        # print("## pass, Logit ##")
        
        # 4-d
        # print("## RGB, not original size but 640x640 ##") 
        rgb_down = rgbs_flatten[:,:,:,1:image_height].to(self.device)
        rgb_up = rgbs_flatten[:,:,:,0:image_height-1].to(self.device)
        rgb_left = rgbs_flatten[:,:,0:image_width-1,:].to(self.device)
        rgb_right = rgbs_flatten[:,:,1:image_width,:].to(self.device)
        # other 4-d / 8-d
        rgb_down_left = rgbs_flatten[:,:,0:image_width-1,1:image_height].to(self.device)
        rgb_down_right = rgbs_flatten[:,:,1:image_width,1:image_height].to(self.device)
        rgb_up_left = rgbs_flatten[:,:,0:image_width-1,0:image_height-1].to(self.device)
        rgb_up_right = rgbs_flatten[:,:,1:image_width,0:image_height-1].to(self.device)
        # print("## pass, RGB phase 1 ##")
        
        rgb_down_count = torch.ones(rgb_down.shape, device=self.device)
        rgb_up_count = torch.ones(rgb_up.shape, device=self.device)
        rgb_left_count = torch.ones(rgb_left.shape, device=self.device)
        rgb_right_count = torch.ones(rgb_right.shape, device=self.device)
        
        rgb_down_left_count = torch.ones(rgb_down_left.shape, device=self.device)
        rgb_down_right_count = torch.ones(rgb_down_right.shape, device=self.device)
        rgb_up_left_count = torch.ones(rgb_up_left.shape, device=self.device)
        rgb_up_right_count = torch.ones(rgb_up_right.shape, device=self.device)
        
        concat_row_rgb = torch.zeros([1, 4, image_width, 1], device=self.device)
        concat_column_rgb = torch.zeros([1, 4, 1, image_height], device=self.device)
        concat_column_minus_rgb = torch.zeros([1, 4, 1, image_height-1], device=self.device)
        # print("## pass, RGB phase 2 ##")
        
        rgb_down = torch.cat([rgb_down, concat_row_rgb], 3).to(self.device)
        rgb_up = torch.cat([concat_row_rgb, rgb_up], 3).to(self.device)
        rgb_left = torch.cat([concat_column_rgb, rgb_left], 2).to(self.device)
        rgb_right = torch.cat([rgb_right, concat_column_rgb], 2).to(self.device)
        
        rgb_down_left = torch.cat([concat_column_minus_rgb, rgb_down_left], 2).to(self.device)
        rgb_down_left = torch.cat([rgb_down_left, concat_row_rgb], 3).to(self.device)
        rgb_down_right = torch.cat([rgb_down_right, concat_column_minus_rgb], 2).to(self.device)
        rgb_down_right = torch.cat([rgb_down_right, concat_row_rgb], 3).to(self.device)
        rgb_up_left = torch.cat([concat_column_minus_rgb, rgb_up_left], 2).to(self.device)
        rgb_up_left = torch.cat([concat_row_rgb, rgb_up_left], 3).to(self.device)
        rgb_up_right = torch.cat([rgb_up_right, concat_column_minus_rgb], 2).to(self.device)
        rgb_up_right = torch.cat([concat_row_rgb, rgb_up_right], 3).to(self.device)
        # print("## pass, RGB phase 3 ##")
        #count
        rgb_down_count = torch.cat([rgb_down_count, concat_row_rgb], 3).to(self.device)
        rgb_up_count = torch.cat([concat_row_rgb, rgb_up_count], 3).to(self.device)
        rgb_left_count = torch.cat([concat_column_rgb, rgb_left_count], 2).to(self.device)
        rgb_right_count = torch.cat([rgb_right_count, concat_column_rgb], 2).to(self.device)
        
        rgb_down_left_count = torch.cat([concat_column_minus_rgb, rgb_down_left_count], 2).to(self.device)
        rgb_down_left_count = torch.cat([rgb_down_left_count, concat_row_rgb], 3).to(self.device)
        rgb_down_right_count = torch.cat([rgb_down_right_count, concat_column_minus_rgb], 2).to(self.device)
        rgb_down_right_count = torch.cat([rgb_down_right_count, concat_row_rgb], 3).to(self.device)
        rgb_up_left_count = torch.cat([concat_column_minus_rgb, rgb_up_left_count], 2).to(self.device)
        rgb_up_left_count = torch.cat([concat_row_rgb, rgb_up_left_count], 3).to(self.device)
        rgb_up_right_count = torch.cat([rgb_up_right_count, concat_column_minus_rgb], 2).to(self.device)
        rgb_up_right_count = torch.cat([concat_row_rgb, rgb_up_right_count], 3).to(self.device)
        # print("## pass, RGB phase 4 ##")

        rgb_orig_down = rgbs_flatten * rgb_down_count
        rgb_orig_up = rgbs_flatten * rgb_up_count
        rgb_orig_left = rgbs_flatten * rgb_left_count
        rgb_orig_right = rgbs_flatten * rgb_right_count
        rgb_orig_down_left = rgbs_flatten * rgb_down_left_count
        rgb_orig_down_right = rgbs_flatten * rgb_down_right_count
        rgb_orig_up_left = rgbs_flatten * rgb_up_left_count
        rgb_orig_up_right = rgbs_flatten * rgb_up_right_count
        
        
        # 2018/8/1 added, processing of gts
        # print(ground_truth_flatten.shape)
        '''
        ## mono direct
        prediction_down = b_fuse_out_flatten[:,:,1:image_height].to(self.device)
        prediction_up = b_fuse_out_flatten[:,:,0:image_height-1].to(self.device)
        prediction_left = b_fuse_out_flatten[:,0:image_width-1,:].to(self.device)
        prediction_right = b_fuse_out_flatten[:,1:image_width,:].to(self.device)
        ## bi-direct
        prediction_down_left = b_fuse_out_flatten[:,0:image_width-1,1:image_height].to(self.device)
        prediction_down_right = b_fuse_out_flatten[:,1:image_width,1:image_height].to(self.device)
        prediction_up_left = b_fuse_out_flatten[:,0:image_width-1,0:image_height-1].to(self.device)
        prediction_up_right = b_fuse_out_flatten[:,1:image_width,0:image_height-1].to(self.device)
        '''
        gt_down = ground_truth_flatten[:,:,1:image_height].to(self.device)
        gt_up = ground_truth_flatten[:,:,0:image_height-1].to(self.device)
        gt_left = ground_truth_flatten[:,0:image_width-1,:].to(self.device)
        gt_right = ground_truth_flatten[:,1:image_width,:].to(self.device)
        gt_down_left = ground_truth_flatten[:,0:image_width-1,1:image_height].to(self.device)
        gt_down_right = ground_truth_flatten[:,1:image_width,1:image_height].to(self.device)
        gt_up_left = ground_truth_flatten[:,0:image_width-1,0:image_height-1].to(self.device)
        gt_up_right = ground_truth_flatten[:,1:image_width,0:image_height-1].to(self.device)
        
        '''
        prediction_down_count = torch.ones(prediction_down.shape,device=self.device)
        prediction_up_count = torch.ones(prediction_up.shape,device=self.device)
        prediction_left_count = torch.ones(prediction_left.shape,device=self.device)
        prediction_right_count = torch.ones(prediction_right.shape,device=self.device)
        prediction_down_left_count = torch.ones(prediction_down_left.shape,device=self.device)
        prediction_down_right_count = torch.ones(prediction_down_right.shape,device=self.device)
        prediction_up_left_count = torch.ones(prediction_up_left.shape,device=self.device)
        prediction_up_right_count = torch.ones(prediction_up_right.shape,device=self.device)
        '''
        gt_down_count = torch.ones(gt_down.shape,device=self.device)
        gt_up_count = torch.ones(gt_up.shape,device=self.device)
        gt_left_count = torch.ones(gt_left.shape,device=self.device)
        gt_right_count = torch.ones(gt_right.shape,device=self.device)
        gt_down_left_count = torch.ones(gt_down_left.shape,device=self.device)
        gt_down_right_count = torch.ones(gt_down_right.shape,device=self.device)
        gt_up_left_count = torch.ones(gt_up_left.shape,device=self.device)
        gt_up_right_count = torch.ones(gt_up_right.shape,device=self.device)
        # print("## GT phase 1 pass ##")
        # reuse prediction partss
        #concat_row = torch.zeros([1, 1, image_width], torch.float32)
        #concat_column = torch.zeros([1,image_height,1], torch.float32)
        #concat_column_minus = torch.zeros([1,image_height-1,1], torch.float32)
        
        '''
        #padding the unfill space
        prediction_down = torch.cat([prediction_down, concat_row], 2).to(self.device)
        prediction_up = torch.cat([concat_row, prediction_up], 2).to(self.device)
        prediction_left = torch.cat([concat_column, prediction_left], 1).to(self.device)
        prediction_right = torch.cat([prediction_right, concat_column], 1).to(self.device)
        # print("## pass, phase 1 ##") 
        prediction_down_left = torch.cat([concat_column_minus, prediction_down_left], 1).to(self.device)
        prediction_down_left = torch.cat([prediction_down_left, concat_row], 2).to(self.device)
        prediction_down_right = torch.cat([prediction_down_right, concat_column_minus], 1).to(self.device)
        prediction_down_right = torch.cat([prediction_down_right, concat_row], 2).to(self.device)
        prediction_up_left = torch.cat([concat_column_minus, prediction_up_left], 1).to(self.device)
        prediction_up_left = torch.cat([concat_row, prediction_up_left], 2).to(self.device)
        prediction_up_right = torch.cat([prediction_up_right, concat_column_minus], 1).to(self.device)
        prediction_up_right = torch.cat([concat_row, prediction_up_right], 2).to(self.device)  
        # print("## pass, phase 2 ##") 
        # padding count 
        prediction_down_count = torch.cat([prediction_down_count, concat_row], 2).to(self.device)
        prediction_up_count = torch.cat([concat_row, prediction_up_count], 2).to(self.device)
        prediction_left_count = torch.cat([concat_column, prediction_left_count], 1).to(self.device)
        prediction_right_count = torch.cat([prediction_right_count, concat_column], 1).to(self.device)
        # print("## pass, phase 3 ##") 
        #print(concat_column_minus.shape)
        #print(prediction_down_left.shape)
        prediction_down_left_count = torch.cat([concat_column_minus, prediction_down_left_count], 1).to(self.device)
        prediction_down_left_count = torch.cat([prediction_down_left_count, concat_row], 2).to(self.device)
        prediction_down_right_count = torch.cat([prediction_down_right_count, concat_column_minus], 1).to(self.device)
        prediction_down_right_count = torch.cat([prediction_down_right_count, concat_row], 2).to(self.device)
        prediction_up_left_count = torch.cat([concat_column_minus, prediction_up_left_count], 1).to(self.device)
        prediction_up_left_count = torch.cat([concat_row, prediction_up_left_count], 2).to(self.device)
        prediction_up_right_count = torch.cat([prediction_up_right_count, concat_column_minus], 1).to(self.device)
        prediction_up_right_count = torch.cat([concat_row, prediction_up_right_count], 2).to(self.device) 
        '''
        
        gt_down = torch.cat([gt_down, concat_row], 2)
        gt_up = torch.cat([concat_row,gt_up], 2)
        gt_left = torch.cat([concat_column, gt_left], 1)
        gt_right = torch.cat([gt_right, concat_column], 1)
        
        gt_down_left = torch.cat([concat_column_minus, gt_down_left], 1)
        gt_down_right = torch.cat([gt_down_right, concat_column_minus], 1)
        gt_up_left = torch.cat([concat_column_minus, gt_up_left], 1)
        gt_up_right = torch.cat([gt_up_right, concat_column_minus], 1)
        gt_down_left = torch.cat([gt_down_left, concat_row], 2)
        gt_down_right = torch.cat([gt_down_right, concat_row], 2)
        gt_up_left = torch.cat([concat_row, gt_up_left], 2)
        gt_up_right = torch.cat([concat_row, gt_up_right], 2)
        # print("## GT phase 2-1 pass ##")        
        gt_down_count = torch.cat([gt_down_count, concat_row], 2)
        gt_up_count = torch.cat([concat_row, gt_up_count], 2)
        gt_left_count = torch.cat([concat_column, gt_left_count], 1)
        gt_right_count = torch.cat([gt_right_count, concat_column], 1)
        gt_down_left_count = torch.cat([concat_column_minus, gt_down_left_count], 1)
        gt_down_right_count = torch.cat([gt_down_right_count, concat_column_minus], 1)
        gt_up_left_count = torch.cat([concat_column_minus, gt_up_left_count], 1)
        gt_up_right_count = torch.cat([gt_up_right_count, concat_column_minus], 1)
        gt_down_left_count = torch.cat([gt_down_left_count, concat_row], 2)
        gt_down_right_count = torch.cat([gt_down_right_count, concat_row], 2)
        gt_up_left_count = torch.cat([concat_row, gt_up_left_count], 2)
        gt_up_right_count = torch.cat([concat_row, gt_up_right_count], 2)
        # print("## GT phase 2-2 pass ##")
        
                
        gt_orig_down = ground_truth_flatten * gt_down_count
        gt_orig_up =  ground_truth_flatten * gt_up_count
        gt_orig_left =  ground_truth_flatten * gt_left_count
        gt_orig_right = ground_truth_flatten * gt_right_count
        gt_orig_down_left =  ground_truth_flatten * gt_down_left_count
        gt_orig_down_right = ground_truth_flatten * gt_down_right_count
        gt_orig_up_left = ground_truth_flatten * gt_up_left_count
        gt_orig_up_right =  ground_truth_flatten * gt_up_right_count
        # print("## GT phase 3 pass ##")
        
        # before 2018/4/21 & 2018/4/23
        '''losses_down = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_down, logits=logit_down)
        losses_up = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_up, logits=logit_up)
        losses_left = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_left, logits=logit_left)
        losses_right = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_right, logits=logit_right)
        losses_down_left = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_down_left, logits=logit_down_left)
        losses_down_right = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_down_right, logits=logit_down_right)
        losses_up_left = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_up_left, logits=logit_up_left)
        losses_up_right = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_up_right, logits=logit_up_right)
        
        losses = (losses_down+losses_up+losses_left+losses_right+losses_down_left+losses_down_right+losses_up_left+losses_up_right) / prediction_count'''
        
        # 2018/4/22
        '''prediction_mean = (prediction_down + prediction_up + prediction_left + prediction_right + prediction_down_left + prediction_down_right + prediction_up_left + prediction_up_right) / prediction_count
        losses = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_mean, logits=prediction_flatten)'''
        
        # 2018/4/23
        # 2018/7/18 updated
        # 2018/8/20 updated
        down_power = torch.pow(rgb_down-rgb_orig_down,2)
        up_power = torch.pow(rgb_up-rgb_orig_up,2)
        left_power = torch.pow(rgb_left-rgb_orig_left,2)
        right_power = torch.pow(rgb_right-rgb_orig_right,2)
        down_left_power = torch.pow(rgb_down_left-rgb_orig_down_left,2)
        down_right_power = torch.pow(rgb_down_right-rgb_orig_down_right,2)
        up_left_power = torch.pow(rgb_up_left-rgb_orig_up_left,2)
        up_right_power = torch.pow(rgb_up_right-rgb_orig_up_right,2)
        '''
        print("## power pass ##")
        
        print(down_power.shape)
        print("self.sumRGBf*(down_power[:,:,:,0]+down_power[:,:,:,1]+down_power[:,:,:,2]) + self.sumDf*down_power[:,:,:,3]")
        '''
        down_sum = self.sumRGBf*(down_power[:,0,:,:]+down_power[:,1,:,:]+down_power[:,2,:,:]) + self.sumDf*down_power[:,3,:,:]
        up_sum = self.sumRGBf*(up_power[:,0,:,:]+up_power[:,1,:,:]+up_power[:,2,:,:]) + self.sumDf*up_power[:,3,:,:]
        left_sum = self.sumRGBf*(left_power[:,0,:,:]+left_power[:,1,:,:]+left_power[:,2,:,:]) + self.sumDf*left_power[:,3,:,:]
        right_sum = self.sumRGBf*(right_power[:,0,:,:]+right_power[:,1,:,:]+right_power[:,2,:,:]) + self.sumDf*right_power[:,3,:,:]
        down_left_sum = self.sumRGBf*(down_left_power[:,0,:,:]+down_left_power[:,1,:,:]+down_left_power[:,2,:,:]) + self.sumDf*down_left_power[:,3,:,:]
        down_right_sum = self.sumRGBf*(down_right_power[:,0,:,:]+down_right_power[:,1,:,:]+down_right_power[:,2,:,:]) + self.sumDf*down_right_power[:,3,:,:]
        up_left_sum = self.sumRGBf*(up_left_power[:,0,:,:]+up_left_power[:,1,:,:]+up_left_power[:,2,:,:]) + self.sumDf*up_left_power[:,3,:,:]
        up_right_sum = self.sumRGBf*(up_right_power[:,0,:,:]+up_right_power[:,1,:,:]+up_right_power[:,2,:,:]) + self.sumDf*up_right_power[:,3,:,:]
        '''
        print(down_sum.shape)
        print("## sumRGBf pass ##")

        print(prediction_down.shape)
        print(logit_down.shape)
        print(gt_down.shape)
        print("torch.exp(-down_sum /(128) ) * torch.abs(prediction_down-logit_down) * torch.abs(-1+prediction_down-gt_down)")
        print(torch.exp(-down_sum /(128) ).shape)
        print(torch.abs(prediction_down-logit_down).shape)
        print(torch.abs(-1+prediction_down-gt_down).shape)
        '''
        
        losses_down = torch.exp(-down_sum /(128) ) * torch.abs(prediction_down-logit_down) * torch.abs(-1+prediction_down-gt_down)
        losses_up = torch.exp(-up_sum /(128))*torch.abs(prediction_up-logit_up)*torch.abs(-1+prediction_up-gt_up)
        losses_left = torch.exp(-left_sum /(128))*torch.abs(prediction_left-logit_left)*torch.abs(-1+prediction_left-gt_left)
        losses_right = torch.exp(-right_sum /(128))*torch.abs(prediction_right-logit_right)*torch.abs(-1+prediction_right-gt_right)
        losses_down_left = torch.exp(-down_left_sum /(128))*torch.abs(prediction_down_left-logit_down_left)*torch.abs(-1+prediction_down_left-gt_down_left)
        losses_down_right = torch.exp(-down_right_sum /(128))*torch.abs(prediction_down_right-logit_down_right)*torch.abs(-1+prediction_down_right-gt_down_right)
        losses_up_left = torch.exp(-up_left_sum /(128))*torch.abs(prediction_up_left-logit_up_left)*torch.abs(-1+prediction_up_left-gt_up_left)
        losses_up_right = torch.exp(-up_right_sum /(128))*torch.abs(prediction_up_right-logit_up_right)*torch.abs(-1+prediction_up_right-gt_up_right)
        # print("## loss =1 pass ##")
        losses = (losses_down+losses_up+losses_left+losses_right+losses_down_left+losses_down_right+losses_up_left+losses_up_right) / prediction_count
        
        losses = torch.mean(losses)
        
        # print("## loss =2 pass ##")
        # losses = torch.cast(losses, tf.float64)
        
        return losses
        
        # """        
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
class SP1d_Spco_Loss(nn.Module):
    def __init__(self, device, weight=[1.0, 1.0, 0.5]):
        # --------------------------------------------
        # Initialization
        # --------------------------------------------
        super(SP1d_Spco_Loss, self).__init__()
        self.weight = weight
        self.cross_entropy_with_logits = nn.BCEWithLogitsLoss()
        self.device=device
        
        #weighting param
        self.sumRGBf=9600000
        self.sumDf=4800000
        
        #inputs => get from forward
        # fuse_result, b1_result, b2_result, b3_result, b4_result, b5_result, b6_result , gt

    def forward(self, epoch, fuse_result, gts, rgbs, depths):
        # --------------------------------------------
        # Define forward pass
        # --------------------------------------------
        # Transform targets to one-hot vector
        
        #####################################
        # example code of custom 
        #####################################
        
        # fusion
        fuse_losses = self.model_fusion_loss(fuse_result,gts)
        '''
        # branches
        # loss_branch = self.model_fusion_loss(b1_result, b2_result, b3_result, b4_result, b5_result, b6_result, gt)
        loss_branch_b1 = self.model_side_loss(b1_result, gts)
        loss_branch_b2 = self.model_side_loss(b2_result, gts)
        loss_branch_b3 = self.model_side_loss(b3_result, gts)
        #print(b4_result.shape)
        #print(gts.shape)
        # print(torch.max(gts),"/",torch.min(gts))
        loss_branch_b4 = self.model_side_loss(b4_result, gts)
        loss_branch_b5 = self.model_side_loss(b5_result, gts)
        loss_branch_b6 = self.model_side_loss(b6_result, gts)
        side_losses=loss_branch_b1+loss_branch_b2+loss_branch_b3+loss_branch_b4+loss_branch_b5+loss_branch_b6
        '''
        # spatial coherence
        '''
        if(len(rgbs.shape)==len(depths.shape) and epoch>30):
          loss_sp = self.model_neighbor_loss(fuse_result.float(), rgbs.float(), depths.float(), gts.float()) # result2d_out / fuse_result.float()
        else:
          loss_sp=0
        '''
        
        final_loss=fuse_losses#+side_losses # + loss_sp
        

        return final_loss
        
    def model_side_loss(self, b_side_out, ground_truth):
        ground_truth=torch.squeeze(ground_truth).to(self.device)
        image_size=ground_truth.shape
        b_side_out = F.interpolate(b_side_out,image_size).to(self.device)
        # sigmoid_b_side_out=torch.sigmoid(b_side_out).to(self.device) # already been sigmoid
        sigmoid_b_side_out=torch.squeeze(b_side_out).to(self.device) 
        #ground_truth.requires_grad=True
        losses = torch.sum(torch.abs(sigmoid_b_side_out-ground_truth))
        #self.cross_entropy_with_logits(sigmoid_b_side_out, ground_truth).to(self.device) # self.cross_entropy_with_logits(sigmoid_b_side_out,ground_truth)
        losses=torch.mean(losses)
        # print("model_side_loss", losses)
        return losses
        
    # RuntimeError: CUDA error: device-side assert triggered.   => F.bce generate the error        
    def model_fusion_loss(self, b_fuse_out, ground_truth):
        #print(b_fuse_out.shape)
        #print(ground_truth.shape)
        ground_truth=torch.squeeze(ground_truth).to(self.device)
        image_size=ground_truth.shape
        b_fuse_out = F.interpolate(b_fuse_out,image_size).to(self.device)
        # sigmoid_b_fuse_out=torch.sigmoid(b_fuse_out).to(self.device) # already been sigmoid        
        sigmoid_b_fuse_out=torch.squeeze(b_fuse_out).to(self.device)
        #ground_truth.requires_grad=True
        
        '''
        print("ground_truth.shape",ground_truth.shape)
        print("sigmoid_b_fuse_out.shape",sigmoid_b_fuse_out.shape)
        print(torch.max(sigmoid_b_fuse_out), torch.min(sigmoid_b_fuse_out))
        print(torch.max(ground_truth), torch.min(ground_truth))
        '''
        losses = torch.sum(torch.abs(sigmoid_b_fuse_out-ground_truth))
        #self.cross_entropy_with_logits(sigmoid_b_fuse_out, ground_truth).to(self.device)  #self.cross_entropy_with_logits(sigmoid_b_fuse_out,ground_truth) #sigmoid_cross_entropy_with_logits
        losses=torch.mean(losses)
        
        # print("sigmoid_b_fuse_out", losses)
        
        return losses
 
    def model_neighbor_loss(self, b_fuse_out, rgbs, depths, gts):
        # groundtruth
        ground_truth_flatten=torch.squeeze(gts,0).to(self.device)
        image_size=torch.squeeze(gts).shape 
        #size
        image_height=image_size[1]
        image_width=image_size[0]
        image_size=[image_width,image_height]
        '''        
        print("## ground_truth ##") 
        print(ground_truth.shape)
        print("## image_size ##") 
        print(image_size)
        '''
        b_fuse_out = F.interpolate(b_fuse_out,image_size).to(self.device)
        b_fuse_out_flatten=torch.squeeze(b_fuse_out, 0) 
        '''
        # norm and depth # !! norm !!
        print("## b_fuse_out_flatten ##") 
        print(b_fuse_out_flatten.shape) 
        
        print("## rgbs/depth ##") 
        print(rgbs.shape)  
        print(depths.shape)
        '''
        rgbs_flatten = torch.cat((rgbs, depths), 1).to(self.device)
        prediction_flatten = b_fuse_out_flatten
        '''
        print("## prediction_flatten ##") 
        print(prediction_flatten.shape)  
        '''
        
        ###  => Here is (B, C, W ,H) (Because of PIL trasnformation, confirmed)
        #8 directs
        ## mono direct
        prediction_down = b_fuse_out_flatten[:,:,1:image_height].to(self.device)
        prediction_up = b_fuse_out_flatten[:,:,0:image_height-1].to(self.device)
        prediction_left = b_fuse_out_flatten[:,0:image_width-1,:].to(self.device)
        prediction_right = b_fuse_out_flatten[:,1:image_width,:].to(self.device)
        ## bi-direct
        prediction_down_left = b_fuse_out_flatten[:,0:image_width-1,1:image_height].to(self.device)
        prediction_down_right = b_fuse_out_flatten[:,1:image_width,1:image_height].to(self.device)
        prediction_up_left = b_fuse_out_flatten[:,0:image_width-1,0:image_height-1].to(self.device)
        prediction_up_right = b_fuse_out_flatten[:,1:image_width,0:image_height-1].to(self.device)
        
        # 1-count
        prediction_down_count = torch.ones(prediction_down.shape,device=self.device)
        prediction_up_count = torch.ones(prediction_up.shape,device=self.device)
        prediction_left_count = torch.ones(prediction_left.shape,device=self.device)
        prediction_right_count = torch.ones(prediction_right.shape,device=self.device)
        prediction_down_left_count = torch.ones(prediction_down_left.shape,device=self.device)
        prediction_down_right_count = torch.ones(prediction_down_right.shape,device=self.device)
        prediction_up_left_count = torch.ones(prediction_up_left.shape,device=self.device)
        prediction_up_right_count = torch.ones(prediction_up_right.shape,device=self.device)
        
        # zero count
        concat_row = torch.zeros([1, image_width, 1], device=self.device)
        concat_column = torch.zeros([1, 1, image_height], device=self.device)
        concat_column_minus = torch.zeros([1, 1, image_height-1], device=self.device)

        #padding the unfill space
        prediction_down = torch.cat([prediction_down, concat_row], 2).to(self.device)
        prediction_up = torch.cat([concat_row, prediction_up], 2).to(self.device)
        prediction_left = torch.cat([concat_column, prediction_left], 1).to(self.device)
        prediction_right = torch.cat([prediction_right, concat_column], 1).to(self.device)
        # print("## pass, phase 1 ##") 
        prediction_down_left = torch.cat([concat_column_minus, prediction_down_left], 1).to(self.device)
        prediction_down_left = torch.cat([prediction_down_left, concat_row], 2).to(self.device)
        prediction_down_right = torch.cat([prediction_down_right, concat_column_minus], 1).to(self.device)
        prediction_down_right = torch.cat([prediction_down_right, concat_row], 2).to(self.device)
        prediction_up_left = torch.cat([concat_column_minus, prediction_up_left], 1).to(self.device)
        prediction_up_left = torch.cat([concat_row, prediction_up_left], 2).to(self.device)
        prediction_up_right = torch.cat([prediction_up_right, concat_column_minus], 1).to(self.device)
        prediction_up_right = torch.cat([concat_row, prediction_up_right], 2).to(self.device)  
        # print("## pass, phase 2 ##") 
        # padding count 
        prediction_down_count = torch.cat([prediction_down_count, concat_row], 2).to(self.device)
        prediction_up_count = torch.cat([concat_row, prediction_up_count], 2).to(self.device)
        prediction_left_count = torch.cat([concat_column, prediction_left_count], 1).to(self.device)
        prediction_right_count = torch.cat([prediction_right_count, concat_column], 1).to(self.device)
        # print("## pass, phase 3 ##") 
        #print(concat_column_minus.shape)
        #print(prediction_down_left.shape)
        prediction_down_left_count = torch.cat([concat_column_minus, prediction_down_left_count], 1).to(self.device)
        prediction_down_left_count = torch.cat([prediction_down_left_count, concat_row], 2).to(self.device)
        prediction_down_right_count = torch.cat([prediction_down_right_count, concat_column_minus], 1).to(self.device)
        prediction_down_right_count = torch.cat([prediction_down_right_count, concat_row], 2).to(self.device)
        prediction_up_left_count = torch.cat([concat_column_minus, prediction_up_left_count], 1).to(self.device)
        prediction_up_left_count = torch.cat([concat_row, prediction_up_left_count], 2).to(self.device)
        prediction_up_right_count = torch.cat([prediction_up_right_count, concat_column_minus], 1).to(self.device)
        prediction_up_right_count = torch.cat([concat_row, prediction_up_right_count], 2).to(self.device)  
        # print("## pass, phase 4 ##")
        
        # before 2018/4/21 & 2018/4/23
        logit_down = prediction_flatten * prediction_down_count
        logit_up = prediction_flatten * prediction_up_count
        logit_left = prediction_flatten * prediction_left_count
        logit_right = prediction_flatten * prediction_right_count
        logit_down_left = prediction_flatten * prediction_down_left_count
        logit_down_right = prediction_flatten * prediction_down_right_count
        logit_up_left = prediction_flatten * prediction_up_left_count
        logit_up_right = prediction_flatten * prediction_up_right_count
        
        prediction_count = prediction_down_count + prediction_up_count + prediction_left_count + prediction_right_count + prediction_down_left_count + prediction_down_right_count + prediction_up_left_count + prediction_up_right_count
        # preprocessing of RGBs
        # print("## pass, Logit ##")
        
        # 4-d
        # print("## RGB, not original size but 640x640 ##") 
        rgb_down = rgbs_flatten[:,:,:,1:image_height].to(self.device)
        rgb_up = rgbs_flatten[:,:,:,0:image_height-1].to(self.device)
        rgb_left = rgbs_flatten[:,:,0:image_width-1,:].to(self.device)
        rgb_right = rgbs_flatten[:,:,1:image_width,:].to(self.device)
        # other 4-d / 8-d
        rgb_down_left = rgbs_flatten[:,:,0:image_width-1,1:image_height].to(self.device)
        rgb_down_right = rgbs_flatten[:,:,1:image_width,1:image_height].to(self.device)
        rgb_up_left = rgbs_flatten[:,:,0:image_width-1,0:image_height-1].to(self.device)
        rgb_up_right = rgbs_flatten[:,:,1:image_width,0:image_height-1].to(self.device)
        # print("## pass, RGB phase 1 ##")
        
        rgb_down_count = torch.ones(rgb_down.shape, device=self.device)
        rgb_up_count = torch.ones(rgb_up.shape, device=self.device)
        rgb_left_count = torch.ones(rgb_left.shape, device=self.device)
        rgb_right_count = torch.ones(rgb_right.shape, device=self.device)
        
        rgb_down_left_count = torch.ones(rgb_down_left.shape, device=self.device)
        rgb_down_right_count = torch.ones(rgb_down_right.shape, device=self.device)
        rgb_up_left_count = torch.ones(rgb_up_left.shape, device=self.device)
        rgb_up_right_count = torch.ones(rgb_up_right.shape, device=self.device)
        
        concat_row_rgb = torch.zeros([1, 4, image_width, 1], device=self.device)
        concat_column_rgb = torch.zeros([1, 4, 1, image_height], device=self.device)
        concat_column_minus_rgb = torch.zeros([1, 4, 1, image_height-1], device=self.device)
        # print("## pass, RGB phase 2 ##")
        
        rgb_down = torch.cat([rgb_down, concat_row_rgb], 3).to(self.device)
        rgb_up = torch.cat([concat_row_rgb, rgb_up], 3).to(self.device)
        rgb_left = torch.cat([concat_column_rgb, rgb_left], 2).to(self.device)
        rgb_right = torch.cat([rgb_right, concat_column_rgb], 2).to(self.device)
        
        rgb_down_left = torch.cat([concat_column_minus_rgb, rgb_down_left], 2).to(self.device)
        rgb_down_left = torch.cat([rgb_down_left, concat_row_rgb], 3).to(self.device)
        rgb_down_right = torch.cat([rgb_down_right, concat_column_minus_rgb], 2).to(self.device)
        rgb_down_right = torch.cat([rgb_down_right, concat_row_rgb], 3).to(self.device)
        rgb_up_left = torch.cat([concat_column_minus_rgb, rgb_up_left], 2).to(self.device)
        rgb_up_left = torch.cat([concat_row_rgb, rgb_up_left], 3).to(self.device)
        rgb_up_right = torch.cat([rgb_up_right, concat_column_minus_rgb], 2).to(self.device)
        rgb_up_right = torch.cat([concat_row_rgb, rgb_up_right], 3).to(self.device)
        # print("## pass, RGB phase 3 ##")
        #count
        rgb_down_count = torch.cat([rgb_down_count, concat_row_rgb], 3).to(self.device)
        rgb_up_count = torch.cat([concat_row_rgb, rgb_up_count], 3).to(self.device)
        rgb_left_count = torch.cat([concat_column_rgb, rgb_left_count], 2).to(self.device)
        rgb_right_count = torch.cat([rgb_right_count, concat_column_rgb], 2).to(self.device)
        
        rgb_down_left_count = torch.cat([concat_column_minus_rgb, rgb_down_left_count], 2).to(self.device)
        rgb_down_left_count = torch.cat([rgb_down_left_count, concat_row_rgb], 3).to(self.device)
        rgb_down_right_count = torch.cat([rgb_down_right_count, concat_column_minus_rgb], 2).to(self.device)
        rgb_down_right_count = torch.cat([rgb_down_right_count, concat_row_rgb], 3).to(self.device)
        rgb_up_left_count = torch.cat([concat_column_minus_rgb, rgb_up_left_count], 2).to(self.device)
        rgb_up_left_count = torch.cat([concat_row_rgb, rgb_up_left_count], 3).to(self.device)
        rgb_up_right_count = torch.cat([rgb_up_right_count, concat_column_minus_rgb], 2).to(self.device)
        rgb_up_right_count = torch.cat([concat_row_rgb, rgb_up_right_count], 3).to(self.device)
        # print("## pass, RGB phase 4 ##")

        rgb_orig_down = rgbs_flatten * rgb_down_count
        rgb_orig_up = rgbs_flatten * rgb_up_count
        rgb_orig_left = rgbs_flatten * rgb_left_count
        rgb_orig_right = rgbs_flatten * rgb_right_count
        rgb_orig_down_left = rgbs_flatten * rgb_down_left_count
        rgb_orig_down_right = rgbs_flatten * rgb_down_right_count
        rgb_orig_up_left = rgbs_flatten * rgb_up_left_count
        rgb_orig_up_right = rgbs_flatten * rgb_up_right_count
        
        
        # 2018/8/1 added, processing of gts
        # print(ground_truth_flatten.shape)
        '''
        ## mono direct
        prediction_down = b_fuse_out_flatten[:,:,1:image_height].to(self.device)
        prediction_up = b_fuse_out_flatten[:,:,0:image_height-1].to(self.device)
        prediction_left = b_fuse_out_flatten[:,0:image_width-1,:].to(self.device)
        prediction_right = b_fuse_out_flatten[:,1:image_width,:].to(self.device)
        ## bi-direct
        prediction_down_left = b_fuse_out_flatten[:,0:image_width-1,1:image_height].to(self.device)
        prediction_down_right = b_fuse_out_flatten[:,1:image_width,1:image_height].to(self.device)
        prediction_up_left = b_fuse_out_flatten[:,0:image_width-1,0:image_height-1].to(self.device)
        prediction_up_right = b_fuse_out_flatten[:,1:image_width,0:image_height-1].to(self.device)
        '''
        gt_down = ground_truth_flatten[:,:,1:image_height].to(self.device)
        gt_up = ground_truth_flatten[:,:,0:image_height-1].to(self.device)
        gt_left = ground_truth_flatten[:,0:image_width-1,:].to(self.device)
        gt_right = ground_truth_flatten[:,1:image_width,:].to(self.device)
        gt_down_left = ground_truth_flatten[:,0:image_width-1,1:image_height].to(self.device)
        gt_down_right = ground_truth_flatten[:,1:image_width,1:image_height].to(self.device)
        gt_up_left = ground_truth_flatten[:,0:image_width-1,0:image_height-1].to(self.device)
        gt_up_right = ground_truth_flatten[:,1:image_width,0:image_height-1].to(self.device)
        
        '''
        prediction_down_count = torch.ones(prediction_down.shape,device=self.device)
        prediction_up_count = torch.ones(prediction_up.shape,device=self.device)
        prediction_left_count = torch.ones(prediction_left.shape,device=self.device)
        prediction_right_count = torch.ones(prediction_right.shape,device=self.device)
        prediction_down_left_count = torch.ones(prediction_down_left.shape,device=self.device)
        prediction_down_right_count = torch.ones(prediction_down_right.shape,device=self.device)
        prediction_up_left_count = torch.ones(prediction_up_left.shape,device=self.device)
        prediction_up_right_count = torch.ones(prediction_up_right.shape,device=self.device)
        '''
        gt_down_count = torch.ones(gt_down.shape,device=self.device)
        gt_up_count = torch.ones(gt_up.shape,device=self.device)
        gt_left_count = torch.ones(gt_left.shape,device=self.device)
        gt_right_count = torch.ones(gt_right.shape,device=self.device)
        gt_down_left_count = torch.ones(gt_down_left.shape,device=self.device)
        gt_down_right_count = torch.ones(gt_down_right.shape,device=self.device)
        gt_up_left_count = torch.ones(gt_up_left.shape,device=self.device)
        gt_up_right_count = torch.ones(gt_up_right.shape,device=self.device)
        # print("## GT phase 1 pass ##")
        # reuse prediction partss
        #concat_row = torch.zeros([1, 1, image_width], torch.float32)
        #concat_column = torch.zeros([1,image_height,1], torch.float32)
        #concat_column_minus = torch.zeros([1,image_height-1,1], torch.float32)
        
        '''
        #padding the unfill space
        prediction_down = torch.cat([prediction_down, concat_row], 2).to(self.device)
        prediction_up = torch.cat([concat_row, prediction_up], 2).to(self.device)
        prediction_left = torch.cat([concat_column, prediction_left], 1).to(self.device)
        prediction_right = torch.cat([prediction_right, concat_column], 1).to(self.device)
        # print("## pass, phase 1 ##") 
        prediction_down_left = torch.cat([concat_column_minus, prediction_down_left], 1).to(self.device)
        prediction_down_left = torch.cat([prediction_down_left, concat_row], 2).to(self.device)
        prediction_down_right = torch.cat([prediction_down_right, concat_column_minus], 1).to(self.device)
        prediction_down_right = torch.cat([prediction_down_right, concat_row], 2).to(self.device)
        prediction_up_left = torch.cat([concat_column_minus, prediction_up_left], 1).to(self.device)
        prediction_up_left = torch.cat([concat_row, prediction_up_left], 2).to(self.device)
        prediction_up_right = torch.cat([prediction_up_right, concat_column_minus], 1).to(self.device)
        prediction_up_right = torch.cat([concat_row, prediction_up_right], 2).to(self.device)  
        # print("## pass, phase 2 ##") 
        # padding count 
        prediction_down_count = torch.cat([prediction_down_count, concat_row], 2).to(self.device)
        prediction_up_count = torch.cat([concat_row, prediction_up_count], 2).to(self.device)
        prediction_left_count = torch.cat([concat_column, prediction_left_count], 1).to(self.device)
        prediction_right_count = torch.cat([prediction_right_count, concat_column], 1).to(self.device)
        # print("## pass, phase 3 ##") 
        #print(concat_column_minus.shape)
        #print(prediction_down_left.shape)
        prediction_down_left_count = torch.cat([concat_column_minus, prediction_down_left_count], 1).to(self.device)
        prediction_down_left_count = torch.cat([prediction_down_left_count, concat_row], 2).to(self.device)
        prediction_down_right_count = torch.cat([prediction_down_right_count, concat_column_minus], 1).to(self.device)
        prediction_down_right_count = torch.cat([prediction_down_right_count, concat_row], 2).to(self.device)
        prediction_up_left_count = torch.cat([concat_column_minus, prediction_up_left_count], 1).to(self.device)
        prediction_up_left_count = torch.cat([concat_row, prediction_up_left_count], 2).to(self.device)
        prediction_up_right_count = torch.cat([prediction_up_right_count, concat_column_minus], 1).to(self.device)
        prediction_up_right_count = torch.cat([concat_row, prediction_up_right_count], 2).to(self.device) 
        '''
        
        gt_down = torch.cat([gt_down, concat_row], 2)
        gt_up = torch.cat([concat_row,gt_up], 2)
        gt_left = torch.cat([concat_column, gt_left], 1)
        gt_right = torch.cat([gt_right, concat_column], 1)
        
        gt_down_left = torch.cat([concat_column_minus, gt_down_left], 1)
        gt_down_right = torch.cat([gt_down_right, concat_column_minus], 1)
        gt_up_left = torch.cat([concat_column_minus, gt_up_left], 1)
        gt_up_right = torch.cat([gt_up_right, concat_column_minus], 1)
        gt_down_left = torch.cat([gt_down_left, concat_row], 2)
        gt_down_right = torch.cat([gt_down_right, concat_row], 2)
        gt_up_left = torch.cat([concat_row, gt_up_left], 2)
        gt_up_right = torch.cat([concat_row, gt_up_right], 2)
        # print("## GT phase 2-1 pass ##")        
        gt_down_count = torch.cat([gt_down_count, concat_row], 2)
        gt_up_count = torch.cat([concat_row, gt_up_count], 2)
        gt_left_count = torch.cat([concat_column, gt_left_count], 1)
        gt_right_count = torch.cat([gt_right_count, concat_column], 1)
        gt_down_left_count = torch.cat([concat_column_minus, gt_down_left_count], 1)
        gt_down_right_count = torch.cat([gt_down_right_count, concat_column_minus], 1)
        gt_up_left_count = torch.cat([concat_column_minus, gt_up_left_count], 1)
        gt_up_right_count = torch.cat([gt_up_right_count, concat_column_minus], 1)
        gt_down_left_count = torch.cat([gt_down_left_count, concat_row], 2)
        gt_down_right_count = torch.cat([gt_down_right_count, concat_row], 2)
        gt_up_left_count = torch.cat([concat_row, gt_up_left_count], 2)
        gt_up_right_count = torch.cat([concat_row, gt_up_right_count], 2)
        # print("## GT phase 2-2 pass ##")
        
                
        gt_orig_down = ground_truth_flatten * gt_down_count
        gt_orig_up =  ground_truth_flatten * gt_up_count
        gt_orig_left =  ground_truth_flatten * gt_left_count
        gt_orig_right = ground_truth_flatten * gt_right_count
        gt_orig_down_left =  ground_truth_flatten * gt_down_left_count
        gt_orig_down_right = ground_truth_flatten * gt_down_right_count
        gt_orig_up_left = ground_truth_flatten * gt_up_left_count
        gt_orig_up_right =  ground_truth_flatten * gt_up_right_count
        # print("## GT phase 3 pass ##")
        
        # before 2018/4/21 & 2018/4/23
        '''losses_down = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_down, logits=logit_down)
        losses_up = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_up, logits=logit_up)
        losses_left = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_left, logits=logit_left)
        losses_right = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_right, logits=logit_right)
        losses_down_left = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_down_left, logits=logit_down_left)
        losses_down_right = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_down_right, logits=logit_down_right)
        losses_up_left = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_up_left, logits=logit_up_left)
        losses_up_right = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_up_right, logits=logit_up_right)
        
        losses = (losses_down+losses_up+losses_left+losses_right+losses_down_left+losses_down_right+losses_up_left+losses_up_right) / prediction_count'''
        
        # 2018/4/22
        '''prediction_mean = (prediction_down + prediction_up + prediction_left + prediction_right + prediction_down_left + prediction_down_right + prediction_up_left + prediction_up_right) / prediction_count
        losses = torch.nn.sigmoid_cross_entropy_with_logits(labels=prediction_mean, logits=prediction_flatten)'''
        
        # 2018/4/23
        # 2018/7/18 updated
        # 2018/8/20 updated
        down_power = torch.pow(rgb_down-rgb_orig_down,2)
        up_power = torch.pow(rgb_up-rgb_orig_up,2)
        left_power = torch.pow(rgb_left-rgb_orig_left,2)
        right_power = torch.pow(rgb_right-rgb_orig_right,2)
        down_left_power = torch.pow(rgb_down_left-rgb_orig_down_left,2)
        down_right_power = torch.pow(rgb_down_right-rgb_orig_down_right,2)
        up_left_power = torch.pow(rgb_up_left-rgb_orig_up_left,2)
        up_right_power = torch.pow(rgb_up_right-rgb_orig_up_right,2)
        '''
        print("## power pass ##")
        
        print(down_power.shape)
        print("self.sumRGBf*(down_power[:,:,:,0]+down_power[:,:,:,1]+down_power[:,:,:,2]) + self.sumDf*down_power[:,:,:,3]")
        '''
        down_sum = self.sumRGBf*(down_power[:,0,:,:]+down_power[:,1,:,:]+down_power[:,2,:,:]) + self.sumDf*down_power[:,3,:,:]
        up_sum = self.sumRGBf*(up_power[:,0,:,:]+up_power[:,1,:,:]+up_power[:,2,:,:]) + self.sumDf*up_power[:,3,:,:]
        left_sum = self.sumRGBf*(left_power[:,0,:,:]+left_power[:,1,:,:]+left_power[:,2,:,:]) + self.sumDf*left_power[:,3,:,:]
        right_sum = self.sumRGBf*(right_power[:,0,:,:]+right_power[:,1,:,:]+right_power[:,2,:,:]) + self.sumDf*right_power[:,3,:,:]
        down_left_sum = self.sumRGBf*(down_left_power[:,0,:,:]+down_left_power[:,1,:,:]+down_left_power[:,2,:,:]) + self.sumDf*down_left_power[:,3,:,:]
        down_right_sum = self.sumRGBf*(down_right_power[:,0,:,:]+down_right_power[:,1,:,:]+down_right_power[:,2,:,:]) + self.sumDf*down_right_power[:,3,:,:]
        up_left_sum = self.sumRGBf*(up_left_power[:,0,:,:]+up_left_power[:,1,:,:]+up_left_power[:,2,:,:]) + self.sumDf*up_left_power[:,3,:,:]
        up_right_sum = self.sumRGBf*(up_right_power[:,0,:,:]+up_right_power[:,1,:,:]+up_right_power[:,2,:,:]) + self.sumDf*up_right_power[:,3,:,:]
        '''
        print(down_sum.shape)
        print("## sumRGBf pass ##")

        print(prediction_down.shape)
        print(logit_down.shape)
        print(gt_down.shape)
        print("torch.exp(-down_sum /(128) ) * torch.abs(prediction_down-logit_down) * torch.abs(-1+prediction_down-gt_down)")
        print(torch.exp(-down_sum /(128) ).shape)
        print(torch.abs(prediction_down-logit_down).shape)
        print(torch.abs(-1+prediction_down-gt_down).shape)
        '''
        
        losses_down = torch.exp(-down_sum /(128) ) * torch.abs(prediction_down-logit_down) * torch.abs(-1+prediction_down-gt_down)
        losses_up = torch.exp(-up_sum /(128))*torch.abs(prediction_up-logit_up)*torch.abs(-1+prediction_up-gt_up)
        losses_left = torch.exp(-left_sum /(128))*torch.abs(prediction_left-logit_left)*torch.abs(-1+prediction_left-gt_left)
        losses_right = torch.exp(-right_sum /(128))*torch.abs(prediction_right-logit_right)*torch.abs(-1+prediction_right-gt_right)
        losses_down_left = torch.exp(-down_left_sum /(128))*torch.abs(prediction_down_left-logit_down_left)*torch.abs(-1+prediction_down_left-gt_down_left)
        losses_down_right = torch.exp(-down_right_sum /(128))*torch.abs(prediction_down_right-logit_down_right)*torch.abs(-1+prediction_down_right-gt_down_right)
        losses_up_left = torch.exp(-up_left_sum /(128))*torch.abs(prediction_up_left-logit_up_left)*torch.abs(-1+prediction_up_left-gt_up_left)
        losses_up_right = torch.exp(-up_right_sum /(128))*torch.abs(prediction_up_right-logit_up_right)*torch.abs(-1+prediction_up_right-gt_up_right)
        # print("## loss =1 pass ##")
        losses = (losses_down+losses_up+losses_left+losses_right+losses_down_left+losses_down_right+losses_up_left+losses_up_right) / prediction_count
        
        losses = torch.mean(losses)
        
        # print("## loss =2 pass ##")
        # losses = torch.cast(losses, tf.float64)
        
        return losses
        
        # """        