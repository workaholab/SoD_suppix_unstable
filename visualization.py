from __future__ import print_function
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import math
from PIL import Image
from datetime import datetime

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

def visualize_results(train_mode, Ver_Train, epochs, img_size, predict_results, rgb_path, gt_path, dep_path, SAVE_REF=True):
    
    unloader = transforms.ToPILImage()

    name_base=os.path.basename(rgb_path)
    number_base=name_base[1:len(name_base)-8]
    
    #result saving
    if (train_mode):
      PATH_dir="VisualResults_T%d/Train/Epoch_%d/results_%s/"%(Ver_Train,epochs,number_base)
    else:
      PATH_dir="VisualResults_T%d/Test/results_%s/"%(Ver_Train,number_base)
      
    if not os.path.exists(PATH_dir):
       os.makedirs(PATH_dir)  #create PATH_dir
  
    PATH_resultfile="result_%s.png"%(number_base)
    PATH_rgbfile="rgb_test_%s.png"%(number_base)
    PATH_gtfile="gt_test_%s.png"%(number_base)
    PATH_depthfile="depth_test_%s.png"%(number_base)

    result_PATH=PATH_dir+PATH_resultfile
    rgb_PATH=PATH_dir+PATH_rgbfile
    gt_PATH=PATH_dir+PATH_gtfile
    depth_PATH=PATH_dir+PATH_depthfile
    #================================================      
    # Inputs shall be sigmoid
    f_result=predict_results[0]
    b1_res=predict_results[1]
    b2_res=predict_results[2]
    b3_res=predict_results[3]
    b4_res=predict_results[4]
    b5_res=predict_results[5]
    b6_res=predict_results[6]
    
    resultF_visual=F.interpolate(f_result.cpu().clone(),img_size) # interpolate too the original 
    b1_res=b1_res.cpu().clone()
    b2_res=b2_res.cpu().clone()
    b3_res=b3_res.cpu().clone()
    b4_res=b4_res.cpu().clone()
    b5_res=b5_res.cpu().clone()
    b6_res=b6_res.cpu().clone()
    # results    
    image=unloader(resultF_visual.squeeze(0)) 
    img_1=unloader(b1_res.squeeze(0))
    img_2=unloader(b2_res.squeeze(0))
    img_3=unloader(b3_res.squeeze(0))
    img_4=unloader(b4_res.squeeze(0))
    img_5=unloader(b5_res.squeeze(0))
    img_6=unloader(b6_res.squeeze(0))
    
    #data saving
    image.save(result_PATH)
    img_1.save(PATH_dir+"b1_result_%s.png"%(number_base))
    img_2.save(PATH_dir+"b2_result_%s.png"%(number_base))
    img_3.save(PATH_dir+"b3_result_%s.png"%(number_base))
    img_4.save(PATH_dir+"b4_result_%s.png"%(number_base))
    img_5.save(PATH_dir+"b5_result_%s.png"%(number_base))
    img_6.save(PATH_dir+"b6_result_%s.png"%(number_base))
    
    #original data / for reference
    if (SAVE_REF):
      rgb_img = Image.open(rgb_path).convert('RGB')
      gt_img = Image.open(gt_path).convert('L')
      # depth
      if(not dep_path):
          #empty
          dep_img=0
      else:
          dep_img = Image.open(dep_path)
      
      # space consuming
      rgb_img.save(rgb_PATH)
      gt_img.save(gt_PATH)
      
      if (dep_path!=0):
        dep_img.save(depth_PATH)
    else:
      csv_file="ref_files_"%(number_base)
      with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['RGB', 'DEPTH', 'GT'])
        writer.writerow([rgb_path, dep_path, gt_path])
        
def visual_tensor(input_tensor):
  input_tensor=torch.sigmoid(input_tensor)
  unloader = transforms.ToPILImage()
  print(input_tensor.shape)
  image=unloader(input_tensor.cpu().squeeze(0))
  print(image.size)
  image.save("output.png")

###############################################################################
def visualize_results_sp(train_mode, Ver_Train, epochs, img_size, predict_results, rgb_path, gt_path, dep_path, sp_segments):
    
    unloader = transforms.ToPILImage()

    name_base=os.path.basename(rgb_path)
    number_base=name_base[1:len(name_base)-8]
    
    #result saving
    if (train_mode):
      PATH_dir="SP_VisualResults_T%d/Train/Epoch_%d/results_%s/"%(Ver_Train,epochs,number_base)
    else:
      PATH_dir="SP_VisualResults_T%d/Test/results_%s/"%(Ver_Train,number_base)
    if not os.path.exists(PATH_dir):
       os.makedirs(PATH_dir)  #create PATH_dir
  
    PATH_resultfile="result_%s.png"%(number_base)
    PATH_rgbfile="rgb_test_%s.png"%(number_base)
    PATH_gtfile="gt_test_%s.png"%(number_base)
    PATH_depthfile="depth_test_%s.png"%(number_base)

    result_PATH=PATH_dir+PATH_resultfile
    rgb_PATH=PATH_dir+PATH_rgbfile
    gt_PATH=PATH_dir+PATH_gtfile
    depth_PATH=PATH_dir+PATH_depthfile
    '''
    print(result_PATH)
    print(rgb_PATH)
    print(gt_PATH)
    print(depth_PATH)
    '''
    #================================================      
    # print(predict_results)
    # back to CPU
    # ndarray
    # print(predict_results)
    resultF_visual=predict_results[0]
    b1_res=predict_results[1]
    b2_res=predict_results[2]
    b3_res=predict_results[3]
    b4_res=predict_results[4]
    b5_res=predict_results[5]
    b6_res=predict_results[6]
    
    resultF_visual=resultF_visual.cpu().clone()
    b1_res=b1_res.cpu().clone()
    b2_res=b2_res.cpu().clone()
    b3_res=b3_res.cpu().clone()
    b4_res=b4_res.cpu().clone()
    b5_res=b5_res.cpu().clone()
    b6_res=b6_res.cpu().clone()
    
    rgb_img = Image.open(rgb_path).convert('RGB')
    gt_img = Image.open(gt_path).convert('L')
      
    if(not dep_path):
        #empty
        dep_img=0
    else:
        dep_img = Image.open(dep_path)
        
    image=unloader(resultF_visual.squeeze(0)) 
    img_1=unloader(b1_res.squeeze(0))
    img_2=unloader(b2_res.squeeze(0))
    img_3=unloader(b3_res.squeeze(0))
    img_4=unloader(b4_res.squeeze(0))
    img_5=unloader(b5_res.squeeze(0))
    img_6=unloader(b6_res.squeeze(0))
    
    #data saving
    image.save(result_PATH)
    rgb_img.save(rgb_PATH)
    gt_img.save(gt_PATH)
    img_1.save(PATH_dir+"b1_result_%s.png"%(number_base))
    img_2.save(PATH_dir+"b2_result_%s.png"%(number_base))
    img_3.save(PATH_dir+"b3_result_%s.png"%(number_base))
    img_4.save(PATH_dir+"b4_result_%s.png"%(number_base))
    img_5.save(PATH_dir+"b5_result_%s.png"%(number_base))
    img_6.save(PATH_dir+"b6_result_%s.png"%(number_base))
    if (dep_path!=0):
      dep_img.save(depth_PATH)
    
    #refill value to sp map
    sp_segments=sp_segments.squeeze(0).numpy()
    #print(sp_segments)
    
    resultF_visual=np.array(image)   
    
    resultF_visual_1d=np.squeeze(resultF_visual)
    #print(resultF_visual_1d)
    #print(resultF_visual_1d.shape)
    
    result2d=np.zeros(sp_segments.shape)
    length=np.amax(sp_segments)+1
    
    # reconstruct
    for i in range(length):
        result2d[sp_segments==i]=resultF_visual_1d[i]
    #print(result2d)
    
    image_2d=Image.fromarray(result2d).convert('L')
    image_2d.save(PATH_dir+"2d_result_%s.png"%(number_base))
      