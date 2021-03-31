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

from datetime import datetime
import time

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

# from dataset_generate import SalientObjDataset
from dataset_generate import SalientObjDataset_1D
from model_Net import SPNet, Sim_CNN
from spatCoherence import SP_Spco_Loss,SP1d_Spco_Loss

# usage_mode: 1: train, 0: test
# input_mode: superpixels used or not
# set_type: 0: RGB / 1: RGBD
INPUT_TYPE=1 #0=pixel, 1=superpixel
DATASET_TYPE=1 #1=RGBD, 0=RGB
DEBUG=False #degub
start_VerTrain=0
TRAIN_ON=True #True
start_EP=0
TEST_ON=True #False
epochs=1000

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__()) 

#****************************************************************VERSION
if(start_VerTrain==0):
    Ver_Train=1
    while os.path.exists("SP1d_modelParam_T%d/"%(Ver_Train)) or os.path.exists("SP1d_VisualResults_T%d/"%(Ver_Train)):
       Ver_Train+=1 #don't cover the previous log
else:
    Ver_Train=start_VerTrain

print("(1d pixels) Train Version: %d"%(Ver_Train))

# ###############

def visualize_results_sp1d(train_mode, Ver_Train, epochs, predict_results, rgb_path, gt_path, dep_path):
    
    unloader = transforms.ToPILImage()

    name_base=os.path.basename(rgb_path)
    number_base=name_base[1:len(name_base)-8]
    
    #result saving
    if (train_mode):
      PATH_dir="SP1d_VisualResults_T%d/Train/Epoch_%d/results_%s/"%(Ver_Train,epochs,number_base)
    else:
      PATH_dir="SP1d_VisualResults_T%d/Test/results_%s/"%(Ver_Train,number_base)
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
    
    resultF_visual=resultF_visual.detach().cpu()
    b1_res=b1_res.cpu().clone()
    b2_res=b2_res.cpu().clone()
    b3_res=b3_res.cpu().clone()
    b4_res=b4_res.cpu().clone()
    b5_res=b5_res.cpu().clone()
    b6_res=b6_res.cpu().clone()
    
    rgb_img = Image.open(rgb_path).convert('RGB')
    gt_img = Image.open(gt_path).convert('L')
    # original size
    img_size=gt_img.size
    img_size=(img_size[1],img_size[0])
      
    if(not dep_path): #empty
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
      
    '''
    #refill value to sp map (pixel)
    sp_segments=np.zeros((img_size[1],img_size[0])) 
    n=0
    for i in range(img_size[1]):
      for j in range(img_size[0]):
        n=n+1
        sp_segments[i,j] = n
    '''

    # resultF_visual=np.array(image)
    resultF_visual_1d=resultF_visual.numpy()
    # resultF_visual_1d=np.squeeze(resultF_visual_1d)
    # print(resultF_visual_1d)
    result2d=np.resize(resultF_visual_1d,img_size)
    '''
    print("===================================")
    print("resultF_visual_1d",resultF_visual_1d.shape)
    print("===================================")
    result2d=np.zeros((img_size[1],img_size[0]))
    length=np.amax(sp_segments)+1
    
    # reconstruct
    print("length of sp_segments",length)
    
    for i in range(img_size[1]):
      for j in range(img_size[0]):
        result2d[i,j]=resultF_visual_1d[i+img_size[0]*j]
    '''
    
    image_2d=Image.fromarray(result2d).convert('L')
    image_2d.save(PATH_dir+"2d_result_%s.png"%(number_base))
    
    return result2d

def visualize_results_sp1d_2(train_mode, Ver_Train, epochs, predict_results, rgb_path, gt_path, dep_path):
    
    unloader = transforms.ToPILImage()

    name_base=os.path.basename(rgb_path)
    number_base=name_base[1:len(name_base)-8]
    
    #result saving
    if (train_mode):
      PATH_dir="SP1d_VisualResults_T%d/Train/Epoch_%d/results_%s/"%(Ver_Train,epochs,number_base)
    else:
      PATH_dir="SP1d_VisualResults_T%d/Test/results_%s/"%(Ver_Train,number_base)
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
    resultF_visual=predict_results
    resultF_visual=resultF_visual.detach().cpu()
    
    rgb_img = Image.open(rgb_path).convert('RGB')
    gt_img = Image.open(gt_path).convert('L')
    # original size
    img_size=gt_img.size
    img_size=(img_size[1],img_size[0])
      
    if(not dep_path): #empty
        dep_img=0
    else:
        dep_img = Image.open(dep_path)
        
    image=unloader(resultF_visual.squeeze(0)) 

    
    #data saving
    image.save(result_PATH)
    rgb_img.save(rgb_PATH)
    gt_img.save(gt_PATH)

    if (dep_path!=0):
      dep_img.save(depth_PATH)
      
    '''
    #refill value to sp map (pixel)
    sp_segments=np.zeros((img_size[1],img_size[0])) 
    n=0
    for i in range(img_size[1]):
      for j in range(img_size[0]):
        n=n+1
        sp_segments[i,j] = n
    '''

    # resultF_visual=np.array(image)
    resultF_visual_1d=resultF_visual.numpy()
    # resultF_visual_1d=np.squeeze(resultF_visual_1d)
    # print(resultF_visual_1d)
    result2d=np.resize(resultF_visual_1d,img_size)
    '''
    print("===================================")
    print("resultF_visual_1d",resultF_visual_1d.shape)
    print("===================================")
    result2d=np.zeros((img_size[1],img_size[0]))
    length=np.amax(sp_segments)+1
    
    # reconstruct
    print("length of sp_segments",length)
    
    for i in range(img_size[1]):
      for j in range(img_size[0]):
        result2d[i,j]=resultF_visual_1d[i+img_size[0]*j]
    '''
    
    image_2d=Image.fromarray(result2d).convert('L')
    image_2d.save(PATH_dir+"2d_result_%s.png"%(number_base))
    
    return result2d    
    
###############################################################################
# Load / Save model parameters
def loadModelParam(model,PATH):
  print("==Load model parameters==")
  model.load_state_dict(torch.load(PATH)['model_state_dict'])

def saveModelParam(Ver_Train, epoch, model, optimizer, loss):
    print("==Saving model parameters==")
    #Save model parameters
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#save
    PATH_dir="SP1d_modelParam_T%d/"%(Ver_Train)
    if not os.path.exists(PATH_dir):
      os.makedirs(PATH_dir)  #create PATH_dir
        
    PATH_file="param_e%02d.pth"%(epoch)
      
    PATH=PATH_dir+PATH_file
      
    torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': loss,
          }, PATH)

# ##################
#GPU device
# ##################
if torch.cuda.is_available():
    device = torch.device("cuda")
else: 
    device = torch.device("cpu")

print("***")
print("Device info: torch.{}.is_available()".format(device))
print("***")


# ##################
# main
# ##################

#log file output
LogVER=1
Log_update=False

if(TRAIN_ON):
  start_t=time.time()
  train_SP_time="1dPixel_trainTime_T%d_V%d_STATE%d.log"%(start_VerTrain, LogVER, 1) #STATE_TRTE=1
  f = open(train_SP_time, "w")
  f.write('Epochs, Loss')
  #::::::::::::::::::::
  # TRAINING
  #::::::::::::::::::::
  print("#::::::::::::::::::::")
  print("# TRAINING")
  print("#::::::::::::::::::::")
  '''
  INPUT_TYPE=0 #1=pixel, 0=superpixel
  DATASET_TYPE=1 #1=RGBD, 0=RGB
  STATE_TRTE=STATE_TRTE=1 #1=Train, 0=Test 
  '''
  STATE_TRTE=1
  #INPUT_TYPE=1 (pixel)
  input_mode=0
  ##parameters init states## =====================================================================================
  batch_size=1
  init_learning_rate=1e-4 #1e-4
  init_weight_decay=0
  ##model initialization
  if(start_VerTrain%2==0):
    model = Sim_CNN(device)#SPNet(device,start_EP)
  else:
    model = SPNet(device,start_EP)
    
  model.to(device)  
    
  if(start_VerTrain!=0 and start_EP!=0): # this time tranining using previous train params
    if(os.path.exists("SP1d_modelParam_T%d/"%(start_VerTrain))):
      loadModelParam(model,"SP1d_modelParam_T%d/param_e%02d.pth"%(start_VerTrain,start_EP-1))
      print("Load parameters , epoch: %d"% (start_EP-1) )
  
  # model structure
  if DEBUG:
    print("warning: |||Model graph|||")
    model_seq=torch.nn.Sequential(*list(model.children())[:])
    print(model_seq)
    print("warning: Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor)
        #print(model.state_dict()[param_tensor]) #init parameters
        print(model.state_dict()[param_tensor].size()) #weights size
  
  #datasets RGB pretrain / RGBD training
  ##pretraining data
  dataset_type=0
  #rgb_pretrain_dataset=SPSalientObjDataset(STATE_TRTE,dataset_type) #Training/RGB dataset
  rgb_pretrain_dataset=SalientObjDataset_1D(STATE_TRTE,dataset_type)
  pretrain_loader = DataLoaderX(rgb_pretrain_dataset, batch_size=batch_size,shuffle=True)
  
  ##training data 
  dataset_type=1
  # train_dataset = SPSalientObjDataset(STATE_TRTE,dataset_type) #Training/RGBD dataset
  train_dataset = SalientObjDataset_1D(STATE_TRTE,dataset_type)
  train_loader = DataLoaderX(train_dataset, batch_size=batch_size,shuffle=True) # Debug used
  
  #loss function
  # https://clay-atlas.com/blog/2020/05/22/pytorch-cn-error-solution-dimension-out-of-range/
  #==> https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
  # criterion = nn.BCEWithLogitsLoss() # change criterion
  criterion = SP_Spco_Loss(device)
  optimizer = optim.Adam(model.parameters(), init_learning_rate)
  #optim.Adam(model.parameters(), lr=init_learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=init_weight_decay, amsgrad=False) # optim.SGD(model.parameters(), lr=init_learning_rate, momentum=0.9)
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
  
    #saved init parameters
  if(start_EP==0):
      PATH_dir="SP1d_modelParam_T%d/"%(Ver_Train)
      if not os.path.exists(PATH_dir):
          os.makedirs(PATH_dir)  #create PATH_dir
      epoch=0
      PATH_file="param_e%02d.pth"%(epoch)
      PATH=PATH_dir+PATH_file
      torch.save({'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  }, PATH) 
  
  # ##################
  # data prefetch
  # https://discuss.pytorch.org/t/how-to-prefetch-data-when-processing-with-gpu/548/17
  # ##################
  
  for epoch in range(start_EP, epochs):  # loop over the dataset multiple times
      # Debug used #################################
      if(epoch<16):
        if(epoch%2==0): #RGB
          print("::::RGB dataset::::")
          dataset_type=0
          loader=pretrain_loader       
        else: #RGBD
          print("::::RGBD dataset (Before e15)::::")
          dataset_type=1
          loader=train_loader        
      else:
        print("::::RGBD dataset (After e15)::::")
        dataset_type=1
        loader=train_loader 
      
      epoch_n=epoch+1
      print("::Currernt Epochs: %d" % (epoch_n))
      running_loss = 0.0
      
      #--------------------------------------------
      for i, data_record in enumerate(loader):
          # img
          img_img=data_record[0]
          depth_img=data_record[1]
          gt_img=data_record[2]         
          # data
          img_data=data_record[3]
          depth_data=data_record[4]
          gt_data=data_record[5]  
          # path
          img_name=data_record[6][0]
          gt_path=data_record[7][0] 
          dep_path=data_record[8][0]          
          # visualization
          im_size=data_record[9] # image size
          ######################### pixel seg ###########################
          # sp_seg => proceesing in visualize_results_sp1d
          # zero the parameter gradients
          optimizer.zero_grad()
          if(start_VerTrain%2==0):
            result  = model(img_data,depth_data,dataset_type) 
            c=SP1d_Spco_Loss(device)
            loss = c(epoch_n, result, gt_data, img_data, depth_data)
          else:
            result, b1_result, b2_result, b3_result, b4_result, b5_result, b6_result = model(img_data,depth_data, dataset_type) 
            # output get sigmoid ########################
            sig_result=torch.sigmoid(result).to(device)
            sig_b1_result=torch.sigmoid(b1_result).to(device)
            sig_b2_result=torch.sigmoid(b2_result).to(device)
            sig_b3_result=torch.sigmoid(b3_result).to(device)
            sig_b4_result=torch.sigmoid(b4_result).to(device)
            sig_b5_result=torch.sigmoid(b5_result).to(device)
            sig_b6_result=torch.sigmoid(b6_result).to(device)
            sig_results=[sig_result, sig_b1_result, sig_b2_result, sig_b3_result, sig_b4_result, sig_b5_result, sig_b6_result]
            result2d_out=visualize_results_sp1d(STATE_TRTE, Ver_Train, epoch_n, sig_results, img_name, gt_path, dep_path) # sp visualization         
            #if(epoch>=10):
            # p_results=[result, b1_result, b2_result, b3_result, b4_result, b5_result, b6_result]
            im_size=im_size.detach().numpy()
            # print("imsize: ",im_size)
            ##################################
            # defined Loss function
            ##################################
            #SP_Spco_Loss(device)
            loss = criterion(epoch_n, result, b1_result, b2_result, b3_result, b4_result, b5_result, b6_result , gt_data, img_data, depth_data, result2d_out) # to float type
            
            
          loss.backward()
          optimizer.step()
  
          # print statistics
          loss_term=float(loss.item())
          running_loss += loss_term
          # assert math.isnan(loss_term), "!!! Loss value is nan !!! the error is at image: %s/ loss: %f"%(img_name, loss_term)
          # print("**loss: ", loss_term)
          if i % 2000 == 1999:    # print
              print('%s: [%d, %5d] loss: %.4f' % (datetime.now(), epoch + 1, i + 1, running_loss / 2000))
              f.write('%d, %.4f' % (epoch + 1, running_loss / 2000)) # epochs, loss
              running_loss = 0.0    
              # PR curve/ F measure
              # https://pytorch.org/ignite/_modules/ignite/metrics/precision.html
              
      #Save model parameters
      # https://pytorch.org/tutorials/beginner/saving_loading_models.html#save
      saveModelParam(Ver_Train, epoch, model, optimizer, loss)
      
  print('Finished Training')
  
  end_t=time.time()
  print("Training execution %f secs" % (end_t-start_t) )
  f.write("execution %f secs" % (end_t-start_t))
  f.close()
  
###########################################################################################################################################
#TEST##########################################################################################################################################
###########################################################################################################################################
if(TEST_ON):

  if(Log_update):
    while os.path.exists("testing_T%d_V%d.log"%(start_VerTrain,LogVER)):
      LogVER+=1 #don't cover the previous log   
  #::::::::::::::::::::
  # TESTING
  #::::::::::::::::::::
  print("#::::::::::::::::::::")
  print("# TESTING")
  print("#::::::::::::::::::::")
  STATE_TRTE=0
  input_mode=0
  dataset_type=1 #SET: 1~5
  batch_size=1
  # TODO: testing
  # loading parameters
  epoch=39
  PATH_dir="SP1d_modelParam_T%d/"%(Ver_Train)
  PATH_file="param_e%02d.pth"%(epoch)
  PATH=PATH_dir+PATH_file
  
    ##model initialization
  if(start_VerTrain%2==0):
    model_test = Sim_CNN(device)#SPNet(device,start_EP)
  else:
    model_test = SPNet(device,start_EP)

  model_test.to(device)  
  # load train model
  print("Load model parameters")
  model_test.load_state_dict(torch.load(PATH)['model_state_dict'])
  
  # evaluated results logging
  test_log_output="SP1d_testing_T%d_V%d_STATE%d.log"%(start_VerTrain,LogVER, STATE_TRTE)
  f = open(test_log_output, "w")
  
  for dataset_type in range(1,4): #dataset_type=1 #SET: 1~3
    # load different datasets
    test_dataset =  SalientObjDataset_1D(STATE_TRTE,dataset_type)
    test_loader = DataLoaderX(test_dataset, batch_size=batch_size,shuffle=False) 
    
    #init params of f-measure
    NUM_OF_IMAGES=0
    index = 0
    r_MAE = 0
    r_f_measure = np.zeros(256)
    r_precision = np.zeros(256)
    r_recall = np.zeros(256)
    r_a_precision = 0
    r_a_recall = 0
    r_a_f_measure = 0
    
    # testing results
    for i, data_record in enumerate(test_loader):
          # img
          img_img=data_record[0]
          depth_img=data_record[1]
          gt_img=data_record[2]         
          # data
          img_data=data_record[3]
          depth_data=data_record[4]
          gt_data=data_record[5]  
          # path
          img_name=data_record[6][0]
          gt_path=data_record[7][0] 
          dep_path=data_record[8][0]          
          # visualization
          im_size=data_record[9] # image size
          # output from training model (last epoch)
          if(start_VerTrain%2==0):
            result  = model_test(img_data,depth_data,dataset_type) 
            result_val_2d=visualize_results_sp1d_2(STATE_TRTE, Ver_Train, 40, result, img_name, gt_path, dep_path)
          else:
            result, b1_result, b2_result, b3_result, b4_result, b5_result, b6_result  = model_test(img_data,depth_data,dataset_type) 
            # output get sigmoid ########################
            sig_result=torch.sigmoid(result).to(device)
            sig_b1_result=torch.sigmoid(b1_result).to(device)
            sig_b2_result=torch.sigmoid(b2_result).to(device)
            sig_b3_result=torch.sigmoid(b3_result).to(device)
            sig_b4_result=torch.sigmoid(b4_result).to(device)
            sig_b5_result=torch.sigmoid(b5_result).to(device)
            sig_b6_result=torch.sigmoid(b6_result).to(device)
            sig_results=[sig_result, sig_b1_result, sig_b2_result, sig_b3_result, sig_b4_result, sig_b5_result, sig_b6_result]       
          
            # visualization
            result_val_2d=visualize_results_sp1d(STATE_TRTE, Ver_Train, 40, sig_results, img_name, gt_path, dep_path)
          #visualize_results_sp1d(STATE_TRTE, Ver_Train, epoch, (im_size[0]), sig_results, rgb_path, gt_path, dep_path, sp_seg)   
        
          #########################################
        
          NUM_OF_IMAGES+=1
          
          gt_val=np.array(Image.open(gt_path).convert('L')) 
          result_val=  result_val_2d
          result_val_reranged=result_val_2d*255
          #print(gt_val.shape)
          #print(result_val_reranged.shape)
          
          for j in range(0, 256):
              result_thresholded = result_val_reranged >= float(j)
              result_thresholded = result_thresholded.astype(np.int32)
              gt_reranged = gt_val
                    
              # True positives are the salient pixels that your classifier judges them as salient
              TruePositives = np.count_nonzero(np.logical_and(result_thresholded, gt_reranged).astype(np.int32))
              # True positives are the salient pixels that your classifier judges them as salient
              FalsePositives = np.count_nonzero(result_thresholded) - np.count_nonzero(np.logical_and(result_thresholded, gt_reranged).astype(np.int32))
              # True positives are the salient pixels that your classifier judges them as salient
              FalseNegatives = np.count_nonzero(gt_reranged) - np.count_nonzero(np.logical_and(result_thresholded, gt_reranged).astype(np.int32))
              # Precision = TP/(TP+FP)
              if TruePositives == 0 and FalsePositives == 0:
                  precision = 0.0
              else:
                  precision = TruePositives/(TruePositives+FalsePositives)
              # Recall = TP/(TP+FN)
              if TruePositives == 0 and FalseNegatives == 0:
                  recall = 0.0
              else:
                  recall = TruePositives/(TruePositives+FalseNegatives)
                    
              r_precision[j] += precision
              r_recall[j] += recall
          #print(r_precision)
          #print(r_recall)
                
          # This code use adaptive thresholds to calculate average recall, average precision, and f-measure
          #result_val_reranged_tho = (result_val_reranged >= 0.95).astype(np.int32)
          result_a_thresholded = result_val_reranged >= float(2*np.mean(result_val_reranged))
          #result_a_thresholded = result_val_reranged >= float(225)
          result_a_thresholded = result_a_thresholded.astype(np.int32)
          #print ("adaptive threshold: %.5f" % (float(2*np.mean(result_val_reranged))))
          # True positives are the salient pixels that your classifier judges them as salient
          TruePositives = np.count_nonzero(np.logical_and(result_a_thresholded, gt_val).astype(np.int32))
          # True positives are the salient pixels that your classifier judges them as salient
          FalsePositives = np.count_nonzero(result_a_thresholded) - np.count_nonzero(np.logical_and(result_a_thresholded, gt_val).astype(np.int32))
          # True positives are the salient pixels that your classifier judges them as salient
          FalseNegatives = np.count_nonzero(gt_val) - np.count_nonzero(np.logical_and(result_a_thresholded, gt_val).astype(np.int32))
          # Precision = TP/(TP+FP)
          #print ("TP: %8d, FP: %8d, FN: %8d" % (TruePositives, FalsePositives, FalseNegatives))
          if TruePositives == 0 and FalsePositives == 0:
              precision = 0.0
          else:
              precision = TruePositives/(TruePositives+FalsePositives)
          # Recall = TP/(TP+FN)
          if TruePositives == 0 and FalseNegatives == 0:
              recall = 0.0
          else:
              recall = TruePositives/(TruePositives+FalseNegatives)
                
          r_a_precision += precision
          r_a_recall += recall
          if (precision==0.0) and (recall==0.0):
              r_a_f_measure += 0.0
          else:
              r_a_f_measure += 1.3*precision*recall/(0.3*precision+recall)
          # print(r_a_f_measure)
          
          r_MAE += np.mean(np.absolute(result_val-gt_val/255.0))
          #r_MAE += np.mean(np.absolute(crf_result_val-gt_val/255.0))
          
          
    r_MAE /= NUM_OF_IMAGES
    r_precision /= NUM_OF_IMAGES
    r_recall /= NUM_OF_IMAGES
    r_a_precision /= NUM_OF_IMAGES
    r_a_recall /= NUM_OF_IMAGES
    r_a_f_measure /= NUM_OF_IMAGES
            
    for j in range(0, 256):
        # f-measure = (1+beta^2)*precision*recall/(beta^2*precision+recall)
        if (r_precision[j] == 0.0) and (r_recall[j] == 0.0):
            r_f_measure[j] = 0.0
        else:
            r_f_measure[j] = 1.3*r_precision[j]*r_recall[j] /(0.3*r_precision[j]+r_recall[j])
                    
    max_f_measure = np.amax(r_f_measure)
    step=epoch
    dataset_name=['NLPR','NJUDS','LFSD','RGBD','PKU']
    # monitor
    print("### dataset name: %s ###"%(dataset_name[dataset_type-1]))
    print ("%s: %d[epoch]: Mean Absolute Error (result) : %.5f" % (datetime.now(), epoch, r_MAE)) # MAE
    print ("%s: %d[epoch]: Max F measure : %.5f" % (datetime.now(), epoch, max_f_measure))
    print ("Applying adaptive thresholds:")
    print ("%s: %d[epoch]: average Precision : %.5f" % (datetime.now(), epoch, r_a_precision))
    print ("%s: %d[epoch]: average Recall : %.5f" % (datetime.now(), epoch, r_a_recall))
    print ("%s: %d[epoch]: F measure : %.5f" % (datetime.now(), epoch, r_a_f_measure))
    # write file
    # for fast report
    f.write ("---------------------------------------------\n")
    f.write ("Dataset name: %s, epoch: %d ### \n" %( dataset_name[dataset_type-1], epoch))
    f.write ("Mean Absolute Error (result) : %.5f\n" % (r_MAE)) 
    f.write ("F measure : %.5f\n" % (r_a_f_measure))
    f.write ("---------------------------------------------\n")
    # for check details
    # f.write ("##### Details, dataset name: %s ##### \n"%(dataset_name[dataset_type-1]))
    f.write ("%s: %d[epoch]: Mean Absolute Error (result) : %.5f\n" % (datetime.now(), epoch, r_MAE)) 
    f.write ("%s: %d[epoch]: Max F measure : %.5f\n" % (datetime.now(), epoch, max_f_measure))
    f.write ("Applying adaptive thresholds:\n")
    f.write ("%s: %d[epoch]: average Precision : %.5f\n" % (datetime.now(), epoch, r_a_precision))
    f.write ("%s: %d[epoch]: average Recall : %.5f\n" % (datetime.now(), epoch, r_a_recall))
    f.write ("%s: %d[epoch]: F measure : %.5f\n" % (datetime.now(), epoch, r_a_f_measure))

  print('@Testing worked@')
  f.close()

