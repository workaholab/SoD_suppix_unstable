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
from dataset_generation import SPSalientObjDataset
from model_Net import SPNet
from spatCoherence import SP_Spco_Loss
#import dataloader_prefetch

# usage_mode: 1: train, 0: test
# input_mode: superpixels used or not
# set_type: 0: RGB / 1: RGBD

INPUT_TYPE=1 #0=pixel, 1=superpixel
DATASET_TYPE=1 #1=RGBD, 0=RGB

#degub
DEBUG=False
start_VerTrain=4
TRAIN_ON=False #True
start_EP=0
TEST_ON=True #False
epochs=40

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#****************************************************************VERSION
if(start_VerTrain==0):
    Ver_Train=1
    while os.path.exists("SP_modelParam_T%d/"%(Ver_Train)) or os.path.exists("SP_VisualResults_T%d/"%(Ver_Train)):
       Ver_Train+=1 #don't cover the previous log
else:
    Ver_Train=start_VerTrain

print("(superpixel) Train Version: %d"%(Ver_Train))

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())  

# ###############

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
    
    return result2d

    
    
###############################################################################
# Load / Save model parameters
def loadModelParam(model,PATH):
  print("==Load model parameters==")
  model.load_state_dict(torch.load(PATH)['model_state_dict'])


# -----

def saveModelParam(Ver_Train, epoch, model, optimizer, loss):
    print("==Saving model parameters==")
    #Save model parameters
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#save
    PATH_dir="SP_modelParam_T%d/"%(Ver_Train)
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
print("Device info: torch.{}.is_available()".format(device))


# ##################
# main
# ##################

#log file output
LogVER=1
Log_update=False

if(TRAIN_ON):
  start_t=time.time()
  train_SP_time="sp_trainTime_T%d_V%d_STATE%d.log"%(start_VerTrain, LogVER, 1) #state_train_test=1
  f = open(train_SP_time, "w")
  f.write('Epochs, Loss')


  STATE_TRTE=1
  #::::::::::::::::::::
  # TRAINING
  #::::::::::::::::::::
  print("#::::::::::::::::::::")
  print("# TRAINING")
  print("#::::::::::::::::::::")
  '''
  INPUT_TYPE=0 #1=pixel, 0=superpixel
  DATASET_TYPE=1 #1=RGBD, 0=RGB
  STATE_TRTE=1 #1=Train, 0=Test 
  '''
  ##parameters init states## =====================================================================================
  init_learning_rate=1e-3 #1e-4
  init_weight_decay=0
  
  ##model initialization
  model = SPNet(device,start_EP)
  model_seq=torch.nn.Sequential(*list(model.children())[:])
  model.to(device)  
    
  if(start_VerTrain!=0 and start_EP!=0):
    if(os.path.exists("SP_modelParam_T%d/"%(start_VerTrain))):
      loadModelParam(model,"SP_modelParam_T%d/param_e%02d.pth"%(start_VerTrain,start_EP-1))
      print("Load parameters , epoch: %d",start_EP-1 )
  
  # model structure
  if DEBUG:
    print("warning: |||Model graph|||")
    print(model_seq)
    print("warning: Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor)
        #print(model.state_dict()[param_tensor]) #init parameters
        print(model.state_dict()[param_tensor].size()) #weights size
        
  batch_size=1
  
  #INPUT_TYPE=1 (pixel)
  input_mode=0
  state_train_test=1
  #datasets RGB pretrain / RGBD training
  
  ##pretraining data
  dataset_type=0
  #rgb_pretrain_dataset=SPSalientObjDataset(state_train_test,dataset_type) #Training/RGB dataset
  rgb_pretrain_dataset=SPSalientObjDataset(state_train_test,dataset_type,debug=False,sp_update=False,create=False)
  pretrain_loader = DataLoaderX(rgb_pretrain_dataset, batch_size=batch_size,shuffle=True)
  
  ##training data 
  dataset_type=1
  # train_dataset = SPSalientObjDataset(state_train_test,dataset_type) #Training/RGBD dataset
  train_dataset = SPSalientObjDataset(state_train_test,dataset_type,debug=False,sp_update=False,create=False)
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
      PATH_dir="SP_modelParam_T%d/"%(Ver_Train)
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
          #print information
          # print(data_record.shape)
              
          # get the inputs; data is a list of  data_record = [img, depth, gt]
          #print("::DATA::")
          # rgb_path, img_data, depth_data , gt_data, img.size, gt_path, dep_path
          img_name=data_record[0][0]
          img_data=data_record[1]
          depth_data=data_record[2]
          gt_data=data_record[3]
          # visualization
          im_size=data_record[4] # image size
          gt_path=data_record[5][0] 
          dep_path=data_record[6][0]
          sp_seg=data_record[7]
          
          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          # visualize_test(STATE_TRTE, Ver_Train, -1, [img_data,depth_data,gt_data],i)  
          result, b1_result, b2_result, b3_result, b4_result, b5_result, b6_result = model(img_data,depth_data) #,dataset_type
          # output get sigmoid ########################
          sig_result=torch.sigmoid(result).to(device)
          sig_b1_result=torch.sigmoid(b1_result).to(device)
          sig_b2_result=torch.sigmoid(b2_result).to(device)
          sig_b3_result=torch.sigmoid(b3_result).to(device)
          sig_b4_result=torch.sigmoid(b4_result).to(device)
          sig_b5_result=torch.sigmoid(b5_result).to(device)
          sig_b6_result=torch.sigmoid(b6_result).to(device)
          sig_results=[sig_result, sig_b1_result, sig_b2_result, sig_b3_result, sig_b4_result, sig_b5_result, sig_b6_result]
          
          #if(epoch>=10):
          # p_results=[result, b1_result, b2_result, b3_result, b4_result, b5_result, b6_result]
          result2d_out=visualize_results_sp(STATE_TRTE, Ver_Train, epoch_n, (im_size[0]), sig_results, img_name, gt_path, dep_path, sp_seg) # sp visualization

          ##################################
          # defined Loss function
          ##################################
          # IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
          # https://discuss.pytorch.org/t/equivalent-of-tensorflows-sigmoid-cross-entropy-with-logits-in-pytorch/1985/7
          # https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelSoftMarginLoss.html
          # https://discuss.pytorch.org/t/loss-is-nan-for-crossentropyloss/74656
          
          loss = criterion(epoch_n, result, b1_result, b2_result, b3_result, b4_result, b5_result, b6_result , gt_data, img_data, depth_data, sp_seg, result2d_out) # to float type
          
          # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
          loss.backward()
          optimizer.step()
  
          # print statistics
          loss_term=float(loss.item())
          running_loss += loss_term
          # assert math.isnan(loss_term), "!!! Loss value is nan !!! the error is at image: %s/ loss: %f"%(img_name, loss_term)
          # print("**loss: ", loss_term)
          if i % 2000 == 1999:    # print every (2000)=>20 mini-batches
              '''
              print("============= fusion_conv, b1_conv3, b2_conv3, b3_conv3, b4_conv3, b5_conv3, b6_conv3 =============")
              print(result)
              print(b1_result)
              print(b2_result)
              print(b3_result)
              print(b4_result)
              print(b5_result)
              print(b6_result)
              print("============= fusion_conv, b1_conv3, b2_conv3, b3_conv3, b4_conv3, b5_conv3, b6_conv3 =============")          
              '''
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
###########################################################################################################################################
###########################################################################################################################################
if(TEST_ON):

  if(Log_update):
    while os.path.exists("testing_T%d_V%d.log"%(start_VerTrain,LogVER)):
      LogVER+=1 #don't cover the previous log 

  STATE_TRTE=0
  #::::::::::::::::::::
  # TESTING
  #::::::::::::::::::::
  print("#::::::::::::::::::::")
  print("# TESTING")
  print("#::::::::::::::::::::")
  input_mode=0 #pixels
  state_train_test=0
  dataset_type=1 #SET: 1~5
  
  batch_size=1
  
  # TODO: testing
  # loading parameters
  epoch=0
  PATH_dir="SP_modelParam_T%d/"%(Ver_Train)
  PATH_file="param_e%02d.pth"%(epoch)
  PATH=PATH_dir+PATH_file
  
  model_test =  SPNet(device,start_EP)
  model_test.to(device)  
  # degub
  '''
  if DEBUG:
    print("warning: |||Model (test) graph|||")
    model_Tseq=torch.nn.Sequential(*list(model_test.children())[:])
    print(model_Tseq)
    print("warning: Model's state_dict (test):")
    for param_tensor in model_test.state_dict():
        print(param_tensor)
        #print(model.state_dict()[param_tensor]) #init parameters
        print(model_test.state_dict()[param_tensor].size()) #weights size
  '''
  print("Load model parameters")
  model_test.load_state_dict(torch.load(PATH)['model_state_dict'])
  
  for dataset_type in range(1,4): #dataset_type=1 #SET: 1~3
    
    test_dataset = SPSalientObjDataset(state_train_test,dataset_type,debug=DEBUG,sp_update=False,create=False) 
    test_loader = DataLoaderX(test_dataset, batch_size=batch_size)
    
    test_log_output="SP_testing_T%d_V%d_STATE%d.log"%(start_VerTrain,LogVER, state_train_test)
    f = open(test_log_output, "w")
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
          # rgb_path, img_data, depth_data , gt_data, img.size, gt_path, dep_path
          
          # path and image Tensors
          rgb_path=data_record[0][0]
          img_data=data_record[1]
          depth_data=data_record[2]
          gt_data=data_record[3]  
          
          # image size
          im_size=data_record[4] 
          
          # image path (visualization results)
          gt_path=data_record[5][0]
          dep_path=data_record[6][0] 
          sp_seg=data_record[7]
          
    
          # output from training model (last epoch)
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
          result_val_2d=visualize_results_sp(STATE_TRTE, Ver_Train, epoch, (im_size[0]), sig_results, rgb_path, gt_path, dep_path, sp_seg)   
        
          #########################################
        
          NUM_OF_IMAGES+=1
          
          #data to numpy
          '''
          # unloader = transforms.ToPILImage()
          # numpy.array(image)
          '''
          """
          eval_result=(result-torch.min(result))/(torch.max(result)-torch.min(result)) # re-normalization
          result_resize=F.interpolate(eval_result,(gt_data.shape[2],gt_data.shape[3])).cpu() # result => sig_result
          # both range shall in [0,255]
          result_val=result_resize.cpu().detach().numpy()
          gt_val=gt_data.cpu().detach().numpy() * 255 
          #np.array(transforms.ToPILImage()(gt_data.squeeze(0))) 
          # gt_data.cpu().detach().numpy()
          # print(gt_val.shape) = (640, 480)
          
          #########################################################################################################
          # This code is going to calculate the maximum f-measure
          # print(result_val.shape)
          result_val = np.squeeze(result_val, axis=(0,1)) # range=[0,1]
          # reranged: to rerange from (0, 1) to (0, 255)
          #crf_result_val = np.squeeze(crf_result_val, axis=(0,))
          result_val_reranged = result_val * 255.0 #np.array(transforms.ToPILImage()(result_resize.squeeze(0))) #result_val * 255.0 # range=[0,255]
          #result_val_reranged = crf_result_val * 255.0
          
          
          
          '''
          print("result_val")
          print(result_val.shape)
          print(np.max(result_val))
          print("result_val_reranged")
          print(result_val_reranged.shape)
          print(np.max(result_val_reranged))
          '''
          gt_val = np.squeeze(gt_val, axis=(0, 1)) # range=[0,255]
          '''
          print("gt_val")
          print(gt_val.shape)
          print(np.max(gt_val)) 
          '''
          """
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
              '''
              print("TP")
              print(TruePositives)
              print("FP")
              print(FalsePositives)
              print("FN")
              print(FalseNegatives)
              '''
              
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
    print ("%s: %d[epoch]: Mean Absolute Error (result) : %.5f" % (datetime.now(), step, r_MAE)) # MAE
    print ("%s: %d[epoch]: Max F measure : %.5f" % (datetime.now(), step, max_f_measure))
    print ("Applying adaptive thresholds:")
    print ("%s: %d[epoch]: average Precision : %.5f" % (datetime.now(), step, r_a_precision))
    print ("%s: %d[epoch]: average Recall : %.5f" % (datetime.now(), step, r_a_recall))
    print ("%s: %d[epoch]: F measure : %.5f" % (datetime.now(), step, r_a_f_measure))
    # write file
    # for fast report
    f.write ("---------------------------------------------")
    f.write ("Dataset name: %s, epoch: %d ### \n" %( dataset_name[dataset_type-1], step))
    f.write ("Mean Absolute Error (result) : %.5f\n" % (r_MAE)) 
    f.write ("F measure : %.5f\n" % (r_a_f_measure))
    f.write ("---------------------------------------------")
    # for check details
    # f.write ("##### Details, dataset name: %s ##### \n"%(dataset_name[dataset_type-1]))
    f.write ("%s: %d[epoch]: Mean Absolute Error (result) : %.5f\n" % (datetime.now(), step, r_MAE)) 
    f.write ("%s: %d[epoch]: Max F measure : %.5f\n" % (datetime.now(), step, max_f_measure))
    f.write ("Applying adaptive thresholds:\n")
    f.write ("%s: %d[epoch]: average Precision : %.5f\n" % (datetime.now(), step, r_a_precision))
    f.write ("%s: %d[epoch]: average Recall : %.5f\n" % (datetime.now(), step, r_a_recall))
    f.write ("%s: %d[epoch]: F measure : %.5f\n" % (datetime.now(), step, r_a_f_measure))

  print('@Testing worked@')
  f.close()

