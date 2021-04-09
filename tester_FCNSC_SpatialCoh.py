# imports utils
from __future__ import print_function
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import math
from PIL import Image
from datetime import datetime
import time

from prefetch_generator import BackgroundGenerator
#pytorch
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# my modules
import dataset_generate
from dataset_generate import SalientObjDataset, SuperpixelDataset
from model_Net import Net, Net2, Net3, Net_interpolation, SPNet, SPNet_interpolation #DSS_parallel
from spatCoherence import Spco_Loss

# from visualization import visualize_results
# import dataloader_prefetch
# usage_mode: 1: train, 0: test
# input_mode: superpixels used or not
# set_type: 0: RGB / 1: RGBD
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from config import Config
cfg=Config()
#degub
DEBUG=cfg.DEBUG
# share by Train and Test
start_VerTrain=1 #0: no specific version need to be replaced
MODEL_SEL=0
#VERSION # updating
if(start_VerTrain==0): 
  Ver_Train=1
  while os.path.exists("modelParam_T%d/"%(Ver_Train)) or os.path.exists("VisualResults_T%d/"%(Ver_Train)):
        Ver_Train+=1 #don't cover the previous log
else:
  Ver_Train=start_VerTrain


# training set (define the starting point)
TRAIN_ON=True
start_EP=0
# testing set (end of model training parameters to use)
TEST_ON=False
epochs=118

#log file output
LogVER=1
Log_update=True

####### preload dataset #######
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
###############

# utilities ###############
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
       
###############################################################################
# Load / Save model parameters
def loadModelParam(model,PATH):
  print("==Load model parameters==")
  model.load_state_dict(torch.load(PATH)['model_state_dict'])

def saveModelParam(Ver_Train, epoch, model, optimizer, loss):
    print("==Saving model parameters==")
    #Save model parameters
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#save
    PATH_dir="modelParam_T%d/"%(Ver_Train)
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
print("(tester) Device info: torch.{}.is_available()".format(device))

################################################################################################
################################################################################################
################################################################################################
################################################################################################ 
# ##################
# main
# ##################
batch_size=1
if(TRAIN_ON):

  #::::::::::::::::::::
  # TRAINING
  #::::::::::::::::::::
  train_start=time.time()
  print("#::::::::::::::::::::")
  print("# TRAINING")
  print("#::::::::::::::::::::")
  #INPUT_TYPE=1 (pixel)
  input_mode=0
  state_train_test=1
  #datasets RGB pretrain / RGBD training
  '''
  INPUT_TYPE=0 #1=pixel, 0=superpixel
  DATASET_TYPE=1 #1=RGBD, 0=RGB
  state_train_test=1 #1=Train, 0=Test
  ''' 
  
  print("Train Version: %d"%(Ver_Train))
  if(start_EP!=0):
    print("Starting training epochs: %d"%(start_EP))
 
  if(Log_update):
    while os.path.exists("training_T%d_V%d.log"%(start_VerTrain,LogVER)):
      LogVER+=1 #don't cover the previous log  
    
  train_log_output="training_T%d_V%d_STATE%d.log"%(start_VerTrain,LogVER, state_train_test)
  f = open(train_log_output, "w")
  
  ##parameters init states## =====================================================================================
  init_learning_rate=1e-4
  init_weight_decay=0
  
  ##model initialization
  # old used model
    # DSS_parallel(device) #(delete)
    # Net(device,start_EP)
    # Net_interpolation(device,start_EP) #upsample is interpolation
  # decide model mode
  print("Select the model mode: %d"%(MODEL_SEL))
  f.write("Select the model mode: %d \n"%(MODEL_SEL))
  if(MODEL_SEL==0):
    model =  Net(device,start_EP)
    print("model class name: Net.")
    f.write("model class name: Net. \n")
  elif(MODEL_SEL==1):
    model =  Net_interpolation(device,start_EP)
    print("model class name: Net_interpolation.")
    f.write("model class name: Net_interpolation. \n")
  elif(MODEL_SEL==2):
    model = Net2(device,start_EP)
    print("model class name: Net2.")
    f.write("model class name: Net2. \n")
  elif(MODEL_SEL==3):
    model = Net3(device,start_EP)
    print("model class name: Net3.")
    f.write("model class name: Net3. \n")

  model.to(device)  
  
  if(start_VerTrain!=0 and start_EP!=0):
      if(os.path.exists("modelParam_T%d/"%(start_VerTrain))):
        loadModelParam(model,"modelParam_T%d/param_e%02d.pth"%(start_VerTrain,start_EP))
  
  # model structure # debug
  struc_chk=DEBUG
  if struc_chk:
    print("warning: |||Model graph|||")
    model_seq=torch.nn.Sequential(*list(model.children())[:])
    print(model_seq)
  
  
  if DEBUG:
    print("warning: Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor)
        #print(model.state_dict()[param_tensor]) #init parameters
        print(model.state_dict()[param_tensor].size()) #weights size
  
  
  
  if(MODEL_SEL>1):
    ##pretraining data 
    dataset_type=0
    rgb_pretrain_dataset=SuperpixelDataset(state_train_test,dataset_type) #Training/RGB dataset
    pretrain_loader = DataLoaderX(rgb_pretrain_dataset, batch_size=batch_size,shuffle=True)
    ##training data 
    dataset_type=1
    train_dataset = SuperpixelDataset(state_train_test,dataset_type) #Training/RGBD dataset
    train_loader = DataLoaderX(train_dataset, batch_size=batch_size,shuffle=True)    
  else:
    ##pretraining data 
    dataset_type=0
    rgb_pretrain_dataset=SalientObjDataset(state_train_test,dataset_type) #Training/RGB dataset
    pretrain_loader = DataLoaderX(rgb_pretrain_dataset, batch_size=batch_size,shuffle=True)
    ##training data 
    dataset_type=1
    train_dataset = SalientObjDataset(state_train_test,dataset_type) #Training/RGBD dataset
    train_loader = DataLoaderX(train_dataset, batch_size=batch_size,shuffle=True)
  
  '''
  # https://tianws.github.io/skill/2019/08/27/gpu-volatile/
  
  dataset_type=1 #RGBD as testing set
  loader=DataLoaderX(SalientObjDataset(state_train_test,dataset_type,debug=DEBUG), batch_size=batch_size,shuffle=False)
  '''
  
  #loss function
  # https://clay-atlas.com/blog/2020/05/22/pytorch-cn-error-solution-dimension-out-of-range/
  #==> https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
  
  # criterion = nn.BCEWithLogitsLoss() # change criterion
  criterion = Spco_Loss(device)
  optimizer = optim.Adam(model.parameters(), init_learning_rate)
  #optim.Adam(model.parameters(), lr=init_learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=init_weight_decay, amsgrad=False) # optim.SGD(model.parameters(), lr=init_learning_rate, momentum=0.9)
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
  
  #saved init parameters
  if(start_EP==0):
      PATH_dir="modelParam_T%d/"%(Ver_Train)
      if not os.path.exists(PATH_dir):
          os.makedirs(PATH_dir)  #create PATH_dir
      epoch=0
      PATH_file="param_e%02d.pth"%(epoch)
      PATH=PATH_dir+PATH_file
      torch.save({
                  'epoch': epoch,
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
          f.write("::::RGB dataset::::\n")
          # dataset_type=0
          loader=pretrain_loader       
        else: #RGBD
          print("::::RGBD dataset (Before e15)::::")
          f.write("::::RGBD dataset (Before e15)::::\n")
          # dataset_type=1
          loader=train_loader        
      else:
        print("::::RGBD dataset (After e15)::::")
        f.write("::::RGBD dataset (After e15)::::\n")
        dataset_type=1
        loader=train_loader 
      
      epoch_n=epoch+1
      
      print("::Currernt Epochs: %d" % (epoch_n))
      f.write("::Currernt Epochs: %d\n" % (epoch_n))
      running_loss = 0.0

      #--------------------------------------------
      for i, data_record in enumerate(loader):
          # data
          img_data=data_record[0]
          depth_data=data_record[1]
          gt_data=data_record[2]          
          # path
          img_name=data_record[3][0]
          gt_path=data_record[4][0] 
          dep_path=data_record[5][0]          
          # visualization
          im_size=data_record[6] # image size

          if(MODEL_SEL>1):
            sp_map=data_record[7]
          
          '''
          print("img_data",img_data.shape) 
          print("depth_data",depth_data.shape)
          '''
          # zero the parameter gradients
          optimizer.zero_grad()
          
          if(MODEL_SEL>1):
            result, b1_result, b2_result, b3_result, b4_result, b5_result, b6_result = model(img_data, depth_data, state_train_test, img_name, dep_path, sp_map)
          else:
            result, b1_result, b2_result, b3_result, b4_result, b5_result, b6_result = model(img_data, depth_data, state_train_test, img_name, dep_path)

          '''  
          print("result",result.shape) 
          print("b1_result",b1_result.shape) 
          print("b2_result",b2_result.shape) 
          print("b3_result",b3_result.shape) 
          print("b4_result",b4_result.shape) 
          print("b5_result",b5_result.shape) 
          print("b6_result",b6_result.shape) 
          '''

          ##################################
          # defined Loss function
          ##################################
          loss = criterion(epoch_n, result, b1_result, b2_result, b3_result, b4_result, b5_result, b6_result , gt_data, img_data, depth_data)
          # delete with torch.no_grad():  solve RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
          loss.backward() 
          # print("loss",loss)   
          optimizer.step()
  
          # print statistics
          loss_term=float(loss.item())
          running_loss += loss_term
          # assert math.isnan(loss_term), "!!! Loss value is nan !!! the error is at image: %s/ loss: %f"%(img_name, loss_term)
          # print("**loss: ", loss_term)
          if i % 2000 == 1999:    # print every 2000 mini-batches
              print('[%d, %5d] loss: %.4f' % (epoch + 1, i + 1, running_loss / 2000))
              f.write('[%d, %5d] loss: %.4f\n' % (epoch + 1, i + 1, running_loss / 2000))
              running_loss = 0.0    
              # PR curve/ F measure
              # https://pytorch.org/ignite/_modules/ignite/metrics/precision.html

          ###Visualization
          # output get sigmoid ########################
          sig_result=torch.sigmoid(result).to(device)
          sig_b1_result=torch.sigmoid(b1_result).to(device)
          sig_b2_result=torch.sigmoid(b2_result).to(device)
          sig_b3_result=torch.sigmoid(b3_result).to(device)
          sig_b4_result=torch.sigmoid(b4_result).to(device)
          sig_b5_result=torch.sigmoid(b5_result).to(device)
          sig_b6_result=torch.sigmoid(b6_result).to(device)
          sig_results=[sig_result, sig_b1_result, sig_b2_result, sig_b3_result, sig_b4_result, sig_b5_result, sig_b6_result]
          # visualization In the begining and last 
          if(epoch>epochs*0.8):
            visualize_results(state_train_test, Ver_Train, epoch_n, (im_size[1],im_size[0]), sig_results, img_name, gt_path, dep_path)  
       
      #Save model parameters
      # https://pytorch.org/tutorials/beginner/saving_loading_models.html#save
      saveModelParam(Ver_Train, epoch_n, model, optimizer, loss)
  
  print('Finished Training')
  train_end=time.time()
  f.write('Training Execution time: %f secs\n'%(train_end-train_start))
  f.close()

################################################################################################
################################################################################################
################################################################################################
################################################################################################

if(TEST_ON):
  STATE_TRTE=0
  #::::::::::::::::::::
  # TESTING
  #::::::::::::::::::::
  print("#::::::::::::::::::::")
  print("# TESTING")
  print("#::::::::::::::::::::")
  # TODO: testing
  # loading parameters
  epoch=40
  PATH_dir="modelParam_T%d/"%(Ver_Train)
  PATH_file="param_e%02d.pth"%(epoch)
  PATH=PATH_dir+PATH_file
   
  ##model initialization
  #model_test = Net(device,epoch) 
  # decide model mode equal (the trained model)
  if(MODEL_SEL==0):
    model_test =  Net(device,start_EP)
  elif(MODEL_SEL==1):
    model_test =  Net_interpolation(device,start_EP)
  elif(MODEL_SEL==2):
    model_test = Net2(device,start_EP)
  elif(MODEL_SEL==3):
    model_test = Net3(device,start_EP)
  model_test.to(device)
  print("Load model parameters, param file: %s"%(PATH))
  model_test.load_state_dict(torch.load(PATH)['model_state_dict'])

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
  #init parameters # reuse in forward() of model
  self.state_dict()['fusion_conv_inf.weight']= self.vgg16().state_dict()['fusion_conv.weight'] 
  self.state_dict()['fusion_conv_inf.bias']= self.vgg16().state_dict()['fusion_conv.bias'] 
  '''
  state_train_test=0
  if(Log_update):
    while os.path.exists("testing_T%d_V%d.log"%(start_VerTrain,LogVER)):
      LogVER+=1 #don't cover the previous log  
    
  test_log_output="testing_T%d_V%d_STATE%d.log"%(start_VerTrain,LogVER, state_train_test)
  f = open(test_log_output, "w")
  
  for dataset_type in range(1,4): #dataset_type=1 #SET: 1~3
  
    test_dataset = SalientObjDataset(state_train_test,dataset_type,debug=DEBUG) 
    test_loader = DataLoaderX(test_dataset, batch_size=batch_size)
    
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
        # test input data
        img_data=data_record[0]
        depth_data=data_record[1]
        gt_data=data_record[2]          
        # path
        img_name=data_record[3][0]
        gt_path=data_record[4][0] 
        dep_path=data_record[5][0]          
        # visualization
        im_size=data_record[6] # image size

        
        if(MODEL_SEL>1):
            sp_map=data_record[7]
            
        if(MODEL_SEL>1):
            result, b1_result, b2_result, b3_result, b4_result, b5_result, b6_result = model(img_data, depth_data, state_train_test, img_name, dep_path, sp_map)
        else:
            result, b1_result, b2_result, b3_result, b4_result, b5_result, b6_result = model(img_data, depth_data, state_train_test, img_name, dep_path)

        # results=[result, b1_result, b2_result, b3_result, b4_result, b5_result, b6_result]   
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
        visualize_results(state_train_test, Ver_Train, epoch, (im_size[1],im_size[0]), sig_results, img_name, gt_path, dep_path)  
        
        NUM_OF_IMAGES+=1
        #data to numpy
        '''
        # unloader = transforms.ToPILImage()
        # numpy.array(image)
        '''        
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
    f.write ("---------------------------------------------\n")
    f.write ("Dataset name: %s, epoch: %d ### \n" %( dataset_name[dataset_type-1], step))
    f.write ("Mean Absolute Error (result) : %.5f\n" % (r_MAE)) 
    f.write ("F measure : %.5f\n" % (r_a_f_measure))
    f.write ("---------------------------------------------\n")
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
