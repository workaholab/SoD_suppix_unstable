'''
#dataset processing reference 
https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py
https://sagivtech.com/2017/09/19/optimizing-pytorch-training-code/
'''
import torchvision.transforms as trns
import torch
import PIL
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import pandas as pd
from torch.utils import data
import numpy as np
from torchvision import transforms
import os
import glob

from skimage.segmentation import slic
from skimage import io, color

from slic import RGBD_SLICProcessor

from config import Config
cfg=Config()

mean = cfg.mean #[126.47384089, 124.27250705, 122.54440459, 160.28555339]
mean_train = cfg.mean_train #[112.13209312, 108.6547371, 101.00103511, 113.27608583]
mean_msra = cfg.mean_msra #[114.86545157, 110.46705426, 95.90594382]
mean_rgbtrain = cfg.mean_rgbtrain #[120.67928353, 114.42846415, 100.75286721]

mean_test = cfg.mean_test #[112.22424188, 108.63577088, 100.11004508, 126.90818425]
mean_test_NLPR = cfg.mean_test_NLPR #[123.70062814, 121.94022074, 120.47344181, 94.61725053]
mean_test_NJUD = cfg.mean_test_NJUD #[106.6313464, 102.52297522, 91.48286773, 119.98602273]
mean_LFSD = cfg.mean_LFSD #[128.65956412, 117.2438686, 107.97547353, 116.08585262]
mean_RGBD = cfg.mean_RGBD #[125.93331761, 121.60306597, 116.12658972, 151.30741493]
mean_PKU = cfg.mean_PKU #[106.99048091, 101.79961792, 91.27128536, 188.2142267]

DEBUG=False #self.DEBUG
reszie_sz=cfg.input_size

# ##################
#GPU device
# ##################
if torch.cuda.is_available():
    device = torch.device("cuda")
else: 
    device = torch.device("cpu")
print("Device info: torch.{}.is_available()".format(device))


#================================================================================
#================================================================================
#===============================PIXEL BASE DATASET===============================
#================================================================================
#================================================================================
class SalientObjDataset(data.Dataset):
    def __init__(self,STATE_TRTE,DATASET_TYPE,augmentation=None,debug=False):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.rgb_Dirpath=cfg.rgb_Dirpath
        self.dep_Dirpath=cfg.dep_Dirpath
        self.gt_Dirpath=cfg.gt_Dirpath

        if(STATE_TRTE): #training or testing
          if (DATASET_TYPE==1):
             #RGBD
             self.START_INDEX=1
             self.END_INDEX=2000
             # PRE-CALUCLATE THE DATASET MEAN
             self.meanR=mean_train[0]
             self.meanG=mean_train[1]
             self.meanB=mean_train[2]
             self.meanD=mean_train[3]
             
          else:
             #RGB
             self.START_INDEX=10001
             self.END_INDEX=30553 
             # PRE-CALUCLATE THE DATASET MEAN
             self.meanR=mean_rgbtrain[0]
             self.meanG=mean_rgbtrain[1]
             self.meanB=mean_rgbtrain[2]
             self.meanD=0

        else:  #Testing image index
          if DATASET_TYPE == 1:
              #self.NUM_OF_IMAGES = 500
              self.START_INDEX = 2001
              self.END_INDEX = 2501
              
              # PRE-CALUCLATE THE DATASET MEAN
              self.meanR=mean_test_NLPR[0]
              self.meanG=mean_test_NLPR[1]
              self.meanB=mean_test_NLPR[2]
              self.meanD=mean_test_NLPR[3]
              
          elif DATASET_TYPE == 2:
              #self.NUM_OF_IMAGES = 485
              self.START_INDEX = 2501
              self.END_INDEX = 2986
              
              # PRE-CALUCLATE THE DATASET MEAN
              self.meanR=mean_test_NJUD[0]
              self.meanG=mean_test_NJUD[1]
              self.meanB=mean_test_NJUD[2]
              self.meanD=mean_test_NJUD[3]
              
          elif DATASET_TYPE == 3:
              #self.NUM_OF_IMAGES = 100
              self.START_INDEX = 2986
              self.END_INDEX = 3086
              
              # PRE-CALUCLATE THE DATASET MEAN
              self.meanR=mean_LFSD[0]
              self.meanG=mean_LFSD[1]
              self.meanB=mean_LFSD[2]
              self.meanD=mean_LFSD[3]
              
          elif DATASET_TYPE == 4:
              #self.NUM_OF_IMAGES = 135
              self.START_INDEX = 3086
              self.END_INDEX = 3221
              
              # PRE-CALUCLATE THE DATASET MEAN
              self.meanR=mean_RGBD[0]
              self.meanG=mean_RGBD[1]
              self.meanB=mean_RGBD[2]
              self.meanD=mean_RGBD[3]
              
          elif DATASET_TYPE == 5:
              #self.NUM_OF_IMAGES = 80
              self.START_INDEX = 3221
              self.END_INDEX = 3301
              
              # PRE-CALUCLATE THE DATASET MEAN
              self.meanR=mean_PKU[0]
              self.meanG=mean_PKU[1]
              self.meanB=mean_PKU[2]
              self.meanD=mean_PKU[3]
              
              
        if (debug):
          self.START_INDEX = 1
          self.END_INDEX = 10
              
        self.NUM_OF_IMAGES = self.END_INDEX-self.START_INDEX+1
        
        # augmentation setting
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        """
        self.rgb_trans_compose = transforms.Compose([
            transforms.Resize(cfg.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.depth_trans_compose = transforms.Compose([
            transforms.Resize(cfg.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean = (self.meanD/255), std = (1/255))
        ])
        self.gt_trans_compose = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
      
        #index of file name
        file_index=self.START_INDEX+index
        rgb_filenm="%05d_rgb.*"%(file_index)
        dep_filenm="%05d_depth.*"%(file_index)
        gt_filenm="%05d_gt.*"%(file_index)
        
        rgb_path = glob.glob(os.path.join(self.rgb_Dirpath,rgb_filenm))[0]
        gt_path = glob.glob(os.path.join(self.gt_Dirpath,gt_filenm))[0]

        img = PIL.Image.open(rgb_path).convert('RGB')
        gt_img = PIL.Image.open(gt_path).convert('L')        
        img_data = self.rgb_trans_compose(img) 
        gt_data = self.gt_trans_compose(gt_img)
        
        # depth data
        if not len(glob.glob(os.path.join(self.dep_Dirpath,dep_filenm))): # is empty
          dep_path = 0
          dep_img = PIL.Image.new('L',gt_img.size) #torch.zeros(gt_data.shape) #transforms.ToPILImage()( np.zeros((gt_img.size[1],gt_img.size[0]),dtype=np.float32) )
          
          # superpixel label map
          #superpixel_maps=slic(io.imread(rgb_path))
          
        else:
          dep_path = glob.glob(os.path.join(self.dep_Dirpath,dep_filenm))[0]
          dep_img = PIL.Image.open(dep_path)
          
          # superpixel label map
          #superpixel_maps = RGBD_SLICProcessor(rgb_path,dep_path)
        
        depth_data = self.depth_trans_compose(dep_img)
        #sp_maps=self.gt_trans_compose(superpixel_maps) # superpixel label tensor
        img_size=img.size

        # tensor out
        return  img_data, depth_data , gt_data, rgb_path, gt_path, dep_path, img_size #, sp_maps

        
        
    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        
        return  self.NUM_OF_IMAGES #len(data)

#================================================================================
#================================================================================
#=============================== SuperpixelDataset ==============================
#================================================================================
#================================================================================

class SuperpixelDataset(data.Dataset):
    def __init__(self,STATE_TRTE,DATASET_TYPE,augmentation=None,debug=False):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.rgb_Dirpath=cfg.rgb_Dirpath
        self.dep_Dirpath=cfg.dep_Dirpath
        self.gt_Dirpath=cfg.gt_Dirpath

        if(STATE_TRTE): #training or testing
          if (DATASET_TYPE==1):
             #RGBD
             self.START_INDEX=1
             self.END_INDEX=2000
             # PRE-CALUCLATE THE DATASET MEAN
             self.meanR=mean_train[0]
             self.meanG=mean_train[1]
             self.meanB=mean_train[2]
             self.meanD=mean_train[3]
             
          else:
             #RGB
             self.START_INDEX=10001
             self.END_INDEX=30553 
             # PRE-CALUCLATE THE DATASET MEAN
             self.meanR=mean_rgbtrain[0]
             self.meanG=mean_rgbtrain[1]
             self.meanB=mean_rgbtrain[2]
             self.meanD=0

        else:  #Testing image index
          if DATASET_TYPE == 1:
              #self.NUM_OF_IMAGES = 500
              self.START_INDEX = 2001
              self.END_INDEX = 2501
              
              # PRE-CALUCLATE THE DATASET MEAN
              self.meanR=mean_test_NLPR[0]
              self.meanG=mean_test_NLPR[1]
              self.meanB=mean_test_NLPR[2]
              self.meanD=mean_test_NLPR[3]
              
          elif DATASET_TYPE == 2:
              #self.NUM_OF_IMAGES = 485
              self.START_INDEX = 2501
              self.END_INDEX = 2986
              
              # PRE-CALUCLATE THE DATASET MEAN
              self.meanR=mean_test_NJUD[0]
              self.meanG=mean_test_NJUD[1]
              self.meanB=mean_test_NJUD[2]
              self.meanD=mean_test_NJUD[3]
              
          elif DATASET_TYPE == 3:
              #self.NUM_OF_IMAGES = 100
              self.START_INDEX = 2986
              self.END_INDEX = 3086
              
              # PRE-CALUCLATE THE DATASET MEAN
              self.meanR=mean_LFSD[0]
              self.meanG=mean_LFSD[1]
              self.meanB=mean_LFSD[2]
              self.meanD=mean_LFSD[3]
              
          elif DATASET_TYPE == 4:
              #self.NUM_OF_IMAGES = 135
              self.START_INDEX = 3086
              self.END_INDEX = 3221
              
              # PRE-CALUCLATE THE DATASET MEAN
              self.meanR=mean_RGBD[0]
              self.meanG=mean_RGBD[1]
              self.meanB=mean_RGBD[2]
              self.meanD=mean_RGBD[3]
              
          elif DATASET_TYPE == 5:
              #self.NUM_OF_IMAGES = 80
              self.START_INDEX = 3221
              self.END_INDEX = 3301
              
              # PRE-CALUCLATE THE DATASET MEAN
              self.meanR=mean_PKU[0]
              self.meanG=mean_PKU[1]
              self.meanB=mean_PKU[2]
              self.meanD=mean_PKU[3]
              
              
        if (debug):
          self.START_INDEX = 1
          self.END_INDEX = 10
              
        self.NUM_OF_IMAGES = self.END_INDEX-self.START_INDEX+1
        
        # augmentation setting
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        """
        self.rgb_trans_compose = transforms.Compose([
            transforms.Resize(cfg.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.depth_trans_compose = transforms.Compose([
            transforms.Resize(cfg.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean = (self.meanD/255), std = (1/255))
        ])
        self.gt_trans_compose = transforms.Compose([
            transforms.ToTensor()
        ])

        
    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
      
        #index of file name
        file_index=self.START_INDEX+index
        rgb_filenm="%05d_rgb.*"%(file_index)
        dep_filenm="%05d_depth.*"%(file_index)
        gt_filenm="%05d_gt.*"%(file_index)
        
        rgb_path = glob.glob(os.path.join(self.rgb_Dirpath,rgb_filenm))[0]
        gt_path = glob.glob(os.path.join(self.gt_Dirpath,gt_filenm))[0]

        img = PIL.Image.open(rgb_path).convert('RGB')
        gt_img = PIL.Image.open(gt_path).convert('L')        
        img_data = self.rgb_trans_compose(img) 
        gt_data = self.gt_trans_compose(gt_img)
        
        # depth data
        if not len(glob.glob(os.path.join(self.dep_Dirpath,dep_filenm))): # is empty
          dep_path = 0
          dep_img = PIL.Image.new('L',gt_img.size) #torch.zeros(gt_data.shape) #transforms.ToPILImage()( np.zeros((gt_img.size[1],gt_img.size[0]),dtype=np.float32) )
          
          # superpixel label map
          superpixel_maps=slic(io.imread(rgb_path))
          
        else:
          dep_path = glob.glob(os.path.join(self.dep_Dirpath,dep_filenm))[0]
          dep_img = PIL.Image.open(dep_path)
          
          # superpixel label map
          superpixel_maps = RGBD_SLICProcessor(rgb_path,dep_path)
        
        depth_data = self.depth_trans_compose(dep_img)
        sp_maps=self.gt_trans_compose(superpixel_maps) # superpixel label tensor
        img_size=img.size

        # tensor out
        return  img_data, depth_data , gt_data, rgb_path, gt_path, dep_path, img_size, sp_maps
        
        
    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        
        return  self.NUM_OF_IMAGES #len(data)
        

        
#================================================================================
#================================================================================
#===================================1D PIXEL TENSOR=============================
#================================================================================
#================================================================================        
        
class SalientObjDataset_1D(data.Dataset):
    def __init__(self,STATE_TRTE,DATASET_TYPE,augmentation=None,debug=False):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.rgb_Dirpath="../RGBD_model_testing/data/rgb/"
        self.dep_Dirpath="../RGBD_model_testing/data/depth/"
        self.gt_Dirpath="../RGBD_model_testing/data/groundtruth/"

        if(STATE_TRTE): #training or testing
          if (DATASET_TYPE==1):
             #RGBD
             self.START_INDEX=1
             self.END_INDEX=2000
             
             # PRE-CALUCLATE THE DATASET MEAN
             self.meanR=mean_train[0]
             self.meanG=mean_train[1]
             self.meanB=mean_train[2]
             self.meanD=mean_train[3]
             
          else:
             #RGB
             self.START_INDEX=10001
             self.END_INDEX=30553 
             # PRE-CALUCLATE THE DATASET MEAN
             self.meanR=mean_rgbtrain[0]
             self.meanG=mean_rgbtrain[1]
             self.meanB=mean_rgbtrain[2]
             self.meanD=0

        else:  #Testing image index
          if DATASET_TYPE == 1:
              #self.NUM_OF_IMAGES = 500
              self.START_INDEX = 2001
              self.END_INDEX = 2501
              
              # PRE-CALUCLATE THE DATASET MEAN
              self.meanR=mean_test_NLPR[0]
              self.meanG=mean_test_NLPR[1]
              self.meanB=mean_test_NLPR[2]
              self.meanD=mean_test_NLPR[3]
              
          elif DATASET_TYPE == 2:
              #self.NUM_OF_IMAGES = 485
              self.START_INDEX = 2501
              self.END_INDEX = 2986
              
              # PRE-CALUCLATE THE DATASET MEAN
              self.meanR=mean_test_NJUD[0]
              self.meanG=mean_test_NJUD[1]
              self.meanB=mean_test_NJUD[2]
              self.meanD=mean_test_NJUD[3]
              
          elif DATASET_TYPE == 3:
              #self.NUM_OF_IMAGES = 100
              self.START_INDEX = 2986
              self.END_INDEX = 3086
              
              # PRE-CALUCLATE THE DATASET MEAN
              self.meanR=mean_LFSD[0]
              self.meanG=mean_LFSD[1]
              self.meanB=mean_LFSD[2]
              self.meanD=mean_LFSD[3]
              
          elif DATASET_TYPE == 4:
              #self.NUM_OF_IMAGES = 135
              self.START_INDEX = 3086
              self.END_INDEX = 3221
              
              # PRE-CALUCLATE THE DATASET MEAN
              self.meanR=mean_RGBD[0]
              self.meanG=mean_RGBD[1]
              self.meanB=mean_RGBD[2]
              self.meanD=mean_RGBD[3]
              
          elif DATASET_TYPE == 5:
              #self.NUM_OF_IMAGES = 80
              self.START_INDEX = 3221
              self.END_INDEX = 3301
              
              # PRE-CALUCLATE THE DATASET MEAN
              self.meanR=mean_PKU[0]
              self.meanG=mean_PKU[1]
              self.meanB=mean_PKU[2]
              self.meanD=mean_PKU[3]
              
              
        if (debug):
          self.START_INDEX = 1
          self.END_INDEX = 10
              
        self.NUM_OF_IMAGES = self.END_INDEX-self.START_INDEX+1
        
        # augmentation setting
        rgb_aug=[]
        depth_aug=[]
        gt_aug=[]
        augmentation = [transforms.ToTensor()] # , transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        # Resize, To tensor + preprocessing (tensor augmenAtation)
        rgb_aug = rgb_aug + [transforms.Resize(reszie_sz)] + augmentation + [transforms.Normalize(mean = (self.meanR/255,self.meanG/255,self.meanB/255), std = (1/255, 1/255, 1/255))]
        depth_aug = depth_aug + [transforms.Resize(reszie_sz)] + augmentation + [transforms.Normalize(mean = (self.meanD/255), std = (1/255))]
        gt_aug=gt_aug + [transforms.Resize(reszie_sz)] + augmentation

        self.rgb_trans_compose = transforms.Compose(rgb_aug)
        self.depth_trans_compose = transforms.Compose(depth_aug)
        self.gt_trans_compose = transforms.Compose(gt_aug)


    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
      
        #index of file name
        index=self.START_INDEX+index
        rgb_filenm="%05d_rgb.*"%(index)
        dep_filenm="%05d_depth.*"%(index)
        gt_filenm="%05d_gt.*"%(index)
        
        rgb_path = glob.glob(os.path.join(self.rgb_Dirpath,rgb_filenm))[0]
        gt_path = glob.glob(os.path.join(self.gt_Dirpath,gt_filenm))[0]

        img = PIL.Image.open(rgb_path).convert('RGB')
        gt_img = PIL.Image.open(gt_path).convert('L') 
        
        # print(img.size)
        
        # depth data
        if not len(glob.glob(os.path.join(self.dep_Dirpath,dep_filenm))): # is empty
          dep_path = 0
          dep_img = PIL.Image.new('L',gt_img.size) #torch.zeros(gt_data.shape) #transforms.ToPILImage()( np.zeros((gt_img.size[1],gt_img.size[0]),dtype=np.float32) )
        else:
          dep_path = glob.glob(os.path.join(self.dep_Dirpath,dep_filenm))[0]
          dep_img = PIL.Image.open(dep_path)
          
        # trans to Tensor 
        img_data = self.rgb_trans_compose(img) 
        gt_data = self.gt_trans_compose(gt_img) 
        depth_data = self.depth_trans_compose(dep_img)
        
        #original size
        img_size=img.size
          
        # 1-D rehape 
        '''
        img_data = torch.reshape(img_data, (3,img_data.shape[2]*img_data.shape[1]))
        depth_data = torch.reshape(depth_data, (1,depth_data.shape[2]*depth_data.shape[1]))
        gt_data = torch.reshape(gt_data, (1,gt_img.size[1]*gt_img.size[0]))
        '''
        sp_map=sp_map_Gen((gt_data.shape[2],gt_data.shape[1]))
        #1d data
        img_data, depth_data, gt_data=color_uniq_seq(img_data, depth_data, gt_data, sp_map)
        # numpy
        img=np.array(img)
        gt_img=np.array(gt_img)
        dep_img=np.array(dep_img)
        
        
        
        # original img / tensors/ path / size
        return  img, gt_img, dep_img, img_data, depth_data , gt_data, rgb_path, gt_path, dep_path, np.array(img_size)
        
        
    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        
        return  self.NUM_OF_IMAGES #len(data)
        
def sp_map_Gen(image_size):
    # print("func: sp_map_Gen")
    sp_map=np.zeros(image_size,dtype="int32")
    ind=0
    for i in range (image_size[0]):
      for j in range (image_size[1]):
        ind+=1
        sp_map[i,j]=ind
    
    #print("==sp_map==")  
    # print(sp_map)
    
    return sp_map
    
'''    
    Q_color = zeros(N,N,3);
    dist = zeros(N);
    
    for i = 1:N
        for j = 1:N
            [y_i, x_i] = ind2sub(im_size, P(i,1));
            p_i = [y_i; x_i];
            [y_j, x_j] = ind2sub(im_size, P(j,1));
            p_j = [y_j; x_j];
            dist(i,j) = norm(p_i - p_j);
            t_j = numel(label_idx{j});
            dist_weight = gaussian_weight(dist(i,j),0,dist_sigma);
            Q(i,j,1) = t_j*abs(C(i,1)-C(j,1))*gauss_weight*dist_weight;
            Q(i,j,2) = t_j*abs(C(i,2)-C(j,2))*gauss_weight*dist_weight;
            Q(i,j,3) = t_j*abs(C(i,3)-C(j,3))*gauss_weight*dist_weight;
        end
        [~,I] = sort(dist(i,:));
        Q_color(i,:,:) = Q(i,I,:);
    end
    
    all_Q(1,a) = {Q_color};

  save('all_Q.mat','all_Q');
'''

def color_uniq_seq(img_data, depth_data, gt_data, sp_map):
    # print("::func: color_uniq_seq")
    
    ret_rgb_data=torch.reshape(img_data, (3,640*640))
    ret_depth_data=torch.reshape(depth_data, (1,640*640))
    ret_gt_data=torch.reshape(gt_data, (1,640*640))
    
    return ret_rgb_data, ret_depth_data, ret_gt_data
    
    '''
    data_channel=len(input_img.split())
    # print(data_channel) # channel data RGB=3 , depth=1, GT=1
    
    img=input_img.load()
    tj= sp_map.shape[0]*sp_map.shape[1] # pixel nums
    print("::RGB") # RGB
    prx = [input_img.size[0]/2,input_img.size[1]/2] # positions
    crx = img[input_img.size[0]/2,input_img.size[1]/2] #avg pos color
    # seq of cu_seq
    qc=np.zeros((input_img.size[0]*input_img.size[1])) 
    print(qc.shape)
    ind=0
    for i in range(input_img.size[0]):
        for j in range(input_img.size[1]):
            prj=[i,j]
            crj = img[prj[0],prj[1]]
            cdif = np.abs([crx[0]-crj[0],crx[1]-crj[1],crx[2]-crj[2]]) # color dif
            
            print(cdif)
            print(gauss2dif(prx,prj))
            
            vec_col=tj*cdif*gauss2dif(prx,prj)
            
            print(vec_col.shape)
            #print(type(vec_col))
            qc[ind]=vec_col
            #qc[ind,1]=vec_col[1]
            #qc[ind,0,2]=vec_col[2]
            
            ind+=1


    print("::Depth / GT") # Depth / GT
    '''
      
      
from math import exp
def gauss2dif(a,b):
  sigma = 10
  mu=1
  X=np.linalg.norm([a[0]-b[0], a[1]-b[1]])
  return  exp((-1/(2*pow(sigma,2)))*pow((X-mu),2));
      
#================================================================================
#================================================================================
#===================================ROI POOL Feature=============================
#================================================================================
#================================================================================        
'''     
class ROI_pool_feature(data.Dataset):
    def __init__(self,STATE_TRTE,DATASET_TYPE,augmentation=None,debug=False):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.rgb_Dirpath="../RGBD_model_testing/data/rgb/"
        self.dep_Dirpath="../RGBD_model_testing/data/depth/"
        self.gt_Dirpath="../RGBD_model_testing/data/groundtruth/"

        if(STATE_TRTE): #training or testing
          if (DATASET_TYPE==1):
             #RGBD
             self.START_INDEX=1
             self.END_INDEX=2000
             
             # PRE-CALUCLATE THE DATASET MEAN
             self.meanR=mean_train[0]
             self.meanG=mean_train[1]
             self.meanB=mean_train[2]
             self.meanD=mean_train[3]
             
          else:
             #RGB
             self.START_INDEX=10001
             self.END_INDEX=30553 
             # PRE-CALUCLATE THE DATASET MEAN
             self.meanR=mean_rgbtrain[0]
             self.meanG=mean_rgbtrain[1]
             self.meanB=mean_rgbtrain[2]
             self.meanD=0

        else:  #Testing image index
          if DATASET_TYPE == 1:
              #self.NUM_OF_IMAGES = 500
              self.START_INDEX = 2001
              self.END_INDEX = 2501
              
              # PRE-CALUCLATE THE DATASET MEAN
              self.meanR=mean_test_NLPR[0]
              self.meanG=mean_test_NLPR[1]
              self.meanB=mean_test_NLPR[2]
              self.meanD=mean_test_NLPR[3]
              
          elif DATASET_TYPE == 2:
              #self.NUM_OF_IMAGES = 485
              self.START_INDEX = 2501
              self.END_INDEX = 2986
              
              # PRE-CALUCLATE THE DATASET MEAN
              self.meanR=mean_test_NJUD[0]
              self.meanG=mean_test_NJUD[1]
              self.meanB=mean_test_NJUD[2]
              self.meanD=mean_test_NJUD[3]
              
          elif DATASET_TYPE == 3:
              #self.NUM_OF_IMAGES = 100
              self.START_INDEX = 2986
              self.END_INDEX = 3086
              
              # PRE-CALUCLATE THE DATASET MEAN
              self.meanR=mean_LFSD[0]
              self.meanG=mean_LFSD[1]
              self.meanB=mean_LFSD[2]
              self.meanD=mean_LFSD[3]
              
          elif DATASET_TYPE == 4:
              #self.NUM_OF_IMAGES = 135
              self.START_INDEX = 3086
              self.END_INDEX = 3221
              
              # PRE-CALUCLATE THE DATASET MEAN
              self.meanR=mean_RGBD[0]
              self.meanG=mean_RGBD[1]
              self.meanB=mean_RGBD[2]
              self.meanD=mean_RGBD[3]
              
          elif DATASET_TYPE == 5:
              #self.NUM_OF_IMAGES = 80
              self.START_INDEX = 3221
              self.END_INDEX = 3301
              
              # PRE-CALUCLATE THE DATASET MEAN
              self.meanR=mean_PKU[0]
              self.meanG=mean_PKU[1]
              self.meanB=mean_PKU[2]
              self.meanD=mean_PKU[3]
              
              
        if (debug):
          self.START_INDEX = 1
          self.END_INDEX = 10
              
        self.NUM_OF_IMAGES = self.END_INDEX-self.START_INDEX+1

        # augmentation setting
        rgb_aug=[]
        depth_aug=[]
        gt_aug=[]
        augmentation = [transforms.ToTensor()] # , transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

        rgb_aug=rgb_aug +[transforms.Resize(reszie_sz)]
        depth_aug=depth_aug +[transforms.Resize(reszie_sz)]
        gt_aug=gt_aug          
        
        # To tensor + preprocessing (tensor augmenAtation)
        rgb_aug = rgb_aug + augmentation + [transforms.Normalize(mean = (self.meanR/255,self.meanG/255,self.meanB/255), std = (1/255, 1/255, 1/255))]
        depth_aug = depth_aug + augmentation + [transforms.Normalize(mean = (self.meanD/255), std = (1/255))]
        gt_aug=gt_aug+augmentation

        self.rgb_trans_compose = transforms.Compose(rgb_aug)
        self.depth_trans_compose = transforms.Compose(depth_aug)
        self.gt_trans_compose = transforms.Compose(gt_aug)


    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
      
        #index of file name
        index=self.START_INDEX+index
        rgb_filenm="%05d_rgb.*"%(index)
        dep_filenm="%05d_depth.*"%(index)
        gt_filenm="%05d_gt.*"%(index)
        
        rgb_path = glob.glob(os.path.join(self.rgb_Dirpath,rgb_filenm))[0]
        gt_path = glob.glob(os.path.join(self.gt_Dirpath,gt_filenm))[0]
        
        # RPN / ROI pooling
        
        ## read images
        image = skimage.io.imread(rgb_path)
        molded_images, image_metas, anchors=detect_Anchors(image)
        
        img = PIL.Image.open(rgb_path).convert('RGB')
        gt_img = PIL.Image.open(gt_path).convert('L')   
        # depth data
        if not len(glob.glob(os.path.join(self.dep_Dirpath,dep_filenm))): # is empty
          dep_path = 0
          dep_img = PIL.Image.new('L',gt_img.size) 
          #torch.zeros(gt_data.shape) #transforms.ToPILImage()( np.zeros((gt_img.size[1],gt_img.size[0]),dtype=np.float32) )
        else:
          dep_path = glob.glob(os.path.join(self.dep_Dirpath,dep_filenm))[0]
          dep_img = PIL.Image.open(dep_path)
          
        # trans to Tensor       
        img_data = self.rgb_trans_compose(img) 
        gt_data = self.gt_trans_compose(gt_img)
        depth_data = self.depth_trans_compose(dep_img)
        
        #original size
        img_size=img.size
        
        #Debug
        if(DEBUG):
          print(img_data.shape)
          print(depth_data.shape)
          print(gt_data.shape)
          print(rgb_path)
          print(gt_path)
          print(dep_path)
          print(img_size)

        
        return  img_data, depth_data , gt_data, rgb_path, gt_path, dep_path, np.array(img_size)

        
    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        
        return  self.NUM_OF_IMAGES #len(data)
        
    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.
        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows
        
    def detect_Anchors(self, image):
        """Runs the detection pipeline.
        images: List of images, potentially of different sizes.
        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
            
        return molded_images, image_metas, anchors
        
    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]
'''       
'''
class RPN_net(nn.Module):
    def __init__(self, anchors_per_location, gpu_device=torch.device("cpu")):
        super(RPN_net, self).__init__()
        self.device=gpu_device
        self.shared_conv=nn.Conv2d(3, 512, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.logits_conv=nn.Conv2d(512, anchors_per_location*2, 3, padding=1)
        self.linear1=nn.Linear()
        self.logits_conv=nn.Conv2d(512, anchors_per_location*2, 3, padding=1)
        
        self.softmax = nn.Softmax()
        

    def forward(self, feature_map, anchors_per_location, anchor_stride):  #### fcn_2d ####
        """Builds the computation graph of Region Proposal Network.
        feature_map: backbone features [batch, height, width, depth]
        anchors_per_location: number of anchors per pixel in the feature map
        anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                       every pixel in the feature map), or 2 (every other pixel).
        Returns:
            rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
            rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
            rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                      applied to anchors.
        """
        # TODO: check if stride of 2 causes alignment issues if the feature map
        # is not even.
        """
        # Shared convolutional base of the RPN
        shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                           strides=anchor_stride,
                           name='rpn_conv_shared')(feature_map)
    
        # Anchor Score. [batch, height, width, anchors per location * 2].
        x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                      activation='linear', name='rpn_class_raw')(shared)
    
        # Reshape to [batch, anchors, 2]
        rpn_class_logits = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)
    
        # Softmax on last dimension of BG/FG.
        rpn_probs = KL.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)
    
        # Bounding box refinement. [batch, H, W, anchors per location * depth]
        # where depth is [x, y, log(w), log(h)]
        x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",activation='linear', name='rpn_bbox_pred')(shared)
    
        # Reshape to [batch, anchors, 4]
        rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)
        """
        w1=torch.randn(2 * anchors_per_location, 512,1,1)
        w2=torch.randn(anchors_per_location * 4, 512,1,1)
        
        shared=F.relu(self.shared_conv(feature_map), inplace=True).to(device)
        x=F.linear(F.conv2d(shared, w1, padding=1), inplace=True).to(device)
        # Reshape to [batch, anchors, 2]
        rpn_class_logits = torch.reshape(x, (torch.shape(x)[0], -1, 2))
        # Softmax on last dimension of BG/FG.
        rpn_probs = F.softmax(rpn_class_logits)
        
        
        # Bounding box refinement. [batch, H, W, anchors per location * depth]
        # where depth is [x, y, log(w), log(h)]
        x = F.linear(F.conv2d(shared, w2, padding=1), inplace=True).to(device)
        # Reshape to [batch, anchors, 4]
        rpn_bbox = torch.reshape(t, (torch.shape(t)[0], -1, 4))
        
        
        return rpn_class_logits, rpn_probs, rpn_bbox
       
def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.
    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])
            
'''            
# ==============================================
# ==============================================
# ==============================================
# ==============================================
'''
from prefetch_generator import BackgroundGenerator
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__()) 

# construction test
if __name__ == '__main__':
    rgb_pretrain_dataset=SalientObjDataset_1D(1,0)
    pretrain_loader = DataLoaderX(rgb_pretrain_dataset, batch_size=1,shuffle=True)
    for i, data_record in enumerate(pretrain_loader):
        print(i)
        print("==========data_record===========")
        #print(data_record)
        # print(data_record.shape)
'''    
    
    