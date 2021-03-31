from __future__ import print_function 
# import chainer
# import chainer.functions as F
# import chainer.links as L
# import util.nameDef as names
import numpy as np
import chainer
import math
# from util.utilsGPU import read_code, load_kernel

import torch
import torchvision.transforms as trns
from torchvision import transforms

from scipy import ndimage
from numba import jit
import sys
from os import path
import os
import glob
import PIL
from PIL import Image

from skimage.segmentation import slic
from skimage.io import imread
from slic import RGBD_SLICProcessor, SLICProcessor
from config import Config
cfg=Config()

class Superpixel_Pool():
  
  def __init__(self):
      # params
      '''
      self.batchsize=batchsize     
      self.height=img_height     
      self.width=img_width  
      self.input_channel=input_channel     
      self.output_channel=output_channel 
      # self.pool_input=[]
      # weights of kernel
      # self.X=torch.randn((batchsize,input_channel,xSize,ySize), dtype=torch.float32, device=GPU)                
      '''
  
  # def max_pool(self, region):
    
  def spx_pooling(self, input_tensor, spx, gt):
      print("::Max pooling::")
      pool_input=in_tensor # [batch_size, nchannels, x, y]

      # batch = in_tensor.shape[0] #self.batchsize
      # in_channel = in_tensor.shape[1] #self.input_channel
      pooled_out=[]
      
      # segmentation labels
      np_img = input_tensor.numpy()
      # print(np_img.shape)
      self.spx=spx 
      self.K = spx.max()+1
      K = self.K
      # gt
      self.gt=gt 
      # print(np_img.shape)
      
      self.in_channel = np_img.shape[0]
      self.imWidth = imWidth =  np_img.shape[2]
      self.imHeight =imHeight = np_img.shape[1]
      # out_channel = 16
      
      # blocks (?)
      '''
      x = threadIdx.x*threadW 
      y = threadIdx.y*threadH
      
      imgStartIdx = batch*out_channel*imWidth*imHeight+
                          in_channel*imWidth*imHeight+
                          y*imWidth+
                          x
    
      labelStartIdx = batch*imWidth*imHeight +
                        y*imWidth+
                        x
      '''
      # print(K)                  
      for labels in range(K):
          # outIndex=out_channel*K + K*in_channel + labels #batch*out_channel*K
          # print("outIndex",outIndex)
          cur_idx = (spx==labels) # region for label
          out_data=np.zeros((self.in_channel, K))
          for ch in range(self.in_channel):
              data=np_img[ch,:,:]
              out_data[ch,:]=np.max(data[cur_idx])
              # print(data[cur_idx])
              # print(np.max(data[cur_idx]))
      return out_data #dim = (3 (ch), K)
          
      
      '''                   
       if (x < imWidth && y < imHeight && channel < nClasses && batch < batchSize)
        {
            imgIndex = imgStartIdx;
            labelIndex = labelStartIdx;
            label;
            outIndex;
            for (int idY=0; idY < threadH; idY++)
            {
                imgIndex = imgStartIdx + idY*imWidth;
                labelIndex = labelStartIdx + idY*imWidth;
                if (y+idY < imHeight)
                {
                    for (int idX=0; idX<threadW; idX++)
                    {
                        if (x + idX < imWidth){
                            label = labels[labelIndex];
                            outIndex = batch*nClasses*K + K*channel + label;
                            # atomicMaxIndex(&outIdx[outIndex], imgIndex, image);
                            imgIndex += 1;
                            labelIndex += 1; 
                        }
                        else{break;}
                    }
                }else{break;}
            }
        }
       '''
    
  def superpixel_unpool(self, pooled, spx):

    channel=self.in_channel
    # spx=spx
    K = spx.max()+1
    width=self.imWidth
    height=self.imHeight
    
    unpool_data=np.zeros((channel,width,height))
    
    channels=pooled.shape[0]
    
    # print(spx)
    print(pooled)
    
    #for labels in range(K):
    #cur_idx = (spx==labels) # region for label
    for i in range(height):
      for j in range(width):
        for ch in range(channels): 
            ref_data=pooled[ch, :]
            ref_label=spx[i,j]
            # print(ref_label)
            unpool_data[ch, i, j ] = ref_data[ref_label]
            
            
    return unpool_data
    
    
    '''
    int batch = blockIdx.z*blockDim.z + threadIdx.z;
    int label = blockIdx.x*blockDim.x + threadIdx.x;
    int channel = blockIdx.y*blockDim.y + threadIdx.y;
    int lstIdx = batch*nClasses*K+ channel*K + label;
    int lstSize = batchSize*nClasses*K;
    if (lstIdx<lstSize)
    {       
        int imgIndex = indices[lstIdx];
        if (imgIndex >= 0)
        {
            grad_in[imgIndex] = grad_outputs[lstIdx];
        }
    }
    '''


'''
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm> //min
#include <math.h> // ceil
#include <stdio.h>

// -------
// KERNELS 
// ------- 

template <typename scalar_t> 
__global__
void spx_max_pooling_forward_kernel(
    const scalar_t* __restrict__ image,
    const int* __restrict__ labels,
    int* outIdx,
    const int imWidth,
    const int imHeight, 
    const int threadW, 
    const int threadH,
    const int nClasses, 
    const int batchSize,
    const int K)
{   
    // extern __shared__ int sharedMem[];

    int batch = blockIdx.y;
    int channel = blockIdx.x;
    int x = threadIdx.x*threadW; 
    int y = threadIdx.y*threadH; 
    int imgStartIdx = batch*nClasses*imWidth*imHeight+
                      channel*imWidth*imHeight+
                      y*imWidth+
                      x;
    
    int labelStartIdx = batch*imWidth*imHeight +
                        y*imWidth+
                        x; 

    if (x < imWidth && y < imHeight && channel < nClasses && batch < batchSize)
    {
        int imgIndex = imgStartIdx;
        int labelIndex = labelStartIdx;
        int label;
        int outIndex;
        // int runningIdx;

        for (int idY=0; idY < threadH; idY++)
        {
            imgIndex = imgStartIdx + idY*imWidth;
            labelIndex = labelStartIdx + idY*imWidth;
            if (y+idY < imHeight)
            {
                for (int idX=0; idX<threadW; idX++)
                {
                    if (x + idX < imWidth){
                        label = labels[labelIndex];
                        outIndex = batch*nClasses*K + K*channel + label;
                        atomicMaxIndex(&outIdx[outIndex], imgIndex, image);
                        imgIndex += 1;
                        labelIndex += 1; 
                    }
                    else{break;}
                }
            }else{break;}
        }
    }
}



std::vector<at::Tensor> suppixpool_max_cuda_forward(
    at::Tensor img,
    at::Tensor spx_labels,
    at::Tensor output,
    at::Tensor outIdx,
    const int K)   
{
    /* 
    Shape assumptions: 
    - image: [nBatch, nChannel, x, y]
    - spx_labels: [nBatch, x, y]
    */

    const int batch_size = img.size(0);
    const int channels_size = img.size(1);

    const int imW = img.size(3);
    const int imH = img.size(2); 
    // const int nPixels = img.size(2)*img.size(3);

    int blockSizeX = std::min(32, imW);
    const int threadW    = ceil(imW/(float)blockSizeX);

    int blockSizeY = std::min(32, imH);
    const int threadH    = ceil(imH/(float)blockSizeY);

    // const int nbPixPerThread = ceil(nPixels/((float)blockSize));

    const dim3 blocks(channels_size, batch_size);
    const dim3 threads(blockSizeX, blockSizeY);

    AT_DISPATCH_FLOATING_TYPES(img.type(), "spx_max_pooling_forward_cuda", ([&] {
    spx_max_pooling_forward_kernel<scalar_t><<<blocks, threads>>>(
        img.data<scalar_t>(),
        spx_labels.data<int>(),
        outIdx.data<int>(),
        imW,
        imH,
        threadW,
        threadH, 
        channels_size, 
        batch_size,
        K);
    }));

    // fill in values at max positions (second kernel)
    blockSizeX = 16;
    blockSizeY = 1024/blockSizeX;
    const int nbBlocksX = ceil(channels_size/((float)blockSizeX));
    const int nbBlocksY = ceil(K/((float)blockSizeY));
    const dim3 blocksFill(nbBlocksX, nbBlocksY, batch_size);
    const dim3 threadsFill(blockSizeX, blockSizeY);

    AT_DISPATCH_FLOATING_TYPES(img.type(), "fill_max_values", ([&] {
    fill_values<scalar_t><<<blocksFill, threadsFill>>>(
        img.data<scalar_t>(),
        outIdx.data<int>(),
        K, 
        channels_size, 
        batch_size, 
        output.data<scalar_t>()
        );
    }));
    return {output, outIdx};
}
'''

if __name__ == '__main__':

    rgb_Dirpath=cfg.rgb_Dirpath
    dep_Dirpath=cfg.dep_Dirpath
    gt_Dirpath=cfg.gt_Dirpath 
    
    file_index=10001
    rgb_filenm="%05d_rgb.*"%(file_index)
    dep_filenm="%05d_depth.*"%(file_index)
    gt_filenm="%05d_gt.*"%(file_index)
       
    img_path=glob.glob(os.path.join(rgb_Dirpath,rgb_filenm))[0]
    print(img_path)
    gt_path=glob.glob(os.path.join(gt_Dirpath,gt_filenm))[0]
    print(gt_path)
    
    spx=slic(imread(img_path)) # SLICProcessor(img_path)
    img=PIL.Image.open(img_path).convert('RGB')   
    gt=PIL.Image.open(gt_path).convert('L')
    img_width, img_height = img.size
    
    
    trans_compose = transforms.Compose([transforms.ToTensor()])
    in_tensor = trans_compose(img)
    
    suppool_obj=Superpixel_Pool()
    pooled_out=suppool_obj.spx_pooling(in_tensor, spx, gt)
    unpooled_out=suppool_obj.superpixel_unpool(pooled_out,spx)
