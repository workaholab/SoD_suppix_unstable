# SoD_suppix_unstable
SoD_suppix_unstable
Author: workaholab 



## 2021/3/25
This is the unstable verion of Salient object detection with srchitecture "Deeply Supervised Salient Object Detection with Short Connection"

tester*.py is the main execution file. Currently we have three versions.
ver1 is the original pixel-based 
ver 2 and ver 3 are the original DSS architecture with superpixel pooling (https://github.com/idealwei/SuperPixelPool.pytorch)

the current version we use is https://github.com/workaholab/suppixpool_pytorch (my exeacutable version)

Before use it, modify the config of data directory.


## 2021/3/30
remove " with torch.no_grad():" in tester_FCNSC_SpatialCoh.py (around ln: 248)

## 2021/4/09
1. fixed the issues 
   * dataset_generate.py: SalientObjDataset, SuperpixelDataset
   * model_Net.py: a conv2d error (fusion layer)

add Config: self.model_sel
current mode setting are: 
```
  if(MODEL_SEL==0):
    model =  Net(device,start_EP)
  elif(MODEL_SEL==1):
    model =  Net_interpolation(device,start_EP)
  elif(MODEL_SEL==2):
    model = Net2(device,start_EP)
  elif(MODEL_SEL==3)
    model = Net3(device,start_EP)
```

## 2021/3/30
model_Net:: Net2, Net3 <==> dataset::SalientObjDataset 
  1. add superpixel segmentation labels tensor
  2. forward() pass the superpixel label tensor (for superpixel poiling usage)


## Issues
### multi GPU
https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html#multi-gpu-examples

### Net correctin
model_Net:: SP_Net <==> dataset::SalientObjDataset or dataset::SuperpixelDataset

