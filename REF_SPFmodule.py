import tensorflow as tf
from tensorflow.python.platform import gfile
import os
import sys
import numpy as np
import glob
import csv
import math
from PIL import Image
import matplotlib.pyplot as plt

import scipy.io as sio
from scipy import signal

from skimage.color import rgb2gray, rgb2lab
from skimage.filters import sobel
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float	
from skimage import io
from skimage.measure import regionprops



DEBUG_PRINT=False

# Mean Color ([R, G, B, D])
mean = [126.47384089, 124.27250705, 122.54440459, 160.28555339]
# Mean Color ([R, G, B])
mean_msra = [114.86545157, 110.46705426, 95.90594382]

# dataset = tf.data.Dataset.from_tensor_slices((images, labels))

class SPFmodule:

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def csv_dataRead(self, train_csv_file, opt_dataset):
        #opt_dataset = 0, RGB; 1, RGBD
        with open(train_csv_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                qx_fm_list=[]
                gt_seq_list=[]
                if(opt_dataset):
                  dep_seq_list=[]
                # print(line_count)
                #if(line_count == 0):
                    #print(row)
                    #print(len(row))
                # row size
                rgb_path=row[0]
                gt_path=row[2]

                if(opt_dataset):
                  depth_path=row[1]
                  img_dp=io.imread(depth_path)
                  
                  #print(rgb_path,"/",depth_path,"/",gt_path)
                
                # print(rgb_path)
                image=io.imread(rgb_path)
                gt=io.imread(gt_path)
                
                
                #superpixel segmentation
                sp_segment = self.superpixel_segmentation(image)
                
                #feature transform
                if(opt_dataset):
                  print("=RGBD")
                  qc_featuremap, gt_seq, dep_seq = self.hcf_feature_transform_rgbd(sp_segment, image, gt, img_dp)
                  dep_seq_list.append(dep_seq)

                else:
                  print("=RGB")
                  qc_featuremap, gt_seq = self.hcf_feature_transform(sp_segment, image, gt)
                  qx_fm_list.append(qc_featuremap)
                  gt_seq_list.append(gt_seq)

                line_count += 1
                
                if(line_count==3):
                  break
                
                
            if(opt_dataset):
                # rgbd=tf.convert_to_tensor(qx_fm_list, dtype=tf.dtypes.float32, name='img_spf')
                #rgbd_dataset = tf.data.Dataset.from_tensor_slices(qx_fm_list) 
                #rgbd_gt_dataset = tf.data.Dataset.from_tensor_slices(gt_seq_list)   
                #rgbd_dp_dataset = tf.data.Dataset.from_tensor_slices(dep_seq_list)   
                rgbd_dataset = tf.data.Dataset.from_tensor_slices((qx_fm_list, gt_seq_list, dep_seq_list ))               
                # print(rgbd_dataset)
                return rgbd_dataset #, rgbd_gt_dataset, rgbd_dp_dataset
                  
            else:
                #rgb_dataset = tf.data.Dataset.from_tensor_slices(qx_fm_list)
                #rgb_gt_dataset =  tf.data.Dataset.from_tensor_slices(gt_seq_list)
                rgb_dataset = tf.data.Dataset.from_tensor_slices(( qx_fm_list, gt_seq_list ))
                # print(rgb_dataset)
                return rgb_dataset #, rgb_gt_dataset
            

    def superpixel_segmentation(self, image_read): # segmetnation label
        # prepared for SLIC => lab
        #lab_data=rgb2lab(image_read)
        l_data= image_read#lab_data[:,:,0]
        
        # segmentation algo (switchable)#######################################
        segments = slic(l_data, n_segments=200, compactness=10, sigma=1) #SLIC
        #######################################################################
        """
        if (line_count==1 and DEBUG_PRINT):
            print("segments_slic: ",np.asarray(segments).shape)
            # print(l_data)
            # print(l_data[0])
            print("lab: ", np.asarray(lab_data).shape)
            print("l: ",np.asarray(l_data).shape)
            print(image.shape[:2])

        if(DEBUG_PRINT):
          fig = plt.figure("Mask %d" % (line_count))
          ax = fig.add_subplot(1, 1, 1)
          ax.imshow(mask)
          plt.show()


        '''
        for (i, segVal) in enumerate(np.unique(segments)):
            print(i)
            print(segVal)
            #print ("[%d] inspecting segment %d" % (i, segVal))
            #mask = np.zeros(image.shape[:2], dtype = "uint8")
            #mask[segments == segVal] = 255
        '''

        """        
        return segments
        
        
    ###################### dataset construction ######################
    
    
    ###################### dataset construction ######################
            
    def hcf_feature_transform(self, L, image_color, image_gt):
        # self.input_segments=input_segments
        #print(image_gt.shape)
        # image_gt=image_gt/255 #normalized 
        # Vectorize superpixels in R and make mean color vector for each segmented regions (rx)
        N=np.amax(L)+1
        # L=input_segments
        print('********superpixel segment number: ', N)
        #im_size = io.imread(path_img).size;
        color_seq = np.zeros((N,3))
        gt_seq=np.zeros((1,N,1))
        # dep_seq=np.zeros((N,1))
        pos_seg = np.zeros((N,2))
        tr_seq = np.zeros((N))
        #r_val=im_image[:,:,0]
        #g_val=im_image[:,:,1]
        #b_val=im_image[:,:,2]
        
        # full round of data collection
        regions = regionprops(L)
        for props in regions: #for i in range(1,N): #color data
            i=props.label# label index
            # color mean remove
            image_color[:,:,0]=image_color[:,:,0]-mean[0]
            image_color[:,:,1]=image_color[:,:,1]-mean[1]
            image_color[:,:,2]=image_color[:,:,2]-mean[2]

            # color features
            red_spi_value=np.mean(image_color[(np.where(L==i,1,0)==1),0])
            green_spi_value=np.mean(image_color[(np.where(L==i,1,0)==1),1])
            blue_spi_value=np.mean(image_color[(np.where(L==i,1,0)==1),2])
            gt_seq[0,i,0]=np.mean(image_gt[(np.where(L==i,1,0)==1)])
            # dep_seq[i]=np.mean(image_dp[(np.where(L==i,1,0)==1)])
            # position
            mean_centers = np.round(np.array(np.mean(np.nonzero(L==i),axis=1)))
            
            # data 
            tr_seq[i]=props.area 
            color_seq[i,:]=[red_spi_value, green_spi_value, blue_spi_value]
            pos_seg[i,:]=mean_centers #[mean_centers[0],mean_centers[1]];
        # print(gt_seq)
        
        # Color Uniqueness Sequence
        qc = np.zeros((N,N,3))
        #qc_featuremap = np.zeros((1,N,N*3))
        qc_featuremap = np.zeros((1,N,3))
        for i in range(0,N):
            # j is the distance sorted index of superpixel 
            for j in range(0,N):        
                '''
                print(tr_seq[i]) 
                print(gaussian_wt_pos( pos_seg[i] , pos_seg[j_others]))     
                print(color_abs(color_seq[i] , color_seq[j_others] ) ) 
                print(tr_seq[i]  * gaussian_wt_pos( pos_seg[i] , pos_seg[j_others]) * color_abs(color_seq[i] , color_seq[j_others] )) 
                '''
                qc[i,j,:]=  tr_seq[i]  * gaussian_wt_pos( pos_seg[i] , pos_seg[j]) * color_abs(color_seq[i] , color_seq[j] ) # R
                #qc_featuremap[0,i,j]=qc[i,j,0]
                #qc_featuremap[0,i,(j+N)]=qc[i,j,1]
                #qc_featuremap[0,i,(j+2*N)]=qc[i,j,2]
                
            qc_featuremap[0,i,0]=np.mean(qc[i,:,0])
            qc_featuremap[0,i,1]=np.mean(qc[i,:,1])
            qc_featuremap[0,i,2]=np.mean(qc[i,:,2])    

        #print(qc_featuremap)
        # t_1d_spFeature=tf.convert_to_tensor(qc_featuremap, dtype=tf.dtypes.float32, name='img_spf')  #sp features, shape=(N,N*3)
        # t_1d_gt=tf.convert_to_tensor(gt_seq, dtype=tf.dtypes.float32, name='gt_spf') #gt
        # print(t_1d_spFeature)
        # print(t_1d_gt)
        
        return qc_featuremap, gt_seq #t_1d_spFeature, t_1d_gt #return qc_featuremap, gt_seq, dep_seq

        
    def hcf_feature_transform_rgbd(self, L, image_color, image_gt, image_dp):
        # self.input_segments=input_segments
        #print(image_gt.shape)
        # image_gt=image_gt/255 #normalized 
        # Vectorize superpixels in R and make mean color vector for each segmented regions (rx)
        N=np.amax(L)+1
        # L=input_segments
        print('********superpixel segment number: ', N)
        #im_size = io.imread(path_img).size;
        color_seq = np.zeros((N,3))
        gt_seq=np.zeros((1,N,1))
        dep_seq=np.zeros((1,N,1))
        pos_seg = np.zeros((N,2))
        tr_seq = np.zeros((N))
        #r_val=im_image[:,:,0]
        #g_val=im_image[:,:,1]
        #b_val=im_image[:,:,2]
        # mean remove
        image_color[:,:,0]=image_color[:,:,0]-mean[0]
        image_color[:,:,1]=image_color[:,:,1]-mean[1]
        image_color[:,:,2]=image_color[:,:,2]-mean[2]
        # image_dp[:,:,0]=image_dp[:,:,0]
        
        # full round of data collection
        regions = regionprops(L)
        for props in regions: #for i in range(1,N): #color data
            i=props.label# label index
            # ttl_r=props.area # total pixels
            # color
            red_spi_value=np.mean(image_color[(np.where(L==i,1,0)==1),0])
            green_spi_value=np.mean(image_color[(np.where(L==i,1,0)==1),1])
            blue_spi_value=np.mean(image_color[(np.where(L==i,1,0)==1),2])
            gt_seq[0,i,0]=np.mean(image_gt[(np.where(L==i,1,0)==1)])
            dep_seq[0,i,0]=np.mean(image_dp[(np.where(L==i,1,0)==1)])
            # position
            mean_centers = np.round(np.array(np.mean(np.nonzero(L==i),axis=1)))
            
            # data 
            tr_seq[i]=props.area 
            color_seq[i,:]=[red_spi_value, green_spi_value, blue_spi_value]
            pos_seg[i,:]=mean_centers #[mean_centers[0],mean_centers[1]];
        # print(gt_seq)
        
        # Color Uniqueness Sequence
        qc = np.zeros((N,N,3))
        #qc_featuremap = np.zeros((1,N,N*3))
        qc_featuremap = np.zeros((1,N,3))
        for i in range(0,N):
            # j is the distance sorted index of superpixel
             
            for j in range(0,N):        
                '''
                print(tr_seq[i]) 
                print(gaussian_wt_pos( pos_seg[i] , pos_seg[j_others]))     
                print(color_abs(color_seq[i] , color_seq[j_others] ) ) 
                print(tr_seq[i]  * gaussian_wt_pos( pos_seg[i] , pos_seg[j_others]) * color_abs(color_seq[i] , color_seq[j_others] )) 
                '''
                qc[i,j,:]=  tr_seq[i]  * gaussian_wt_pos( pos_seg[i] , pos_seg[j]) * color_abs(color_seq[i] , color_seq[j] ) # R
                '''
                qc_featuremap[0,i,j]=qc[i,j,0]
                qc_featuremap[0,i,(j+N)]=qc[i,j,1]
                qc_featuremap[0,i,(j+2*N)]=qc[i,j,2]
                '''
            qc_featuremap[0,i,0]=np.mean(qc[i,:,0])
            qc_featuremap[0,i,1]=np.mean(qc[i,:,1])
            qc_featuremap[0,i,2]=np.mean(qc[i,:,2])
            
        #print(qc_featuremap)
        '''
        t_1d_spFeature=tf.convert_to_tensor(qc_featuremap, dtype=tf.dtypes.float32, name='img_spf')  #sp features, shape=(N,N*3)
        t_1d_gt=tf.convert_to_tensor(gt_seq, dtype=tf.dtypes.float32, name='gt_spf') #gt
        t_ld_dp=tf.convert_to_tensor(dep_seq, dtype=tf.dtypes.float32, name='dp_spf')
        print(t_1d_spFeature)
        print(t_1d_gt)
        print(t_ld_dp)
        '''
        
        
        return qc_featuremap, gt_seq, dep_seq #t_1d_spFeature, t_1d_gt, t_ld_dp
        
        ########################################################################################################################
    def hcf_feature_transform_rmd(self, L, image_color, image_gt):
        # self.input_segments=input_segments
        #print(image_gt.shape)
        # image_gt=image_gt/255 #normalized 
        # Vectorize superpixels in R and make mean color vector for each segmented regions (rx)
        N=np.amax(L)+1
        # L=input_segments
        # print('********superpixel segment number: ', N)
        #im_size = io.imread(path_img).size;
        color_seq = np.zeros((N,3))
        gt_seq=np.zeros((N))
        # dep_seq=np.zeros((N,1))
        pos_seg = np.zeros((N,2))
        tr_seq = np.zeros((N))
        #r_val=im_image[:,:,0]
        #g_val=im_image[:,:,1]
        #b_val=im_image[:,:,2]
        
        # full round of data collection
        regions = regionprops(L)
        for props in regions: #for i in range(1,N): #color data
            i=props.label# label index
            # color mean remove
            image_color[:,:,0]=image_color[:,:,0]-mean[0]
            image_color[:,:,1]=image_color[:,:,1]-mean[1]
            image_color[:,:,2]=image_color[:,:,2]-mean[2]

            # color features
            red_spi_value=np.mean(image_color[(np.where(L==i,1,0)==1),0])
            green_spi_value=np.mean(image_color[(np.where(L==i,1,0)==1),1])
            blue_spi_value=np.mean(image_color[(np.where(L==i,1,0)==1),2])
            gt_seq[i]=np.mean(image_gt[(np.where(L==i,1,0)==1)])
            # dep_seq[i]=np.mean(image_dp[(np.where(L==i,1,0)==1)])
            # position
            mean_centers = np.round(np.array(np.mean(np.nonzero(L==i),axis=1)))
            
            # data 
            tr_seq[i]=props.area 
            color_seq[i,:]=[red_spi_value, green_spi_value, blue_spi_value]
            pos_seg[i,:]=mean_centers #[mean_centers[0],mean_centers[1]];
        # print(gt_seq)
        
        # Color Uniqueness Sequence
        qc = np.zeros((N,N,3))
        #qc_featuremap = np.zeros((1,N,N*3))
        qc_featuremap = np.zeros((N,3))
        for i in range(0,N):
            # j is the distance sorted index of superpixel 
            for j in range(0,N):        
                '''
                print(tr_seq[i]) 
                print(gaussian_wt_pos( pos_seg[i] , pos_seg[j_others]))     
                print(color_abs(color_seq[i] , color_seq[j_others] ) ) 
                print(tr_seq[i]  * gaussian_wt_pos( pos_seg[i] , pos_seg[j_others]) * color_abs(color_seq[i] , color_seq[j_others] )) 
                '''
                qc[i,j,:]=  tr_seq[i]  * gaussian_wt_pos( pos_seg[i] , pos_seg[j]) * color_abs(color_seq[i] , color_seq[j] ) # R
                #qc_featuremap[0,i,j]=qc[i,j,0]
                #qc_featuremap[0,i,(j+N)]=qc[i,j,1]
                #qc_featuremap[0,i,(j+2*N)]=qc[i,j,2]
                
            qc_featuremap[i,0]=np.mean(qc[i,:,0])
            qc_featuremap[i,1]=np.mean(qc[i,:,1])
            qc_featuremap[i,2]=np.mean(qc[i,:,2])    

        #print(qc_featuremap)
        # t_1d_spFeature=tf.convert_to_tensor(qc_featuremap, dtype=tf.dtypes.float32, name='img_spf')  #sp features, shape=(N,N*3)
        # t_1d_gt=tf.convert_to_tensor(gt_seq, dtype=tf.dtypes.float32, name='gt_spf') #gt
        # print(t_1d_spFeature)
        # print(t_1d_gt)
        
        return qc_featuremap, gt_seq #t_1d_spFeature, t_1d_gt #return qc_featuremap, gt_seq, dep_seq

        
    def hcf_feature_transform_rgbd_rmd(self, L, image_color, image_gt, image_dp):
        # self.input_segments=input_segments
        #print(image_gt.shape)
        # image_gt=image_gt/255 #normalized 
        # Vectorize superpixels in R and make mean color vector for each segmented regions (rx)
        N=np.amax(L)+1
        # L=input_segments
        # print('********superpixel segment number: ', N)
        #im_size = io.imread(path_img).size;
        color_seq = np.zeros((N,3))
        gt_seq=np.zeros((N))
        dep_seq=np.zeros((N))
        pos_seg = np.zeros((N,2))
        tr_seq = np.zeros((N))
        #r_val=im_image[:,:,0]
        #g_val=im_image[:,:,1]
        #b_val=im_image[:,:,2]
        # mean remove
        image_color[:,:,0]=image_color[:,:,0]-mean[0]
        image_color[:,:,1]=image_color[:,:,1]-mean[1]
        image_color[:,:,2]=image_color[:,:,2]-mean[2]
        # image_dp[:,:,0]=image_dp[:,:,0]
        
        # full round of data collection
        regions = regionprops(L)
        for props in regions: #for i in range(1,N): #color data
            i=props.label# label index
            # ttl_r=props.area # total pixels
            # color
            red_spi_value=np.mean(image_color[(np.where(L==i,1,0)==1),0])
            green_spi_value=np.mean(image_color[(np.where(L==i,1,0)==1),1])
            blue_spi_value=np.mean(image_color[(np.where(L==i,1,0)==1),2])
            gt_seq[i]=np.mean(image_gt[(np.where(L==i,1,0)==1)])
            dep_seq[i]=np.mean(image_dp[(np.where(L==i,1,0)==1)])
            # position
            mean_centers = np.round(np.array(np.mean(np.nonzero(L==i),axis=1)))
            
            # data 
            tr_seq[i]=props.area 
            color_seq[i,:]=[red_spi_value, green_spi_value, blue_spi_value]
            pos_seg[i,:]=mean_centers #[mean_centers[0],mean_centers[1]];
        # print(gt_seq)
        
        # Color Uniqueness Sequence
        qc = np.zeros((N,N,3))
        #qc_featuremap = np.zeros((1,N,N*3))
        qc_featuremap = np.zeros((N,3))
        for i in range(0,N):
            # j is the distance sorted index of superpixel
             
            for j in range(0,N):        
                '''
                print(tr_seq[i]) 
                print(gaussian_wt_pos( pos_seg[i] , pos_seg[j_others]))     
                print(color_abs(color_seq[i] , color_seq[j_others] ) ) 
                print(tr_seq[i]  * gaussian_wt_pos( pos_seg[i] , pos_seg[j_others]) * color_abs(color_seq[i] , color_seq[j_others] )) 
                '''
                qc[i,j,:]=  tr_seq[i]  * gaussian_wt_pos( pos_seg[i] , pos_seg[j]) * color_abs(color_seq[i] , color_seq[j] ) # R
                '''
                qc_featuremap[0,i,j]=qc[i,j,0]
                qc_featuremap[0,i,(j+N)]=qc[i,j,1]
                qc_featuremap[0,i,(j+2*N)]=qc[i,j,2]
                '''
            qc_featuremap[i,0]=np.mean(qc[i,:,0])
            qc_featuremap[i,1]=np.mean(qc[i,:,1])
            qc_featuremap[i,2]=np.mean(qc[i,:,2])
            
        #print(qc_featuremap)
        '''
        t_1d_spFeature=tf.convert_to_tensor(qc_featuremap, dtype=tf.dtypes.float32, name='img_spf')  #sp features, shape=(N,N*3)
        t_1d_gt=tf.convert_to_tensor(gt_seq, dtype=tf.dtypes.float32, name='gt_spf') #gt
        t_ld_dp=tf.convert_to_tensor(dep_seq, dtype=tf.dtypes.float32, name='dp_spf')
        print(t_1d_spFeature)
        print(t_1d_gt)
        print(t_ld_dp)
        '''
        
        
        return qc_featuremap, gt_seq, dep_seq #t_1d_spFeature, t_1d_gt, t_ld_dp
              

###
# self define methods
###
def gaussian_wt_pos(P_ri, P_rj):
    sigma=10
    dist_rij=np.linalg.norm(P_ri-P_rj)
    # print("dist:",dist_rij)
    w_pos= math.exp((-1/(2*sigma*sigma))*((dist_rij)*(dist_rij)));
    return w_pos
    
def color_abs(c_ri, c_rj):
    c_abs=c_ri-c_rj
    return c_abs   

"""
def image_preproc():
    '''
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
    	help="path to input image")
    ap.add_argument("-s", "--segments", type=int, default=100,
    	help="# of superpixels")
    args = vars(ap.parse_args())
    '''
    if not gfile.Exists(SP_DIR):
        gfile.MakeDirs(SP_DIR)
        
    rgb_list=glob.glob("data/rgb/*.jpg")
    # print(rgb_list)
    
    #CSV of superpixel position 
                
    
    #save the segment label
    for rgb_ls in rgb_list:
        fileNAME=str(os.path.splitext(os.path.basename(rgb_ls))[0])
        # print(fileNAME)
        image = io.imread(rgb_ls)
        [im_height, im_width]=[image.shape[0],image.shape[1]]
        #print(im_height, im_width)        
        #************************************************************************
        # segmentation module
        segments = slic(img_as_float(image), n_segments=(64*64), slic_zero=True)+1

        #print(segments)
        n_seg=len(np.unique(segments))
        len_map=math.floor( n_seg**(1/2) )
        size_map=len_map**2
        
        #n_seg =(reduce to)=> len_map superpixel merge
        '''
        #references
        # https://www.researchgate.net/figure/Superpixel-Merging-Algorithm_fig9_311445466
        '''
        
        print("SLIC number of segments: %d" % (n_seg)) 
        #print(len_map)     
        #************************************************************************

        # save SP label
        save_fileNAME=''.join([SP_DIR, '/spLabel_', fileNAME, '.csv'])
        # print(save_fileNAME)
        # np.save(save_fileNAME, segments)
        with open(save_fileNAME, 'w', newline='') as csvfile:
              writer = csv.writer(csvfile)
              writer.writerows(segments)    
        

        # row unique <++++++++++++++++++++++++++++++++++++++++  
        segments_rowuni=lines_uniq(segments)
        segments_rowuni=np.array(segments_rowuni,ndmin=1)
        print(segments_rowuni)
        '''
        for row_idx in range(im_height):
            segments_line=np.unique(segments[row_idx,:])
            #print(segments_line)
            segments_rowuni.append(segments_line)
            
        segments_rowuni=np.array(segments_rowuni,ndmin=2)
        # print(segments_rowuni)
        
        # write file to check the content
        with open('test', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(segments_rowuni)
        '''
        
        # col unique <++++++++++++++++++++++++++++++++++++++++ 
        segments_rowuni=segments_rowuni.transpose()
        # write file to check the content
        with open('test_trp', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(segments_rowuni)
            
            
        segments_uni=lines_uniq(segments_rowuni)
        '''
        segments_uni=[]
        for col_idx in range(im_width):
            segments_col=np.unique(segments_rowuni[col_idx])
            segments_uni.append(segments_col)
        
        segments_uni=segments_uni.transpose()    
        # print(segments_uni)
        '''
        
        '''
        save_fileNAME=''.join([SP_DIR, '/spLabel_unique_', fileNAME, '.csv'])
        # print(save_fileNAME)
        # np.save(save_fileNAME, segments)
        with open(save_fileNAME, 'w', newline='') as csvfile:
              segments_uni=np.unique(segments)
              writer = csv.writer(csvfile)
              writer.writerows(segments_uni)              
        '''
        

        # locate the position        
        regions = regionprops(segments)
        ## sorting sp position
        list_data=[]
        sorted_data=[]
        # (A) label sort before         
        for props in regions:
            y0, x0 = props.centroid
            y0=format(y0, '.4f')
            x0=format(x0, '.4f')                        
            # [label, x0, y0]
            label_rec=(props.label, float(x0), float(y0)) #use tuple
            list_data.append(label_rec)
        
        
        # (B) label sorting file  
        csv_spLabel=''.join([SP_DIR, '/spLabelPOS_', fileNAME, 'splabel_pos.csv'])               
        with open(csv_spLabel, 'w', newline='') as csvfile:
              writer = csv.writer(csvfile)
              writer.writerows(list_data)                                        
                        
        # (C) label sort After 
        # print(list_data) Two index min
        sort_x_list=sorted(list_data, key=lambda list_data:[list_data[1],list_data[2]]) # sort by x, y
                    
        #print(list_data[1])        
        #print(sort_x_list[1])
        
        
        # (D) label sorted file  
        csv_spLabel=''.join([SP_DIR, '/spLabelPOS_', fileNAME, 'sorted_splabel_pos.csv'])               
        with open(csv_spLabel, 'w', newline='') as csvfile:
              writer = csv.writer(csvfile)
              writer.writerows(sort_x_list)
              
        #matrix map forming
        map_index= np.zeros((len_map,len_map))
        # print(tup_sp) #=> first x then y
        num_pos=0
        for i in range(0,len_map):
            for j in range(0,len_map):
                fill_tup=sort_x_list[num_pos]
                map_index[j,i]=fill_tup[0]
                num_pos=num_pos+1
                
        
        '''
        i=j=0
        map_index[j,i]=sort_x_list[0][0] #first entry
        for l_idx in range(len(sort_x_list)-1):
            cur_sp=sort_x_list[l_idx]
            nxt_sp=sort_x_list[l_idx+1]
            
            #print(tup_sp)
            #print(nxt_sp)
            if(nxt_sp[1]-cur_sp[1]>0):
              i=i+1
              
            if(nxt_sp[2]-cur_sp[2]>0):
              j=j+1
            
            if(i+j==len_map*2):
              break
            
            print(j,'/',i)
            # fill map fix x ,move y
            map_index[j,i]=nxt_sp[0]
        '''          
        csv_spmap=''.join([SP_DIR, '/spLabelMap_', fileNAME, '.csv'])
        with open(csv_spmap, 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerows(map_index)
        #os.system("pause")
        # print(map_index)
        
        print(map_index.size)
        
                                
        # for (segVal) in np.unique(segments):
          	# construct a mask for the segment
          	# mask = np.zeros(image.shape[:2], dtype = "uint8")
          	# seg pos location
           
          	# print('segVal:',segVal)
            # print(image[segments==segVal])           
           
            # keep in [numid, x_val, y_val, RGB_avg, Depth_avg ] 
            # position
            # print ( segments[segVal] )
            
            #RGB feature transfomation
        '''
        fig = plt.figure("Superpixels -- segments" )
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(image, segments))
        plt.axis("off")
        plt.show()    
        '''
"""
