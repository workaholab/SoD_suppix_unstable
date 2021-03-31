

class Config:
   def __init__(self): # initial states of config parameters
        ### train/test shared
        
        ## definition ##
        # STATE_TRTE=1 #1=Train, 0=Test 
        # INPUT_TYPE=0 #0=pixel, 1=superpixel
        # DATASET_TYPE=1 #1=RGBD, 0=RGB
        
        ## catalorize the parameters ##
        # shared ==============================================================================
        self.DEBUG=False
        self.input_size=[640,640]
        self.rgb_input_size=[224,224]
        
        # dataset / pre-processing ==============================================================================
        ### dataset construction
        self.rgb_Dirpath="../RGBD_model_testing/data/rgb/"
        self.dep_Dirpath="../RGBD_model_testing/data/depth/"
        self.gt_Dirpath="../RGBD_model_testing/data/groundtruth/"        
      
        ## mean value of dataset
        self.mean = [126.47384089, 124.27250705, 122.54440459, 160.28555339]
        self.mean_train = [112.13209312, 108.6547371, 101.00103511, 113.27608583]
        self.mean_msra = [114.86545157, 110.46705426, 95.90594382]
        self.mean_rgbtrain = [120.67928353, 114.42846415, 100.75286721]
        
        self.mean_test = [112.22424188, 108.63577088, 100.11004508, 126.90818425]
        self.mean_test_NLPR = [123.70062814, 121.94022074, 120.47344181, 94.61725053]
        self.mean_test_NJUD = [106.6313464, 102.52297522, 91.48286773, 119.98602273]
        self.mean_LFSD = [128.65956412, 117.2438686, 107.97547353, 116.08585262]
        self.mean_RGBD = [125.93331761, 121.60306597, 116.12658972, 151.30741493]
        self.mean_PKU = [106.99048091, 101.79961792, 91.27128536, 188.2142267]  

        
        # model ==============================================================================
        # select model architecture
        self.model_sel=3
        '''
        if(MODEL_SEL==0):
          model =  Net(device,start_EP)
        elif(MODEL_SEL==1):
          model =  Net_interpolation(device,start_EP)
        elif(MODEL_SEL==2):
          model = Net2(device,start_EP)
        elif(MODEL_SEL==3)
          model = Net3(device,start_EP)
        
        '''

        self.model_save_dir="modelParam_T%d/"
        self.model_pth_file="param_e%02d.pth"

        # share by Train and Test
        self.start_VerTrain=3 #0: no specific version need to be replaced
        # training set (define the starting point)
        self.TRAIN_ON=True
        self.start_EP=0
        # testing set (end of model training parameters to use)
        self.TEST_ON=True
        self.end_EP=40        

        # visualization ==============================================================================
        self.visual_save_dir="VisualResults_T%d/"        
        self.train_result_dir="Train/Epoch_%d/results_%s/" # visual_save_dir+..
        self.test_result_dir="Test/results_%s/" # visual_save_dir+..
        
        ### log file output
        self.LogVER=1
        self.Log_update=True
        self.log_dir="logs/"
        # file name
        self.train_log_file="train_T%d_V%d.log"
        self.test_log_file="testing_T%d_V%d.log"
        
        # saving visualization result file name
        self.PATH_resultfile="result_%s.png"
        self.PATH_rgbfile="rgb_test_%s.png"
        self.PATH_gtfile="gt_test_%s.png"
        self.PATH_depthfile="depth_test_%s.png"   
        self.PATH_b1="b1_result_%s.png"
        self.PATH_b2="b2_result_%s.png"
        self.PATH_b3="b3_result_%s.png"
        self.PATH_b4="b4_result_%s.png"
        self.PATH_b5="b5_result_%s.png"
        self.PATH_b6="b6_result_%s.png"
    
