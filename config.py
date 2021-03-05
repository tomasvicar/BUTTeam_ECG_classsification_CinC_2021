from utils.losses import wce,challange_metric_loss
import torch
from utils import transforms
import numpy as np


class Config:
    
    
    
    MODEL_NOTE='test0'
    
    
    SPLIT_RATIO=[9,1]

    NUM_WORKERS=4

    
    BATCH_SIZE=32
    
    
    
    MODELS_SEEDS=[42]



    LR_LIST_INIT=np.array([0.01,0.001,0.0001,0.01,0.001,0.0001])/10
    LR_CHANGES_LIST_INIT=[30,20,10,15,10,10]
    LOSS_FUNTIONS_INIT=[wce,wce,wce,challange_metric_loss,challange_metric_loss,challange_metric_loss]
    

    
    DEVICE=torch.device("cuda:"+str(torch.cuda.current_device()))
    
    
    WEIGHT_DECAY=1e-5
    

    LEVELS=6
    LVL1_SIZE=4
    OUTPUT_SIZE=24
    CONVS_IN_LAYERS=3
    INIT_CONV=LVL1_SIZE
    FILTER_SIZE=3
    
    T_OPTIMIZE_INIT=250
    T_OPTIMIZER_GP=50
    

    
    
    OUTPUT_SAMPLING=125
    STD=0.2
    
    TRANSFORM_DATA_TRAIN=transforms.Compose([
        transforms.Resample(output_sampling=OUTPUT_SAMPLING),
        transforms.ZScore(mean=0,std=STD),
        transforms.RandomAmplifier(p=0.8,max_multiplier=0.2),
        transforms.RandomStretch(p=0.8, max_stretch=0.1),
        ])
    
    TRANSFORM_DATA_VALID=transforms.Compose([
        transforms.Resample(output_sampling=OUTPUT_SAMPLING),
        transforms.ZScore(mean=0,std=STD),
        ])
    
    TRANSFORM_LBL=transforms.SnomedToOneHot()
    
    


