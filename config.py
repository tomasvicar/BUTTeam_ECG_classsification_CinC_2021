import torch
import pandas
import numpy as np

from utils.losses import wce,Challange_metric_loss
from utils.utils import load_weights
from utils import transforms

class Config:

    LEAD_LISTS = [['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
              ['I', 'II', 'III', 'aVL', 'aVR', 'aVF'],
              ['I', 'II', 'V2'],
              ['II', 'V5'],
              ]

    
    EQUIVALENT_CLASSES_MAP  = {
        "59118001": "713427006",
        "63593006": "284470004",
        "17338001": "427172004",
    }
    

    
    table = pandas.read_csv('dx_mapping_scored.csv', usecols=[1, 2])
    snomed_codes, labels = table['SNOMED CT Code'], table['Abbreviation']
    
    SNOMED2ABB_MAP = {str(code): label for code, label in zip(snomed_codes, labels)}
    for k in list(EQUIVALENT_CLASSES_MAP.keys()):
        SNOMED2ABB_MAP.pop(k, None)
 
    SNOMED2IDX_MAP = {key: idx for idx, key in enumerate(SNOMED2ABB_MAP)}


    WEIGHTS = load_weights('weights.csv', list(SNOMED2IDX_MAP.keys()))
    
    
    MODEL_NOTE = 'first_model'
    
    SPLIT_RATIO=[9,1]

    Fs = 150

    DEVICE=torch.device("cuda:"+str(torch.cuda.current_device()))
    
    # DATA_PATH = '../data'
    DATA_PATH = '../../../cardio_shared/data'
    DATA_RESAVE_PATH = '../data_resave'
    
    
    BATCH = 64
    
    MODELS_SEED = 42
    
        
    LR_LIST = np.array([0.01,0.001,0.0001,0.01,0.001,0.0001])/10
    LR_CHANGES_LIST = [30,20,10,15,10,10]
    LOSS_FUNTIONS = [wce,wce,wce,Challange_metric_loss(WEIGHTS),Challange_metric_loss(WEIGHTS),Challange_metric_loss(WEIGHTS)]
    MAX_EPOCH = np.sum(LR_CHANGES_LIST)
    
    
    LEVELS = 6
    LVL1_SIZE = 6*2
    OUTPUT_SIZE = len(SNOMED2IDX_MAP)
    CONVS_IN_LAYERS = 3
    INIT_CONV = LVL1_SIZE
    FILTER_SIZE = 7
    
    
    WEIGHT_DECAY = 1e-5
    
    
    NUM_WORKERS_TRAIN = 7
    NUM_WORKERS_VALID = 7
    
    
    # NUM_WORKERS_TRAIN = 4
    # NUM_WORKERS_VALID = 2
    
    # NUM_WORKERS_TRAIN = 0
    # NUM_WORKERS_VALID = 0
    
    BATCH = 32
    
    TRANSFORM_DATA_TRAIN = transforms.Compose([
        transforms.RandomAmplifier(p=0.8,max_multiplier=0.3),
        transforms.RandomStretch(p=0.8, max_stretch=0.2),
        transforms.RandomShift(p=0.8),
        ])
    
    # TRANSFORM_DATA_TRAIN = None
    
    TRANSFORM_DATA_VALID = None
    
    T_OPTIMIZE_INIT=250
    T_OPTIMIZER_GP=50
    
