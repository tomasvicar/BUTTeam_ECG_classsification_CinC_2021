import torch
import pandas
import numpy as np
import os

from utils.losses import wce,Challange_metric_loss
from utils.utils import load_weights
from utils import transforms

class Config:

    
    twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
    four_leads = ('I', 'II', 'III', 'V2')
    three_leads = ('I', 'II', 'V2')
    two_leads = ('I', 'II')
    LEAD_LISTS = (twelve_leads, six_leads, four_leads, three_leads, two_leads)
    


    
    EQUIVALENT_CLASSES_MAP  = {
        "733534002": "164909002",
        "713427006": "59118001",
        "284470004": "63593006",
        "427172004": "17338001",
    }
    

    
    table = pandas.read_csv('dx_mapping_scored.csv', usecols=[1, 2])
    snomed_codes, labels = table['SNOMEDCTCode'], table['Abbreviation']
    
    SNOMED2ABB_MAP = {str(code): label for code, label in zip(snomed_codes, labels)}
    for k in list(EQUIVALENT_CLASSES_MAP.keys()):
        SNOMED2ABB_MAP.pop(k, None)
 
    SNOMED2IDX_MAP = {key: idx for idx, key in enumerate(SNOMED2ABB_MAP)}


    WEIGHTS = load_weights('weights.csv', list(SNOMED2IDX_MAP.keys()))
    
    
    MODEL_NOTE = 'replica_att'

    
    SPLIT_RATIO=[9,1]

    Fs = 150
    
    # MAX_LEN = 10   ## zbyde 81174/88253
    # MAX_LEN = 11   ## zbyde 81863/88253
    # MAX_LEN = 12   ## zbyde 82690/88253
    MAX_LEN = 15   ## zbyde 84469/88253
    # MAX_LEN = 20   ## zbyde 85817/88253
    # MAX_LEN = 30   ## zbyde 86786/88253
    # MAX_LEN = 40   ## zbyde 87319/88253
    # MAX_LEN = 50   ## zbyde 87526/88253
    # MAX_LEN = 60   ## zbyde 87661/88253
    # MAX_LEN = 70   ## zbyde 87676/88253
    # MAX_LEN = 100   ## zbyde 87696/88253
    # MAX_LEN = 150   ## zbyde 88179/88253

    DEVICE=torch.device("cuda:"+str(torch.cuda.current_device()))
    
    if os.path.isdir('../data'):
        DATA_PATH = '../data'
    if os.path.isdir('../../../cardio_shared/data'):    
        DATA_PATH = '../../../cardio_shared/data'
    
    
    DATA_RESAVE_PATH = '../data_resave'
    
    
    # BATCH = 32
    # BATCH = 60
    BATCH = 60
    # BATCH = 16
    # BATCH = 10
    
    MODELS_SEED = 42
    
        
    LR_LIST = np.array([0.01,0.001,0.0001,0.01,0.001,0.0001])/10
    # LR_CHANGES_LIST = [30,20,10,15,10,10]
    LR_CHANGES_LIST = [40,20,10,20,15,10]
    LOSS_FUNTIONS = [wce,wce,wce,Challange_metric_loss(WEIGHTS),Challange_metric_loss(WEIGHTS),Challange_metric_loss(WEIGHTS)]
    MAX_EPOCH = np.sum(LR_CHANGES_LIST)
    
    
    # LR_LIST = np.array([0.01,0.001,0.0001,0.01,0.001,0.0001])/10
    # LR_CHANGES_LIST = [2,1,1,1]
    # LOSS_FUNTIONS = [wce,wce,Challange_metric_loss(WEIGHTS),Challange_metric_loss(WEIGHTS)]
    # MAX_EPOCH = np.sum(LR_CHANGES_LIST)
    
    
    
    # LEVELS = 7
    # LVL1_SIZE = 2
    # OUTPUT_SIZE = len(SNOMED2IDX_MAP)
    # CONVS_IN_LAYER = 2
    # BLOCKS_IN_LVL = 2
    # FILTER_SIZE = 3
    
    LEVELS = 7
    LVL1_SIZE = 8*3
    # LVL1_SIZE = 8*6
    OUTPUT_SIZE = len(SNOMED2IDX_MAP)
    CONVS_IN_LAYER = 3
    BLOCKS_IN_LVL = 3
    FILTER_SIZE = 3
    
    # DO = 0.3
    DO = None
    
    
    WEIGHT_DECAY = 1e-5
    
    
    NUM_WORKERS_TRAIN = 6
    NUM_WORKERS_VALID = 6
    
    # NUM_WORKERS_TRAIN = 0
    # NUM_WORKERS_VALID = 0
    
    TRANSFORM_DATA_TRAIN_NONREP = transforms.Compose([
        transforms.RandomAmplifier(p=0.8,max_multiplier=0.3),
        transforms.RandomStretch(p=0.8, max_stretch=0.2),
        ])
    
    TRANSFORM_DATA_TRAIN_REP = transforms.Compose([
        transforms.RandomShift(p=0.8),
        ])
    
    TRANSFORM_DATA_VALID_NONREP = None
    TRANSFORM_DATA_VALID_REP = None
    
    T_OPTIMIZE_INIT=250
    T_OPTIMIZER_GP=50
    
