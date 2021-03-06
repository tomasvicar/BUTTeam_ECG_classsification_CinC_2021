import torch
import pandas
import numpy as np

from utils.losses import wce,Challange_metric_loss
from evaluate_model import load_weights

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


    WEIGHTS = load_weights('weights.csv', list(EQUIVALENT_CLASSES_MAP.items()))# to mi peťulka poradila
    
    




    Fs = 150

    DEVICE=torch.device("cuda:"+str(torch.cuda.current_device()))
    
    DATA_PATH = '../data'
    DATA_RESAVE_PATH = '../data_resave'
    
    
    BATCH = 32
    
    MODELS_SEEDS=[42]
    
    
    
    LR_LIST_INIT=np.array([0.01,0.001,0.0001,0.01,0.001,0.0001])/10
    LR_CHANGES_LIST_INIT=[30,20,10,15,10,10]
    LOSS_FUNTIONS_INIT=[wce,wce,wce,Challange_metric_loss(WEIGHTS),Challange_metric_loss(WEIGHTS),Challange_metric_loss(WEIGHTS)]
    MAX_EPOCH_INIT=np.sum(LR_CHANGES_LIST_INIT)
    
    
    
    
    
    
    
    
    

    

