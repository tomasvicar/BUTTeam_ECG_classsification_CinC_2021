import os
import logging
import nvidia_smi

from config import Config
from resave_data import resave_data
from utils.utils import get_gpu_memory




def training_code(data_directory, model_directory):
    
    
    if not os.path.isdir(model_directory):
        resave_data(data_directory,model_directory)
    
    
        
    for lead_list in Config.LEAD_LISTS:
        
        train_one_model(model_directory,lead_list)
        
        
        
def train_one_model(model_directory,lead_list):
    
    LR_LIST=Config.LR_LIST
    LR_CHANGES_LIST=Config.LR_CHANGES_LIST
    LOSS_FUNTIONS=Config.LOSS_FUNTIONS
    MAX_EPOCH=Config.MAX_EPOCH
    
    
    nvidia_smi.nvmlInit()
    measured_gpu_memory = []
    measured_gpu_memory.append(get_gpu_memory(nvidia_smi))
    
    
    
    





if __name__ == '__main__':
    logging.basicConfig(filename='debug.log',level=logging.INFO)
    try:
        data_directory = Config.DATA_PATH
        model_directory = Config.DATA_RESAVE_PATH
        
        
        training_code(data_directory, model_directory)


    except Exception as e:
        print(e)
        logging.critical(e, exc_info=True)
        
        
    
    
    
    
    
    
