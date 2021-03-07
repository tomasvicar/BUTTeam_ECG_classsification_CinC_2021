from collections import namedtuple
import os
from utils.transforms import SnomedToOneHot
from config import Config

def get_data(filenames):
    
    snomedToOneHot = SnomedToOneHot()
    
    names_onehot_lens = []
    
    Data = namedtuple('Data','name onehot len')
    
    for name in filenames:
        
        
        head, tail = os.path.split(name.replace('.npy',''))
        _,snomeds,len_ = tail.split('-')
        
        snomeds = snomeds.split('_')
        
        onehot = snomedToOneHot(snomeds,Config.SNOMED2IDX_MAP)
        
        len_ = int(len_)
        
        names_onehot_lens.append(Data(name,onehot,len_))
        
    return names_onehot_lens
        
        
        
        
        
        
        
        


