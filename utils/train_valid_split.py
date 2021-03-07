import numpy as np 
from config import Config

def train_valid_split(names_onehot_lens,seed,split_ratio):
    
    
    state=np.random.get_state()
    
    
    num_files = len(names_onehot_lens)
    np.random.seed(seed)
    split_ratio_ind = int(np.floor(split_ratio[0] / (split_ratio[0] + split_ratio[1]) * num_files))
    permuted_idx = np.random.permutation(num_files)
    train_ind = permuted_idx[:split_ratio_ind]
    valid_ind = permuted_idx[split_ratio_ind:]
    
    names_onehot_lens_train = [names_onehot_lens[i] for i in train_ind]
    names_onehot_lens_valid = [names_onehot_lens[i] for i in valid_ind]
    
    
    np.random.set_state(state)
    
    return names_onehot_lens_train,names_onehot_lens_valid