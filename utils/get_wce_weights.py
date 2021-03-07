import numpy as np
import torch

from config import Config

def get_wce_weights(names_onehot_lens_train):
    
    
    onehot = [item.onehot for item in names_onehot_lens_train]
    onehot = np.stack(onehot,axis=1)
    lbl_counts = np.sum(onehot,axis=1)
    num_files = len(names_onehot_lens_train)
    
    
    w_positive = num_files / lbl_counts
    w_negative = num_files / (num_files - lbl_counts)
    
    w_positive_tensor=torch.from_numpy(w_positive.astype(np.float32)).to(Config.DEVICE)
    w_negative_tensor=torch.from_numpy(w_negative.astype(np.float32)).to(Config.DEVICE)
    
    
    return w_positive_tensor,w_negative_tensor
