import numpy as np
from config import Config



def compute_challenge_metric_custom(res,lbls,normalize=True):
    
    
    normal_class = '426783006'

    normal_index = Config.SNOMED2IDX_MAP[normal_class]


    lbls=lbls>0
    res=res>0
    
    weights = Config.WEIGHTS
    
    observed_score=np.sum(weights*get_confusion(lbls,res))
    
    if normalize == False:
        return observed_score
    
    correct_score=np.sum(weights*get_confusion(lbls,lbls))
    
    inactive_outputs = np.zeros_like(lbls)
    inactive_outputs[:, normal_index] = 1
    inactive_score=np.sum(weights*get_confusion(lbls,inactive_outputs))
    
    if float(correct_score - inactive_score) != 0:
        normalized_score = (float(observed_score - inactive_score) / float(correct_score - inactive_score))  
    else:
        normalized_score = 0
        print('zero division score')
    
    # normalized_score = (float(observed_score - inactive_score) / float(correct_score - inactive_score))
    
    return normalized_score


        
def get_confusion(lbls,res):

    
    normalizer=np.sum(lbls|res,axis=1)
    normalizer[normalizer<1]=1
    
    A=lbls.astype(np.float32).T@(res.astype(np.float32)/normalizer.reshape(normalizer.shape[0],1))
    
    return A