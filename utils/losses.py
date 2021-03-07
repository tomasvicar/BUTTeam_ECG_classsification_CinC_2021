import torch
import numpy as np


def wce(res,lbls,w_positive_tensor,w_negative_tensor):
    ## weighted crossetropy - weigths are for positive and negative 
    res_c = torch.clamp(res,min=1e-6,max=1-1e-6)
            
    p1=lbls*torch.log(res_c)*w_positive_tensor
    p2=(1-lbls)*torch.log(1-res_c)*w_negative_tensor
    
    return -torch.mean(p1+p2)



class Challange_metric_loss:
    def __init__(self, weights):
        self.weights = weights
        
    def __call__(self,res,lbls,w_positive_tensor,w_negative_tensor):
        
        weights = self.weights
        
        weights = torch.from_numpy(weights.astype(np.float32))
        
        normalizer=torch.sum(lbls+res-lbls*res,dim=1)
        normalizer[normalizer<1]=1
        
        num_sigs,num_classes=list(lbls.size())
        
    
        A=torch.zeros((num_classes,num_classes),dtype=lbls.dtype)
        
        cuda_check = lbls.is_cuda
        if cuda_check:
            cuda_device = lbls.get_device()
            device = torch.device('cuda:' + str(cuda_device) )
            A=A.to(device)
            weights=weights.to(device)
    
    
        for sig_num in range(num_sigs):
            tmp=torch.matmul(torch.transpose(lbls[[sig_num], :],0,1),res[[sig_num], :])/normalizer[sig_num]
            A=A + tmp
        
        return -torch.sum(A*weights)
    

    






