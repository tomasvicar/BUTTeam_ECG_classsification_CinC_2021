


def get_gpu_memory(nvidia_smi):
    
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    
    return info.used/1000000000


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr'
                           
                           

                             
                           
                           