import numpy as np
import torch

from utils.read_header import read_header
from utils.optimize_ts import aply_ts
from utils import transforms

def run_model_fcn(model, header, recording):
    
    
    header_tmp = read_header(header,from_file=False)
    sampling_frequency, resolution, age, sex, snomed_codes,leads = header_tmp
    
    
    leads_cfg = [ leads_cfg for leads_cfg in model.config.LEAD_LISTS if set(leads_cfg)==set(leads)][0]
    

    order = [leads.index(element) for element in leads_cfg]
    
    
    signal = recording[order,:]/np.expand_dims(np.array(resolution),axis=1)
    
    
    resampler = transforms.Resample(output_sampling=model.config.Fs)
    remover_50_100_150_60_120_180Hz = transforms.Remover_50_100_150_60_120_180Hz()
    baseLineFilter = transforms.BaseLineFilter()
    
    signal = remover_50_100_150_60_120_180Hz(signal,input_sampling=sampling_frequency)
    
    signal = resampler(signal,input_sampling=sampling_frequency)
    
    signal = baseLineFilter(signal)
    
    
    
    lens_all = model.lens
    batch = model.config.BATCH
    
    
    lens_sample = np.random.choice(lens_all, batch, replace=False)
    random_batch_length = np.max(lens_sample)
    
    
    len_batch = model.config.MAX_LEN * model.config.Fs
    
    reshaped_data,lens = generate_batch(signal, random_batch_length,len_batch,model.config.Fs)
    
    data = torch.from_numpy(reshaped_data.copy())
    
    lens = torch.from_numpy(np.array(lens).astype(np.float32))

    cuda_check = next(model.parameters()).is_cuda
    if cuda_check:
        cuda_device = next(model.parameters()).get_device()
        device = torch.device('cuda:' + str(cuda_device) )
        lens=lens.to(device)
        data=data.to(device)
    
    
    with torch.no_grad():
        
        data_tmp_all = torch.split(data,int(np.ceil(data.size(0)/batch)),dim=0)
        lens_tmp_all = torch.split(lens,int(np.ceil(lens.size(0)/batch)),dim=0)
        score_tmp = []
        for data_tmp,lens_tmp in zip(data_tmp_all,lens_tmp_all):
            score_tmp.append(model(data_tmp,lens_tmp))
            
            
        score = torch.cat(score_tmp,0)
        
        score = score.detach().cpu().numpy()
        label = aply_ts(score,model.get_ts()).astype(np.float32)
    
    label = merge_labels(label)
    score = merge_labels(score)
    
    
    label = label.astype(int)    
    tmp = model.config.SNOMED2IDX_MAP
    classes = [ list(tmp.keys())[list(tmp.values()).index(k)] for k in range(len(tmp))]
    
    
    return classes,label,score
    
    
    
def generate_batch(sample, random_batch_length,len_batch,sampling_freq):

    lens=[]

    # Compute number of chunks
    if sample.shape[1]>int(len_batch):
    
        max_chunk_length = int(len_batch)
        overlap = int(sampling_freq * 0.5)
        num_of_chunks = (sample.shape[1] - overlap) // (max_chunk_length - overlap)

    
        # Generate split indices
        onsets_list = [idx * (max_chunk_length - overlap) for idx in range(num_of_chunks)]
        offsets_list = [max_chunk_length + idx * (max_chunk_length - overlap) for idx in range(num_of_chunks)]
        offsets_list[-1] = sample.shape[1]
        
        max_length = len_batch
        
        # Initialize batch
        batch = np.zeros([num_of_chunks, sample.shape[0], max_length])

        # Generate batch from sample chunks
        for idx, onset, offset in zip(range(num_of_chunks), onsets_list, offsets_list):
            chunk = sample[:, onset:offset]
            
            idx_pos = 0
            len_ = chunk.shape[1]
            while idx_pos < len_batch:
                tmp = min((idx_pos+len_),batch.shape[2])
                batch[idx, :, idx_pos:tmp] = chunk[:,:(tmp - idx_pos)]
                idx_pos += len_
            
            
            lens.append(len_batch)

    else:
        max_length = len_batch

        # Initialize batch
        batch = np.zeros([1, sample.shape[0], max_length])
        # Generate batch
        # batch[0, :, :sample.shape[1]] = sample
        
        idx_pos = 0
        len_ = sample.shape[1]
        while idx_pos < len_batch:
            tmp = min((idx_pos+len_),batch.shape[2])
            batch[0, :, idx_pos:tmp] = sample[:,:(tmp - idx_pos)]
            idx_pos += len_
        
        
        lens.append(len_batch)

    return batch.astype(np.float32),lens    
    
    
    
def merge_labels(labels):
    """
    Merges labels across single batch
    :param labels: one hot encoded labels, shape=(batch, 1, num_of_classes)
    :return: aggregated one hot encoded labels, shape=(1, 1, num_of_classes)
    """
    return np.max(labels, axis=0)
    
    

    