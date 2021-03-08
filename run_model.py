import numpy as np
import torch

from utils.read_header import read_header
from utils.optimize_ts import aply_ts

def run_model(model, header, recording):
    
    
    header_tmp = read_header(header,from_file=False)
    sampling_frequency, resolution, age, sex, snomed_codes,leads = header_tmp
    
    
    leads_cfg = [ leads_cfg for leads_cfg in model.config.LEAD_LISTS if len(leads_cfg)==len(leads)][0]
    

    order = [leads.index(element) for element in leads_cfg]
    
    
    recording = recording[order,:]/np.expand_dims(np.array(resolution),axis=1)
    
    
    lens_all = model.lens
    batch = model.config.BATCH
    
    
    lens_sample = np.random.choice(lens_all, batch, replace=False)
    random_batch_length = np.max(lens_sample)
    
    
    reshaped_data,lens = generate_batch(recording, random_batch_length,model.config.Fs)
    
    data = torch.from_numpy(reshaped_data.copy())
    
    lens = torch.from_numpy(np.array(lens).astype(np.float32))

    cuda_check = next(model.parameters()).is_cuda
    if cuda_check:
        cuda_device = next(model.parameters()).get_device()
        device = torch.device('cuda:' + str(cuda_device) )
        lens=lens.to(device)
        data=data.to(device)
    
    
    score = model(data,lens)
    score = score.detach().cpu().numpy()
    label = aply_ts(score,model.get_ts()).astype(np.float32)
    
    label = merge_labels(label)
    score = merge_labels(score)
    
    
    label = label.astype(int)    
    tmp = model.config.SNOMED2IDX_MAP
    classes = [ list(tmp.keys())[list(tmp.values()).index(k)] for k in range(len(tmp))]
    
    
    return classes,label,score
    
    
    
def generate_batch(sample, random_batch_length,sampling_freq):

    lens=[]

    # Compute number of chunks
    if sample.shape[1]>int(sampling_freq * 105):
    
        max_chunk_length = int(sampling_freq * 90)
        overlap = int(sampling_freq * 5)
        num_of_chunks = (sample.shape[1] - overlap) // (max_chunk_length - overlap)

    
        # Generate split indices
        onsets_list = [idx * (max_chunk_length - overlap) for idx in range(num_of_chunks)]
        offsets_list = [max_chunk_length + idx * (max_chunk_length - overlap) for idx in range(num_of_chunks)]
        offsets_list[-1] = sample.shape[1]
        max_length = max(random_batch_length, offsets_list[-1] - onsets_list[-1])

        # Initialize batch
        batch = np.zeros([num_of_chunks, sample.shape[0], max_length])

        # Generate batch from sample chunks
        for idx, onset, offset in zip(range(num_of_chunks), onsets_list, offsets_list):
            chunk = sample[:, onset:offset]
            batch[idx, :, :chunk.shape[1]] = chunk
            lens.append(chunk.shape[1])

    else:
        max_length = max(random_batch_length, sample.shape[1])

        # Initialize batch
        batch = np.zeros([1, sample.shape[0], max_length])
        # Generate batch
        batch[0, :, :sample.shape[1]] = sample
        
        lens.append(sample.shape[1])

    return batch.astype(np.float32),lens    
    
    
    
def merge_labels(labels):
    """
    Merges labels across single batch
    :param labels: one hot encoded labels, shape=(batch, 1, num_of_classes)
    :return: aggregated one hot encoded labels, shape=(1, 1, num_of_classes)
    """
    return np.max(labels, axis=0)
    
    

    