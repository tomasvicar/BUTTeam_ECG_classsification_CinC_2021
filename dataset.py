from torch.utils import data
import numpy as np
import torch
from config import Config


class Dataset(data.Dataset):


    def __init__(self, names_onehot_lens_train, transform_nonrep=None,transform_rep=None):

        self.filenames = [item.name for item in names_onehot_lens_train]
        self.onehots = [item.onehot for item in names_onehot_lens_train]
        
        self.transform_nonrep = transform_nonrep
        self.transform_rep = transform_rep
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        sample = np.load(self.filenames[idx])
        
        if np.sum(np.isnan(sample).astype(np.float32))>0:
            print('nanyyyyy  ' +  self.filenames[idx])
            sample[np.isnan(sample)] = 0

        
        y = self.onehots[idx]
        
        if self.transform_nonrep:
            sample = self.transform_nonrep(sample)

        
        return sample, y, self.transform_rep

    @staticmethod
    def pad_collate(batch,val=0):
        """
        Returns padded mini-batch
        :param batch: (list of tuples): tensor, label
        :return: padded_array - a tensor of all examples in 'batch' after padding
        labels - a LongTensor of all labels in batch
        sample_lengths – origin lengths of input data
        """
        
        transform_rep = batch[0][2]
        
        batch_size = len(batch)
        # random_idx = random.shuffle(list(range(batch_size-1)))

        # find the longest sequence
        sample_lengths = [sample[0].shape[1] for sample in batch]
        max_len = max(sample_lengths)

        num_channels = batch[0][0].shape[0]


        len_batch = Config.MAX_LEN * Config.Fs
        # preallocate padded NumPy array
        shape = (batch_size, num_channels, len_batch)
        padded_array = val * np.ones(shape, dtype=np.float32)
        list_of_labels = []

        
        
        for idx, sample in enumerate(batch):
            idx_pos = 0
            len_ = sample_lengths[idx]
            while idx_pos < len_batch:
                tmp = min((idx_pos+len_),padded_array.shape[2])
                padded_array[idx, :, idx_pos:tmp] = sample[0][:,:(tmp - idx_pos)]
                idx_pos += len_
            list_of_labels.append(sample[1])
        
        if transform_rep:
            padded_array = transform_rep(padded_array)
        
        sample_lengths = [len_batch for tmp in sample_lengths]

        # Pass to Torch Tensor
        padded_array = torch.from_numpy(padded_array).float()
        labels = torch.from_numpy(np.array(list_of_labels)).float() 
        sample_lengths = torch.LongTensor(sample_lengths)

        return padded_array, labels, sample_lengths
    
    
    
    
    
    
    
    
    