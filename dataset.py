from torch.utils import data
import numpy as np
import torch


class Dataset(data.Dataset):


    def __init__(self, names_onehot_lens_train, transform=None):

        self.filenames = [item.name for item in names_onehot_lens_train]
        self.onehots = [item.onehot for item in names_onehot_lens_train]
        
        self.transform = transform
        
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        sample = np.load(self.filenames[idx])
        y = self.onehots[idx]
        
        if self.transform:
            sample = self.transform(sample)

        sample_length = sample.shape[1]
        
        return sample, y, sample_length

    @staticmethod
    def pad_collate(batch,val = 0):
            """
            Returns padded mini-batch
            :param batch: (list of tuples): tensor, label
            :return: padded_array - a tensor of all examples in 'batch' after padding
            labels - a LongTensor of all labels in batch
            sample_lengths – origin lengths of input data
            """
            
            
            batch_size = len(batch)
            # random_idx = random.shuffle(list(range(batch_size-1)))
    
            # find the longest sequence
            sample_lengths = [sample[0].shape[1] for sample in batch]
            max_len = max(sample_lengths)
    
            num_channels = batch[0][0].shape[0]
    
            # preallocate padded NumPy array
            shape = (batch_size, num_channels, max_len)
            padded_array = val * np.ones(shape, dtype=np.float32)
            list_of_labels = []
    
            for idx, sample in enumerate(batch):
                padded_array[idx, :, :sample_lengths[idx]] = sample[0]
                list_of_labels.append(sample[1])
    
            # Pass to Torch Tensor
            padded_array = torch.from_numpy(padded_array).float()
            labels = torch.from_numpy(np.array(list_of_labels)).float() 
            sample_lengths = torch.LongTensor(sample_lengths)
    
            return padded_array, labels, sample_lengths