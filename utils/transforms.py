import numpy as np
import random
import torch
from scipy.signal import firwin
from scipy.signal import filtfilt
from scipy.signal import iirnotch
from scipy.signal import windows
from scipy.signal import fftconvolve



class Compose(object):
    """Composes several transforms together.
    Example:
        transforms.Compose([
            transforms.HardClip(10),
            transforms.ToTensor(),
            ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data_sample, **kwargs):
        if self.transforms:
            for t in self.transforms:
                data_sample = t(data_sample, **kwargs)
        return data_sample


class Resample:
    def __init__(self, output_sampling=500):
        self.output_sampling = int(output_sampling)

    def __call__(self, sample, input_sampling):
        
        sample=sample.astype(np.float32)
        for k in range(sample.shape[0]):
            sample[k,:]=sample[k,:]

        # Rescale data
        self.sample = sample
        self.input_sampling = int(input_sampling)
        
        factor=self.output_sampling / self.input_sampling
        
        len_old=self.sample.shape[1]
        num_of_leads=self.sample.shape[0]
        

        new_length = int(factor * len_old)
        resampled_sample = np.zeros((num_of_leads, new_length))

        for channel_idx in range(num_of_leads):
            tmp=self.sample[channel_idx, :]
            
            ### antialias
            if factor<1:
                q=1/factor
                
                half_len = 10 * q  
                n = 2 * half_len
                b, a = firwin(int(n)+1, 1./q, window='hamming'), 1.
                tmp = filtfilt(b, a, tmp)
            
            
            l1=np.linspace(0,len_old - 1, new_length)
            l2=np.linspace(0,len_old - 1, len_old)
            tmp= np.interp(l1,l2,tmp)
            resampled_sample[channel_idx, :] = tmp

        return resampled_sample
    
    
class Remover_50_100_150_60_120_180Hz:
    def __init__(self):
        pass

    def __call__(self, sample, input_sampling):
        
        fs = input_sampling
        
        num_of_leads = sample.shape[0]

        for channel_idx in range(num_of_leads):
            tmp = sample[channel_idx, :]
            
            Q = 15.0
            f0s = [50,60,100,120,150,180]
            
            
            
            for f0 in f0s:
                if f0 < (fs/2):
                    
                    b, a = iirnotch(f0, Q, fs)
                    tmp = filtfilt(b, a, tmp)
            
            
            

            sample[channel_idx, :] = tmp

        return sample
      
        
class BaseLineFilter:
    def __init__(self, window_size=1000):
        self.window_size = window_size

    def __call__(self, sample, **kwargs):
        for channel_idx in range(sample.shape[0]):
            running_mean = BaseLineFilter._running_mean(sample[channel_idx], self.window_size)
            sample[channel_idx] = sample[channel_idx] - running_mean
        return sample

    @staticmethod
    def _running_mean(sample, window_size):
        window = windows.blackman(window_size)
        window = window / np.sum(window)
        return fftconvolve(sample, window, mode="same")
    
    
class SnomedToOneHot(object):
    """Returns one hot encoded labels"""
    def __init__(self):
        pass

    def __call__(self, snomed_codes, mapping):
        encoded_labels = np.zeros(len(mapping)).astype(np.float32)
        for code in snomed_codes:
            if code not in mapping:
                continue
            else:
                encoded_labels[mapping[code]] = 1.0

        return encoded_labels    
    
    
    
class RandomShift:
    """
        Class randomly shifts signal within temporal dimension
    """
    def __init__(self, p=0):
        self.probability = p

    def __call__(self, sample, **kwargs):
        self.sample_length = sample.shape[1]
        self.sample_channels = sample.shape[0]

        if random.random() < self.probability:
           
            shift = torch.randint(self.sample_length, (1, 1)).view(-1).numpy()
            
            sample=np.roll(sample, shift, axis=1)
            
        return sample


class RandomStretch:
    """
    Class randomly stretches temporal dimension of signal
    """
    def __init__(self, p=0, max_stretch=0.1):
        self.probability = p
        self.max_stretch = max_stretch

    def __call__(self, sample, **kwargs):
        self.sample_length = sample.shape[1]
        self.sample_channels = sample.shape[0]

        if random.random() < self.probability:
            relative_change = 1 + torch.rand(1).numpy()[0] * 2 * self.max_stretch - self.max_stretch
            if relative_change<1:
                relative_change=1/(1-relative_change+1)
            
            
            new_len = int(relative_change * self.sample_length)

            stretched_sample = np.zeros((self.sample_channels, new_len))
            for channel_idx in range(self.sample_channels):
                stretched_sample[channel_idx, :] = np.interp(np.linspace(0, self.sample_length - 1, new_len),
                                                             np.linspace(0, self.sample_length - 1, self.sample_length),
                                                             sample[channel_idx, :])
                
            sample=stretched_sample
        return sample


class RandomAmplifier:
    """
    Class randomly amplifies signal
    """
    def __init__(self, p=0, max_multiplier=0.2):
        self.probability = p
        self.max_multiplier = max_multiplier

    def __call__(self, sample, **kwargs):
        self.sample_length = sample.shape[1]
        self.sample_channels = sample.shape[0]

        if random.random() < self.probability:
            for channel_idx in range(sample.shape[0]):
                multiplier = 1 + random.random() * 2 * self.max_multiplier - self.max_multiplier
                
                ##mutliply by 2 is same as equvalent to multiply by 0.5 not 0!
                if multiplier<1:
                    multiplier=1/(1-multiplier+1)
                    
                sample[channel_idx, :] = sample[channel_idx, :] * multiplier

        return sample
    
    