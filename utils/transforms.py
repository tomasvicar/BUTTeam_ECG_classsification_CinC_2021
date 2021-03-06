import numpy as np
from scipy.signal import firwin
from scipy.signal import filtfilt
from scipy.signal import iirnotch
from scipy.signal import windows
from scipy.signal import fftconvolve

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
            
            f0 = 50
            b, a = iirnotch(f0, Q, fs)
            tmp = filtfilt(b, a, tmp)
            
            f0 = 60
            b, a = iirnotch(f0, Q, fs)
            tmp = filtfilt(b, a, tmp)
            
            f0 = 100
            b, a = iirnotch(f0, Q, fs)
            tmp = filtfilt(b, a, tmp)
            
            f0 = 120
            b, a = iirnotch(f0, Q, fs)
            tmp = filtfilt(b, a, tmp)
            
            f0 = 150
            b, a = iirnotch(f0, Q, fs)
            tmp = filtfilt(b, a, tmp)
            
            f0 = 180
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