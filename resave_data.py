from glob import glob
import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from datetime import datetime
from itertools import starmap

from config import Config
from utils import transforms





def resave_one(filename, src_path ,dst_path):
        
        
    LEAD_LISTS = Config.LEAD_LISTS
    Fs = Config.Fs
    
    resampler = transforms.Resample(output_sampling=Fs)
    remover_50_100_150_60_120_180Hz = transforms.Remover_50_100_150_60_120_180Hz()
    baseLineFilter = transforms.BaseLineFilter()
    

    signal,fields = wfdb.io.rdsamp(filename)
    
    signal = signal.T.astype(np.float32)
    

    signal = remover_50_100_150_60_120_180Hz(signal,input_sampling=fields['fs'])
    
    signal = resampler(signal,input_sampling=fields['fs'])
    
    signal = baseLineFilter(signal)
    
    
    Dxs = [sub for sub in fields['comments'] if 'Dx: ' in sub][0].replace('Dx: ','').split(',')
    
    
    Dxs_string = '_'.join(Dxs)


    for lead_list in LEAD_LISTS:
        
        dst_path_tmp = dst_path + '/' + str(len(lead_list))
        
        # .replace('(','').replace(')','').replace(' ','').replace(',','_').replace('\'','')
        
        filename_save = filename.replace(src_path,dst_path_tmp) + '-' + Dxs_string + '-' + str(signal.shape[1]) + '.npy'
        
        use_leads = [fields['sig_name'].index(lead) for lead in lead_list]
        
        signal_tmp = signal[use_leads,:]
        
        np.save(filename_save,signal_tmp)
        
        
def resave_data(src_path,dst_path):
    
    LEAD_LISTS = Config.LEAD_LISTS
    
    
    subdir_names = glob(src_path + '/*/')
    
    
    for lead_list in LEAD_LISTS:
        
        dst_path_tmp = dst_path + '/' + str(len(lead_list))
        if not os.path.exists(dst_path_tmp):
            os.makedirs(dst_path_tmp)
        
        for subdir_name in subdir_names:
            
            subdir_name_save = subdir_name.replace(src_path,dst_path_tmp)
            
            if not os.path.exists(subdir_name_save):
                os.makedirs(subdir_name_save)
            
    
    
    filenames  = [name.replace('.mat','') for name in glob(src_path + r"/**/*.mat", recursive=True)]
    
    
    src_paths = [src_path for filename in filenames]
    dst_paths = [dst_path for filename in filenames]



    # with Pool() as pool:
    #     pool.starmap(resave_one, zip(filenames,src_paths,dst_paths))
        
    # starmap(resave_one, zip(filenames,src_paths,dst_paths))
    for filename,src_path,dst_path in zip(filenames,src_paths,dst_paths):
        resave_one(filename,src_path,dst_path)
    
        
        



if __name__ == '__main__':
    
    if os.path.isdir('../data'):
        src_path = '../data'
    if os.path.isdir('../../../cardio_shared/data'):    
        src_path = '../../../cardio_shared/data'
        
    dst_path = '../data_resave'
    
    resave_data(src_path,dst_path)