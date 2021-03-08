import os
from glob import glob
from multiprocessing import Pool
from shutil import copyfile
from shutil import rmtree

from test_model import test_model
from config import Config
from team_code import load_twelve_lead_model
from extract_leads_wfdb_custom import extract_leads_wfdb_custom


def resave_valid_data(data_path,resave_path,train_names):
    
    if os.path.isdir(resave_path):
        rmtree(resave_path)
    
    if not os.path.isdir(resave_path):
        os.makedirs(resave_path)
    
    
    all_names = glob(data_path + "/**/*.mat", recursive=True)
    all_names_parts = [os.path.split(name)[1].replace('.mat','') for name in all_names]
    
    inds = [k for k,x in enumerate(all_names_parts) if x not in train_names]
        
    filenames_mat_use = [all_names[k] for k in inds]
    
    data_paths = [data_path for filename in filenames_mat_use]
    resave_paths = [resave_path for filename in filenames_mat_use]


    with Pool() as pool:
        pool.starmap(resave_one, zip(filenames_mat_use,data_paths,resave_paths))
    
    
def resave_one(filename_mat,data_path,resave_path):
    
    save_filename_mat = resave_path + '/' + os.path.split(filename_mat)[1]
    copyfile(filename_mat,save_filename_mat)
    copyfile(filename_mat.replace('.mat','.hea'),save_filename_mat.replace('.mat','.hea'))
    
                    

    
if __name__ == '__main__':

    
    model = load_twelve_lead_model(Config.DATA_RESAVE_PATH)
    
    train_names = [os.path.split(name)[1].split('-')[0] for name in model.train_names]
    

    resave_valid_data(Config.DATA_PATH,Config.DATA_RESAVE_PATH + '/test_data12',train_names)
    
    for leads in Config.LEAD_LISTS:
        if len(leads) != 12:
            data_directory = Config.DATA_RESAVE_PATH  + '/test_data' + str(len(leads)) 
            
            src_dir = Config.DATA_RESAVE_PATH + '/test_data12'
            
            extract_leads_wfdb_custom(src_dir,data_directory,leads)
    
    
    model_directory = Config.DATA_RESAVE_PATH
    
    
        
        
    for leads in Config.LEAD_LISTS:
        
        output_directory = Config.DATA_RESAVE_PATH + '/output' + str(len(leads))  
    
        if os.path.isdir(output_directory):
            rmtree(output_directory)
            
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        data_directory = Config.DATA_RESAVE_PATH  + '/test_data' + str(len(leads))  

        test_model(model_directory, data_directory, output_directory)
        
        
        
        
        