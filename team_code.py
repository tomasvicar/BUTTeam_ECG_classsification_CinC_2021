import os
import logging
import nvidia_smi
from glob import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt
from shutil import copyfile

from config import Config
from resave_data import resave_data
from utils.utils import get_gpu_memory
from utils.utils import get_lr
from utils.get_data import get_data
from utils.train_valid_split import train_valid_split
from utils.get_wce_weights import get_wce_weights
from dataset import Dataset
import net
from utils.log import Log
from utils.adjustLearningRateAndLoss import AdjustLearningRateAndLoss
from utils.compute_challenge_metric_custom import compute_challenge_metric_custom
from utils.optimize_ts import optimize_ts
from run_model_fcn import run_model_fcn

def training_code(data_directory, model_directory):
    
    
    if not os.path.isdir(model_directory + '/2'):
        resave_data(data_directory,model_directory)
    
    if not os.path.isdir(model_directory + '/models'):
        os.makedirs(model_directory + '/models')
        
        
    for lead_list in Config.LEAD_LISTS:
        
        train_one_model(model_directory,lead_list)
        
        
        
def train_one_model(model_directory,lead_list):
    
    LR_LIST = Config.LR_LIST
    LR_CHANGES_LIST = Config.LR_CHANGES_LIST
    LOSS_FUNTIONS = Config.LOSS_FUNTIONS
    MAX_EPOCH = Config.MAX_EPOCH
    WEIGHT_DECAY = Config.WEIGHT_DECAY
    
    try:
        nvidia_smi.nvmlInit()
        measured_gpu_memory = []
        measured_gpu_memory.append(get_gpu_memory(nvidia_smi))
    except:
        measured_gpu_memory = []
    

    file_names = glob(model_directory + '/' + str(len(lead_list)) + "/**/*.npy", recursive=True)
    
    names_onehot_lens = get_data(file_names)
    
    
    
    def filter_fcn(name):
        name = os.path.split(name)[1]
        
        return name.startswith('A') or name.startswith('Q') or name.startswith('E') or name.startswith('S') or name.startswith('HR')

    names_onehot_lens_train_all,names_onehot_lens_valid_all = train_valid_split(names_onehot_lens,Config.MODELS_SEED,Config.SPLIT_RATIO)
    
    
    names_onehot_lens_train = list(filter(lambda x : x.len <= (Config.MAX_LEN * Config.Fs), names_onehot_lens_train_all))
    names_onehot_lens_valid = list(filter(lambda x : x.len <= (Config.MAX_LEN * Config.Fs), names_onehot_lens_valid_all))
    
    names_onehot_lens_train = list(filter(lambda x : filter_fcn(x.name) , names_onehot_lens_train))###############♣
    names_onehot_lens_valid = list(filter(lambda x : filter_fcn(x.name) , names_onehot_lens_valid))###############♣
    
    
    
    lens_all = [item.len for item in names_onehot_lens_train]
    
    w_pos,w_neg = get_wce_weights(names_onehot_lens_train)
    
    
    w_pos[w_pos>100] = 100
    

    training_set = Dataset( names_onehot_lens_train,transform_nonrep=Config.TRANSFORM_DATA_TRAIN_NONREP,transform_rep=Config.TRANSFORM_DATA_TRAIN_REP)
    training_generator = DataLoader(training_set,batch_size=Config.BATCH,num_workers=Config.NUM_WORKERS_TRAIN,
                                         shuffle=True,drop_last=True,collate_fn=Dataset.pad_collate )
    
    
    validation_set = Dataset(names_onehot_lens_valid,transform_nonrep=Config.TRANSFORM_DATA_VALID_NONREP,transform_rep=Config.TRANSFORM_DATA_VALID_REP)
    validation_generator = DataLoader(validation_set,batch_size=Config.BATCH,num_workers=Config.NUM_WORKERS_VALID,
                                           shuffle=False,drop_last=False,collate_fn=Dataset.pad_collate )
    
    
    model = net.Net_addition_grow(input_size=len(lead_list),
                                  output_size=Config.OUTPUT_SIZE,
                                  levels=Config.LEVELS,
                                  lvl1_size=Config.LVL1_SIZE,
                                  blocks_in_lvl=Config.BLOCKS_IN_LVL,
                                  convs_in_layer=Config.CONVS_IN_LAYER,
                                  filter_size=Config.FILTER_SIZE,
                                  do = Config.DO
                                  )
    
              
    train_names = [item.name for item in names_onehot_lens_train_all]
    valid_names = [item.name for item in names_onehot_lens_valid_all]
    model.save_filename_train_valid(train_names,valid_names)
    model = model.to(Config.DEVICE)
                        
    
    optimizer = optim.AdamW(model.parameters(),lr = LR_LIST[0] ,betas= (0.9, 0.999),eps=1e-5,weight_decay=WEIGHT_DECAY)
    
    scheduler = AdjustLearningRateAndLoss(optimizer,LR_LIST,LR_CHANGES_LIST,LOSS_FUNTIONS)
    
    log=Log(['loss','challange_metric'])
    
    for epoch in range(MAX_EPOCH):
        
        model.train()
        N=len(training_generator)
        for it,(pad_seqs,lbls,lens) in enumerate(training_generator):
            
            if (it % 50) == 0:
                print(str(it) + '/' + str(N))
                
                
            type_ = 'train'
            
            pad_seqs = pad_seqs.to(Config.DEVICE)
            lens = lens.to(Config.DEVICE)
            lbls = lbls.to(Config.DEVICE)
            
            
            res=model(pad_seqs,lens)
            
            loss=scheduler.actual_loss(res,lbls,w_pos,w_neg)
            
            if type_ == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            loss=loss.detach().cpu().numpy()
            res=res.detach().cpu().numpy()
            lbls=lbls.detach().cpu().numpy()
            lens=lens.detach().cpu().numpy()
            
            challange_metric=compute_challenge_metric_custom(res>0.5,lbls)
            
            if type_ == 'train':
                log.append_train([loss,challange_metric])
            else:
                log.append_test([loss,challange_metric])
            
            
            
            try:
                measured_gpu_memory.append(get_gpu_memory(nvidia_smi))
            except:
                 measured_gpu_memory.append(0)
        
        model.eval() 
        res_all=[]
        lbls_all=[]
        N=len(validation_generator)
        with torch.no_grad():
            for it,(pad_seqs,lbls,lens) in enumerate(validation_generator):
                
                if (it % 50) == 0:
                    print(str(it) + '/' + str(N))
                    
                type_ = 'valid'
                
                
                pad_seqs = pad_seqs.to(Config.DEVICE)
                lens = lens.to(Config.DEVICE)
                lbls = lbls.to(Config.DEVICE)
                
                
                res=model(pad_seqs,lens)
                
                loss=scheduler.actual_loss(res,lbls,w_pos,w_neg)
                
                if type_ == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                loss=loss.detach().cpu().numpy()
                res=res.detach().cpu().numpy()
                lbls=lbls.detach().cpu().numpy()
                lens=lens.detach().cpu().numpy()
                
                challange_metric=compute_challenge_metric_custom(res>0.5,lbls)
                
                if type_ == 'train':
                    log.append_train([loss,challange_metric])
                else:
                    log.append_test([loss,challange_metric])
                
                
                
                lbls_all.append(lbls)
                res_all.append(res)
                
                try:
                    measured_gpu_memory.append(get_gpu_memory(nvidia_smi))
                except:
                     measured_gpu_memory.append(0)
        
        
        model.save_lens(np.stack(lens_all,axis=0))
        
        if epoch>=(MAX_EPOCH-10):
            ts,opt_challenge_metric=optimize_ts(np.concatenate(res_all,axis=0),np.concatenate(lbls_all,axis=0)) 
        else:
            ts,opt_challenge_metric=optimize_ts(np.concatenate(res_all,axis=0),np.concatenate(lbls_all,axis=0),fast=True) 
            
        model.set_ts(ts)
        log.save_opt_challange_metric_test(opt_challenge_metric)
        
        
        
        log.save_and_reset()
        
        lr = get_lr(optimizer)
        
        
        xstr = lambda x:"{:.4f}".format(x)
        info ='model' + str(len(lead_list)) + '_'  + str(epoch) 
        info += '_' + str(lr)  + '_gpu_' + xstr(np.max(measured_gpu_memory)) + '_trainCM_'  + xstr(log.train_log['challange_metric'][-1]) 
        info +='_validCM_' + xstr(log.test_log['challange_metric'][-1]) + '_validoptCM_' + xstr(log.opt_challange_metric_test[-1]) 
        info += '_trainLoss_'  + xstr(log.train_log['loss'][-1]) + '_validLoss_' + xstr(log.test_log['loss'][-1])

        
        print(info)
        
        model_name = model_directory + '/models/' + Config.MODEL_NOTE + info  
        log.save_log_model_name(model_name + '.pt')
        model.save_log(log)
        model.save_config(Config)
        torch.save(model,model_name + '.pt')
            
        log.plot(model_name)
        
        scheduler.step()
        
        
    best_model_name=log.model_names[np.argmax(log.opt_challange_metric_test)]
        
    copyfile(best_model_name,model_directory +'/final_model' + str(len(lead_list))  + '.pt')
        
        
        
        
                
                


def load_model(model_directory, leads):
    
    filename = os.path.join(model_directory, 'final_model' + str(len(leads)) + '.pt')
    
    device = torch.device("cuda:"+str(torch.cuda.current_device()))
     
    return torch.load(filename,map_location=device)

def run_model(model, header, recording):
    return run_model_fcn(model, header, recording)




if __name__ == '__main__':
    
    data_directory = Config.DATA_PATH
    model_directory = Config.DATA_RESAVE_PATH
    
    training_code(data_directory, model_directory)

    
    

    # logging.basicConfig(filename='debug.log',level=logging.INFO)
    # try:
    #     data_directory = Config.DATA_PATH
    #     model_directory = Config.DATA_RESAVE_PATH
        
        
    #     training_code(data_directory, model_directory)


    # except Exception as e:
    #     print(e)
    #     logging.critical(e, exc_info=True)
        
        
    
    
    
    
    
    