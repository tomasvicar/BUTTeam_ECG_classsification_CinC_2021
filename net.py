import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
import matplotlib.pyplot as plt
import os
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x







class myAttention(nn.Module):
    
    def __init__(self,in_size, out_size,levels):
        super().__init__()
        
        self.levels = levels
        
        self.conv_tanh = nn.Conv1d(in_size, in_size, 1)
        self.conv_sigm = nn.Conv1d(in_size, in_size, 1)
        self.conv_w = nn.Conv1d(in_size, out_size, 1)
        
        self.conv_final = nn.Conv1d(in_size,out_size,1)
    
    def forward(self, inputs,remove_matrix):
        
        
        tanh = torch.tanh(self.conv_tanh(inputs))
        
        sigm = torch.sigmoid(self.conv_sigm(inputs))
        
        z = self.conv_w(tanh * sigm) 
        
        z[remove_matrix.repeat(1,list(z.size())[1],1)==1] = -np.Inf
        
        a = torch.softmax(z,dim=2)
        
        
        output = self.conv_final(inputs)
        output = output * a
        
        a2 = output
        
        output = torch.sum(output,dim=2)
        

        return output,a,a2






class myConv(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=1,pad=None,do_batch=1,dov=0):
        super().__init__()
        
        pad=int((filter_size-1)/2)
        
        self.do_batch=do_batch
        self.dov=dov
        self.conv=nn.Conv1d(in_size, out_size,filter_size,stride,pad,bias=False)
        self.bn=nn.BatchNorm1d(out_size,momentum=0.1)
        
        
        if self.dov>0:
            self.do=nn.Dropout(dov)
            
    def swish(self,x):
        return x * F.sigmoid(x)
    
    def forward(self, inputs):
     
        outputs = self.conv(inputs)
        if self.do_batch:
            outputs = self.bn(outputs)  
        
        outputs=F.relu(outputs)
        # outputs=self.swish(outputs)
        
        
        if self.dov>0:
            outputs = self.do(outputs)
            
            
            
        
        return outputs


        
class Net_addition_grow(nn.Module):
    def set_ts(self,ts):
        self.ts=ts
        
    def get_ts(self):
        return self.ts
    
    
    def __init__(self, levels=7,lvl1_size=4,input_size=12,output_size=24,convs_in_layer=3,init_conv=4,filter_size=13,nhid = 2048,nlayers = 8,nhead = 8):
        super().__init__()
        self.levels=levels
        self.lvl1_size=lvl1_size
        self.input_size=input_size
        self.output_size=output_size
        self.convs_in_layer=convs_in_layer
        self.filter_size=filter_size
        
        self.get_atttention = 0
        
        
        self.init_conv=myConv(input_size,init_conv,filter_size=filter_size)
        
        self.layers=nn.ModuleList()
        for lvl_num in range(self.levels):
            
            
            if lvl_num==0:
                self.layers.append(myConv(init_conv, int(lvl1_size*(lvl_num+1)),filter_size=filter_size))
            else:
                self.layers.append(myConv(int(lvl1_size*(lvl_num))+int(lvl1_size*(lvl_num))+init_conv, int(lvl1_size*(lvl_num+1)),filter_size=filter_size))
            
            for conv_num_in_lvl in range(self.convs_in_layer-1):
                self.layers.append(myConv(int(lvl1_size*(lvl_num+1)), int(lvl1_size*(lvl_num+1)),filter_size=filter_size))


        self.conv_final=myConv(int(lvl1_size*(self.levels))+int(lvl1_size*(self.levels))+init_conv, int(lvl1_size*self.levels),filter_size=filter_size)
        
        
        self.pos_enc = PositionalEncoding(int(lvl1_size*self.levels))
        
        
        
        encoder_layers = TransformerEncoderLayer(int(lvl1_size*self.levels), nhead, nhid, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        
        self.attention = myAttention(int(lvl1_size*self.levels),output_size,self.levels)
        
        
        
        
        ## weigths initialization wih xavier method
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                init.xavier_normal_(m.weight)
                if not m.bias == None:
                    init.constant_(m.bias, 0)
        
        
        
    def forward(self, x,lens):
        
        
        
        
        shape = list(x.size())
        remove_matrix = torch.ones((shape[0],1,shape[2]),dtype=x.dtype)
        cuda_check = x.is_cuda
        if cuda_check:
            cuda_device = x.get_device()
            device = torch.device('cuda:' + str(cuda_device) )
            remove_matrix = remove_matrix.to(device)
        
        

        
        for signal_num in range(list(x.size())[0]):
            
            k = int(lens[signal_num])
            
            remove_matrix[signal_num,:,:k] = 0
        
        
        
        x=self.init_conv(x)
        x = x * (1 - remove_matrix)
        
        x0=x
        
        ## aply all convolutions
        layer_num=-1
        for lvl_num in range(self.levels):
            
            
            for conv_num_in_lvl in range(self.convs_in_layer):
                layer_num+=1
                if conv_num_in_lvl==1:
                    y=x
                
                x=self.layers[layer_num](x)
                x = x * (1 - remove_matrix)
                
            ## skip conection to previous layer and to the input
            x=torch.cat((F.avg_pool1d(x0,2**lvl_num,2**lvl_num),x,y),1)
            
            x = F.max_pool1d(x, 2, 2)
            remove_matrix = F.max_pool1d(remove_matrix, 2, 2)
            x = x * (1 - remove_matrix)
            
            
        x=self.conv_final(x)
        x = x * (1 - remove_matrix)

        x = x.permute(2, 0, 1)
        # remove_matrix = remove_matrix.permute(2, 0, 1)

        x = self.pos_enc(x)
        
        tmp_mask = remove_matrix[:,0,:]
        x = self.transformer_encoder(x,src_key_padding_mask = tmp_mask==1)
        
        x = x.permute(1, 2, 0)
        # remove_matrix = remove_matrix.permute(1, 2, 0)
        
        x, a1, a2 = self.attention(x,remove_matrix)
        

        x=torch.sigmoid(x)
        
        if self.get_atttention == 0:
            return x   
        else:
            return x, a1, a2
    
    def save_log(self,log):
        self.log=log
        
    def save_config(self,config):  
        self.config=config
        
    def save_lens(self,lens):
        self.lens=lens
        
    def save_filename_train_valid(self,train_names,valid_names):
        
        self.train_names = train_names
        
        self.valid_names = valid_names
        
