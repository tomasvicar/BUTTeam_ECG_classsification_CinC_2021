import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
import matplotlib.pyplot as plt
import os


# class inputAttention(nn.Module):
#     def __init__(self,out_size):
#         super().__init__()
#         self.out_size = out_size
        
#         self.conv1 = myConv(1,out_size,do_batch=0)
        
        
#         self.convs = nn.Sequential(myConv(out_size,out_size,do_batch=0),myConv(out_size,out_size,do_batch=0),myConv(out_size,out_size,do_batch=0))
        
        
#         self.simple_net = nn.Sequential(myConv(out_size,out_size,do_batch=0),myConv(out_size,out_size,do_batch=1),myConv(out_size,out_size,do_batch=1),nn.MaxPool1d(2,2),
#                                 myConv(out_size,out_size,do_batch=1),myConv(out_size,out_size,do_batch=1),myConv(out_size,out_size,do_batch=1),nn.MaxPool1d(2,2),
#                                 myConv(out_size,out_size,do_batch=1),myConv(out_size,out_size,do_batch=1),myConv(out_size,out_size,do_batch=1),nn.MaxPool1d(2,2),
#                                 myConv(out_size,out_size,do_batch=1),myConv(out_size,out_size,do_batch=1),myConv(out_size,out_size,do_batch=1),nn.MaxPool1d(2,2),
#                                 myConv(out_size,out_size,do_batch=1),myConv(out_size,out_size,do_batch=1),myConv(out_size,out_size,do_batch=1),nn.MaxPool1d(2,2),
#                                 myConv(out_size,out_size,do_batch=1),myConv(out_size,out_size,do_batch=1),myConv(out_size,1,do_batch=1))
        

#         self.squeeze = nn.AdaptiveAvgPool1d(1)
        
#         self.channel_encoding = nn.Parameter(0*torch.randn(12,out_size))
#         # self.channel_encoding.requires_grad = False
        
        
        
        
#     def forward(self, inputs):
        
#         att = []
#         input_new = []
#         for c in range(inputs.size(1)):
            
            
#             x = self.conv1(inputs[:,[c],:])
            
#             enc = self.channel_encoding[c,:].view([1,self.out_size,1])
#             x = x + enc
            
#             x = self.convs(x)
            
#             x[inputs[:,c,0].detach()==-np.inf,:,:] = -np.inf
            
#             input_new.append(x)
            
#             x = self.simple_net(x)
            
#             x = self.squeeze(x)
            
#             x[inputs[:,c,0].detach()==-np.inf,:,:] = -np.inf
            
#             att.append(x)
            
#         # input_new = inputs
#         input_new = torch.stack(input_new,2)
        
#         att = torch.cat(att,2)
#         att = torch.softmax(att,2)
        
#         att2 = att.view([att.size(0),att.size(1),att.size(2),1])
#         att2 = att2.repeat(1,input_new.size(1),1,inputs.size(2))
        
#         inputs2 = inputs.view([inputs.size(0),1,inputs.size(1),inputs.size(2)])
#         inputs2 = inputs2.repeat(1,input_new.size(1),1,1)
        
#         # input_new2 = input_new.view([inputs.size(0),input_new.size(2),inputs.size(1),inputs.size(2)])
#         # input_new2 = input_new2.repeat(1,input_new.size(2),1,1)
#         input_new2 = input_new
        
#         input_new2[inputs2.detach()==-np.inf] = 0
        
#         outputs = input_new2*att2
#         outputs = torch.sum(outputs,2)
        
#         print(torch.sum(outputs))
        
#         return outputs
        
        
            
# class inputAttention(nn.Module):
#     def __init__(self,out_size):
#         super().__init__()
#         self.out_size = out_size
        
#         self.conv1 = myConv(1,out_size,do_batch=0)
        
        
#         self.convs = nn.Sequential(myConv(out_size,out_size,do_batch=0),myConv(out_size,out_size,do_batch=0),myConv(out_size,out_size,do_batch=0))
        
        
#         self.simple_net = nn.Sequential(myConv(out_size,out_size,do_batch=0),myConv(out_size,out_size,do_batch=1),myConv(out_size,out_size,do_batch=1),nn.MaxPool1d(2,2),
#                                 myConv(out_size,out_size,do_batch=1),myConv(out_size,out_size,do_batch=1),myConv(out_size,out_size,do_batch=1),nn.MaxPool1d(2,2),
#                                 myConv(out_size,out_size,do_batch=1),myConv(out_size,out_size,do_batch=1),myConv(out_size,out_size,do_batch=1),nn.MaxPool1d(2,2),
#                                 myConv(out_size,out_size,do_batch=1),myConv(out_size,out_size,do_batch=1),myConv(out_size,out_size,do_batch=1),nn.MaxPool1d(2,2),
#                                 myConv(out_size,out_size,do_batch=1),myConv(out_size,out_size,do_batch=1),myConv(out_size,out_size,do_batch=1),nn.MaxPool1d(2,2),
#                                 myConv(out_size,out_size,do_batch=1),myConv(out_size,out_size,do_batch=1),myConv(out_size,1,do_batch=1))
        

#         self.squeeze = nn.AdaptiveAvgPool1d(1)
        
#         self.channel_encoding = nn.Parameter(0*torch.randn(12,out_size))
#         # self.channel_encoding.requires_grad = False
        
#     def forward(self, inputs):
        
#         outputs = []
#         for b in range(inputs.size(0)):
            
#             noninfs = inputs[b,:,0].detach().cpu().numpy()!=-np.inf
#             num_leads = np.sum(noninfs)
            
#             noninfs = np.nonzero(noninfs)[0]
            
#             tmp = inputs[[b],noninfs,:]
#             tmp = tmp.view([1,tmp.size(0),tmp.size(1)])
            
            
#             for c in range(tmp.size(1)):
                
                
                
                
            
            
#         outputs = torch.cat(outputs,0)
        
        
#         return outputs    
    

          
        
       

class inputConvolutionHeads(nn.Module):
    def __init__(self,out_size):
        super().__init__()
        self.out_size = out_size
        
        self.conv12 = myConv(12,out_size,do_batch=0)
        self.conv6 = myConv(6,out_size,do_batch=0)
        self.conv4 = myConv(4,out_size,do_batch=0)
        self.conv3 = myConv(3,out_size,do_batch=0)
        self.conv2 = myConv(2,out_size,do_batch=0)
        
        
        # self.channel_encoding.requires_grad = False
        
        
        
        
    def forward(self, inputs):

        
        

        outputs = []
        for b in range(inputs.size(0)):
            
            noninfs = inputs[b,:,0].detach().cpu().numpy()!=-np.inf
            num_leads = np.sum(noninfs)
            
            noninfs = np.nonzero(noninfs)[0]
            
            if  num_leads == 12:
                tmp = inputs[[b],noninfs,:]
                tmp = tmp.view([1,tmp.size(0),tmp.size(1)])
                outputs.append(self.conv12(tmp))
            elif  num_leads == 6:
                tmp = inputs[[b],noninfs,:]
                tmp = tmp.view([1,tmp.size(0),tmp.size(1)])
                outputs.append(self.conv6(tmp))
            elif  num_leads == 4:
                tmp = inputs[[b],noninfs,:]
                tmp = tmp.view([1,tmp.size(0),tmp.size(1)])
                outputs.append(self.conv4(tmp))    
            elif  num_leads == 3:
                tmp = inputs[[b],noninfs,:]
                tmp = tmp.view([1,tmp.size(0),tmp.size(1)])
                outputs.append(self.conv3(tmp))
            elif  num_leads == 2:
                tmp = inputs[[b],noninfs,:]
                tmp = tmp.view([1,tmp.size(0),tmp.size(1)])
                outputs.append(self.conv2(tmp))
            else:
                raise
                
                
                
            
            
        outputs = torch.cat(outputs,0)
        
        
        return outputs    
       
        
            
            
            
            
        
        


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
    def __init__(self, in_size, out_size,filter_size=3,stride=1,pad=None,do_batch=1,dov=None):
        super().__init__()
        
        pad=int((filter_size-1)/2)
        
        self.do_batch=do_batch
        self.dov=dov
        self.conv=nn.Conv1d(in_size, out_size,filter_size,stride,pad,bias=False)
        if self.do_batch:
            self.bn=nn.BatchNorm1d(out_size,momentum=0.1)
        
        
        if self.dov:
            self.do=nn.Dropout(dov)
            
    def swish(self,x):
        return x * F.sigmoid(x)
    
    def forward(self, inputs):
     
        outputs = self.conv(inputs)
        if self.do_batch:
            outputs = self.bn(outputs)  
        
        outputs=F.relu(outputs)
        # outputs=self.swish(outputs)
        
        
        if self.dov:
            outputs = self.do(outputs)
            
            
            
        
        return outputs


        
class Net_addition_grow(nn.Module):
    def set_ts(self,ts):
        self.ts=ts
        
    def get_ts(self):
        return self.ts
    
    
    def __init__(self,input_size=12,output_size=24,levels=7,lvl1_size=6,blocks_in_lvl=3,convs_in_layer=2,filter_size=7,do=None):
        super().__init__()
        self.levels=levels
        self.lvl1_size=lvl1_size
        self.input_size=input_size
        self.output_size=output_size
        self.convs_in_layer=convs_in_layer
        self.filter_size=filter_size
        self.blocks_in_lvl=blocks_in_lvl
        self.do = do
        
        init_conv=lvl1_size
        
        self.get_atttention = 0
        
        
        # self.init_conv=myConv(input_size,init_conv,filter_size=filter_size)
        
        # self.inputAttention = inputAttention(lvl1_size)
        
        self.inputAttention = inputConvolutionHeads(lvl1_size)
        
        self.layers=nn.ModuleList()
        for lvl_num in range(self.levels):
            
            for block_num in range(self.blocks_in_lvl):
            
                if block_num==0 and lvl_num>0:
                    self.layers.append(myConv(int(lvl1_size*(lvl_num)), int(lvl1_size*(lvl_num+1)),filter_size=1,dov=do))
                    
                    self.layers.append(myConv(int(lvl1_size*(lvl_num)), int(lvl1_size*(lvl_num+1)),filter_size=filter_size,dov=do))
                else:
                    self.layers.append(myConv(int(lvl1_size*(lvl_num+1)), int(lvl1_size*(lvl_num+1)),filter_size=filter_size,dov=do))
                
                for conv_num_in_lvl in range(self.convs_in_layer-1):
                    self.layers.append(myConv(int(lvl1_size*(lvl_num+1)), int(lvl1_size*(lvl_num+1)),filter_size=filter_size,dov=do))


        self.conv_final = nn.Conv1d(int(lvl1_size*(self.levels)), int(lvl1_size*self.levels),kernel_size=filter_size,padding=1)
        
        self.attention = myAttention(int(lvl1_size*self.levels),output_size,self.levels)
        
        
        self.conv_fc = nn.Conv1d(int(lvl1_size*self.levels),output_size,kernel_size=1)
        
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
        
        
        
        x=self.inputAttention(x)
        x = x * (1 - remove_matrix)
        
        
        ## aply all convolutions
        layer_num=-1
        for lvl_num in range(self.levels):
            for block_num in range(self.blocks_in_lvl):
            
                y = x
                
                if block_num ==0 and lvl_num>0:
                    
                    layer_num+=1
                    y=self.layers[layer_num](y)
                    y = y * (1 - remove_matrix)
                    
                
                for conv_num_in_block in range(self.convs_in_layer):
                    
                    layer_num+=1
                    x=self.layers[layer_num](x)
                    x = x * (1 - remove_matrix)
                    
                ## skip conection to previous layer and to the input
                x = x + y
            
            x = F.max_pool1d(x, 2, 2)
            remove_matrix = F.max_pool1d(remove_matrix, 2, 2)
            x = x * (1 - remove_matrix)
            
            
        x=self.conv_final(x)
        x = x * (1 - remove_matrix)

            
        
        
        # x, a1, a2 = self.attention(x,remove_matrix)
        
        
        x[remove_matrix.repeat(1,list(x.size())[1],1)==1] = -np.Inf
        
        
        x = F.adaptive_max_pool1d(x, 1)
        x = self.conv_fc(x)
        a1 = 1
        a2 = 1
        x = x.view(list(x.size())[:2])
        
        
        
        
        
        

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