#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:01:07 2019

@author: rashmi
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
class TDFbanks(nn.Module):
    def __init__(self):
        super(TDFbanks, self).__init__()
        wlen=25
        wstride=10
        samplerate=8000
        nfilters=40
        window_size = samplerate * wlen // 1000 + 1
        window_stride = samplerate * wstride // 1000
        padding_size = (window_size - 1) // 2
        self.preemp= nn.Conv1d(1, 1, 2, 1, padding=1, groups=1, bias=False)
        self.complex_conv=nn.Conv1d(1, 2 * nfilters, window_size, 1,
            padding=padding_size, groups=1, bias=False)
        self.lowpass = nn.Conv1d(nfilters, nfilters, window_size, window_stride,
            padding=0, groups=nfilters, bias=False)
        self.instancenorm = nn.InstanceNorm1d(nfilters, momentum=1)
    def forward(self, x):
        x = self.preemp(x)
        # Complex convolution
        x = self.complex_conv(x)
        # Squared modulus operator
        x = x.transpose(1, 2)
        x = F.avg_pool1d(x.pow(2), 2, 2, 0, False).mul(2)
        x = x.transpose(1, 2)
        x = self.lowpass(x)
        x = x.abs()
        x = x + 1
        x = x.log()
        x = self.instancenorm(x)
        return x
    
class XVECTORmodel(nn.Module):
    def __init__(self,batch_size,in_channels):
        # Main parameters
        super(XVECTORmodel, self).__init__()
        
        self.batch_size=batch_size
        self.in_channels=in_channels
        self.tdfbanks = TDFbanks()
        self.cnn1=torch.nn.Conv1d(in_channels=self.in_channels,out_channels=500,kernel_size=5,stride=1,padding=0)
        self.cnn2=torch.nn.Conv1d(in_channels=500,out_channels=500,kernel_size=7,stride=2,padding=0)
        self.cnn3=torch.nn.Conv1d(in_channels=500,out_channels=500,kernel_size=1,stride=1,padding=0)
        self.cnn4=torch.nn.Conv1d(in_channels=500,out_channels=3000,kernel_size=1,stride=1,padding=0)
        self.relu=nn.ReLU()
        self.linear1=nn.Linear(3000,1500)
        self.linear2=nn.Linear(1500,600)
        self.linear3=nn.Linear(600,8)
        self.drop = nn.Dropout(p=0.51)
        self.sigmoid=nn.Sigmoid()          
        
    def forward(self,data):
        output1=self.tdfbanks.forward(data)
        output1=self.drop(self.relu(self.cnn1(output1)))
        output1=self.drop(self.relu(self.cnn4(output1)))
        output1=torch.mean(output1,2)
        output1=self.relu(self.linear1(output1))
        output1=self.relu(self.linear2(output1))
        output1=self.linear3(output1)
        return F.log_softmax(output1,dim=-1)
    
#%%
import sys
batch=1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
xvector_model=XVECTORmodel(batch,40).to(device)
loss_file=sys.argv[1]
#xvector_modeltorch.load(loss_file)
checkpoint=torch.load(loss_file)
xvector_model.load_state_dict(checkpoint['state_dict'])
xvector_model.eval()
from HDF5Dataset import HDF5Dataset
from torch.utils import data
test_file='../test_h5_wav_data/'
loader_params = {'num_workers': 1}
#%%
dataset_eval = HDF5Dataset(test_file, recursive=False, load_data=False,data_cache_size=4, transform=None)
dataloader_test = data.DataLoader(dataset_eval,**loader_params)
test_len=dataset_eval.__len__()
print(test_len)
with torch.no_grad():
    predictions=np.zeros(test_len)
    labels_set=np.zeros(test_len)
    ind=0
    for test_data,labels in dataloader_test:
         #wavs,label=read_all(test_files[ind],classes)
         labels_set[ind]=labels[0][0]
         test_data = test_data.to(device)
         test_data = test_data.view(1, 1, -1)
         prediction=xvector_model.forward(test_data)
         predictions[ind]=torch.argmax(prediction)
         ind+=1
    from sklearn.metrics import classification_report
    print("========================test=======================")
    print(classification_report(labels_set, predictions,digits=5))
#torch.save(xvector_model.state_dict(),'cv_atten_analysis_initial_fbank.model')
#print("saved model")


