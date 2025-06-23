# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:57:19 2024

@author: Xin Dong
"""

import torch
# import vaex
import h5py
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader


# class DataGenerator(Dataset):
#     """Creates Dataset and Picks up a SINGLE Random Sample from Dataset"""

#     def __init__(self, path,norm_H=False):
#         self.norm = False

#         self.data = loadmat(path)

#         # self.xdata_complex = self.data['H_est_MMSE_all_data'].transpose(2,1,0)
#         # self.xdata_complex = self.data['H_est_LS_all'].transpose(2,1,0)
#         # self.ydata_complex = self.data['H_all'].transpose(2,1,0)

#         self.xdata_complex = self.data['H_est_LS_all_data'].transpose(2,1,0)
#         self.ydata_complex = self.data['H_all_data'].transpose(2,1,0)
        
#         # self.x_data_real_mean,self.x_data_real_std,self.x_data_imag_mean,self.x_data_imag_std = self.__compute_mean_std__()
        
        
#     def __len__(self):
#         return len(self.xdata_complex)

#     def __getitem__(self, index):

#         index = index%self.__len__()
        
#         x_data = self.xdata_complex[index]
#         y_data = self.ydata_complex[index]
        
#         # x_data = np.concatenate([(x_data.real[np.newaxis,:,:]-self.x_data_real_mean)/self.x_data_imag_std,(x_data.imag[np.newaxis,:,:]-self.x_data_imag_mean)/self.x_data_imag_std],axis=0)
#         # y_data = np.concatenate([y_data.real[np.newaxis,:,:],y_data.imag[np.newaxis,:,:]],axis=0)

#         x_data = np.concatenate([x_data.real[np.newaxis,:,:],x_data.imag[np.newaxis,:,:]],axis=0)
#         y_data = np.concatenate([y_data.real[np.newaxis,:,:],y_data.imag[np.newaxis,:,:]],axis=0)

#         x_data = np.array(x_data,dtype=np.float32)
#         y_data =  np.array(y_data,dtype=np.float32)
        
#         if np.all(x_data == 0) or np.all(y_data == 0):  # 如果 xdata 或 ydata 全为 0
#             return self.__getitem__(index + 1)
        
#         return x_data,y_data

#     def __compute_mean_std__(self):
#         x_data_real_mean = np.mean(self.xdata_complex.real,axis=0)
#         x_data_real_std = np.std(self.xdata_complex.real,axis=0)
#         x_data_imag_mean = np.mean(self.xdata_complex.imag,axis=0)
#         x_data_imag_std = np.std(self.xdata_complex.imag,axis=0)
#         return x_data_real_mean,x_data_real_std,x_data_imag_mean,x_data_imag_std

class DataGenerator2(Dataset):
    """Creates Dataset and Picks up a SINGLE Random Sample from Dataset"""

    def __init__(self, path, norm_H=False):
        self.norm = False
        self.norm_H = norm_H


        self.data = h5py.File(path, 'r') 
        self.V_complex = self.data['V_pinv_LS_all_data'][:]
        
        self.xdata_complex = self.data['H_est_LS_all_data'][:]
        self.ydata_complex = self.data['H_all_data'][:]
        
        # self.x_data_real_mean,self.x_data_real_std,self.x_data_imag_mean,self.x_data_imag_std = self.__compute_mean_std__()
        
        
    def __len__(self):
        return len(self.xdata_complex)

    def __getitem__(self, index):

        index = index%self.__len__()
        
        x_data = self.xdata_complex[index]
        y_data = self.ydata_complex[index]
        
        
        x_data_real = np.array(x_data['real'],dtype=np.float32)[np.newaxis,:,:]
        x_data_imag = np.array(x_data['imag'],dtype=np.float32)[np.newaxis,:,:]
        y_data_real = np.array(y_data['real'],dtype=np.float32)[np.newaxis,:,:]
        y_data_imag = np.array(y_data['imag'],dtype=np.float32)[np.newaxis,:,:]

        x_data = np.concatenate([x_data_real,x_data_imag],axis=0)
        y_data = np.concatenate([y_data_real,y_data_imag],axis=0)
    
        Vpinv_data = self.V_complex[index]
        Vpinv_data_real = np.array(Vpinv_data['real'],dtype=np.float32)[np.newaxis,:,:]
        Vpinv_data_imag = np.array(Vpinv_data['imag'],dtype=np.float32)[np.newaxis,:,:]
        Vpinv_data =  np.concatenate([Vpinv_data_real,Vpinv_data_imag],axis=0)
        
        return x_data,y_data,Vpinv_data
        
