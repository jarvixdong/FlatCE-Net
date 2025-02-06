# -*- coding: utf-8 -*-
"""
Author: Xin Dong
 
Date: 2024-09-01
"""

import torch
from .dataset import *
from torch.utils.data import Dataset, DataLoader

class BaseBunch():
    """BaseBunch:(trainset,[valid]).
    """
    def __init__(self, trainset, valid=None, batch_size=512, shuffle=False, num_workers=0, pin_memory=False, drop_last=True,
                 round_up=False):

        num_samples = len(trainset)
        print('num samples ::',num_samples)
        

        self.train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                                        pin_memory=pin_memory, drop_last=drop_last)
                
        self.num_batch_train = len(self.train_loader)
        
        if valid is not None:
            print('num valid:',len(valid))
        
            valid_batch_size = min(batch_size, len(valid)) # To save GPU memory

            if len(valid) <= 0:
                raise ValueError("Expected num_samples of valid > 0.")

            self.valid_loader = DataLoader(valid, batch_size = valid_batch_size, shuffle=False, num_workers=num_workers, 
                                            pin_memory=pin_memory, drop_last=False)

            self.num_batch_valid = len(self.valid_loader)
        
        else:
            self.valid_loader = None
            self.num_batch_valid = 0
        
    @classmethod
    def get_dataset(self,cfg_dataset, data_loader_params_dict:dict={}):
        train_set = DataGenerator2(path=cfg_dataset.train_path,with_Vpinv=cfg_dataset.with_Vpinv)
        valid_set = DataGenerator2(path=cfg_dataset.valid_path,with_Vpinv=cfg_dataset.with_Vpinv)
        return self(train_set,valid_set, **data_loader_params_dict)
    
    @classmethod
    def load_data(self,cfg_dataset,cfg_dataloader):
        data_loader_params_dict = {
            "batch_size": cfg_dataloader.get("batch_size", 128), 
            "shuffle": cfg_dataloader.get("shuffle", True), 
            "num_workers": cfg_dataloader.get("num_workers", 0),
        }
        
        bunch = self.get_dataset(cfg_dataset,data_loader_params_dict=data_loader_params_dict)
        return bunch
    
    def __len__(self):
        return self.num_batch_train