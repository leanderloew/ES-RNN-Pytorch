#Dataset with random shuffeling: We have to check, that seasonality gets shifted apropriatly 

import numpy as np 
import pandas as pd

import random
from torch.utils.data import Dataset,sampler,DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm


class sequence_labeling_dataset(Dataset):
    
    def __init__(self, input,max_size=100,sequence_labeling=True,seasonality=12,out_preds=12):     
        
        self.data=input
        self.max_size=max_size
        self.sequence_labeling=sequence_labeling
        self.seasonality=seasonality
        self.out_preds=out_preds
        
    def __len__(self):
        
        return int(10000)
    
    def __getitem__(self, index):
        
        data_i=self.data
        
        #we randomly shift the inputs to create more data
        if len(data_i)>self.max_size:
            max_rand_int=len(data_i)-self.max_size
            #take a random start integer
            start_int=random.randint(0,max_rand_int)
            data_i=data_i[start_int:(start_int+self.max_size)]
        else:
            start_int=0

        
        inp=np.array(data_i[:-self.out_preds])
        
        
        if self.sequence_labeling==True:
            #in case of sequence labeling, we shift the input by the range to output
            out=np.array(data_i[self.out_preds:])
        else:
            #in case of sequnec classification we return only the last n elements we
            #need in the forecast
            out=np.array(data_i[-self.out_preds:])
            
        #This defines, how much we have to shift the season 
        shift_steps=start_int%self.seasonality
        
        return inp, out,shift_steps
    