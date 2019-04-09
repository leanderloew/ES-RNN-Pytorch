import numpy as np 
import matplotlib.pyplot as plt

import random
from torch.utils.data import Dataset,sampler,DataLoader
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

class holt_winters_no_trend(torch.nn.Module):
    
    def __init__(self,init_a=0.1,init_g=0.1,slen=12):
      
        super(holt_winters_no_trend, self).__init__()
        
        #Smoothing parameters
        self.alpha=torch.nn.Parameter(torch.tensor(init_a))
        self.gamma=torch.nn.Parameter(torch.tensor(init_g))
        
        #init parameters
        self.init_season=torch.nn.Parameter(torch.tensor(np.random.random(size=slen)))
        
        #season legnth used to pick appropriate past season step 
        self.slen=slen
        
        #Sigmoid used to norm the params to be betweeen 0 and 1 if needed 
        self.sig=nn.Sigmoid()
        
    def forward(self,series ,series_shifts,n_preds=12,rv=False):
        
        #Get Batch size
        batch_size=series.shape[0]
        
        #Get the initial seasonality parameter
        init_season_batch=self.init_season.repeat(batch_size).view(batch_size,-1)
        
        #We use roll to Allow for our random input shifts.
        seasonals=torch.stack([torch.roll(j,int(rol)) for j,rol in zip(init_season_batch,series_shifts)]).float()
        
        #It has to be a list such that we dont need inplace tensor changes. 
        seasonals=list(torch.split(seasonals,1,dim=1))
        seasonals=[x.squeeze() for x in seasonals]
        
        #Now We loop over the input in each forward step
        result = []
        
        #rv can be used for decomposing a series./normalizing in case of ES-RNN
        if rv==True:
            value_list=[]
            season_list=[]
        

        for i in range(series.shape[1]+n_preds):
            
            #0th step we init the parameter 
            if i == 0: 

                smooth = series[:,0]
                value_list.append(smooth)
                season_list.append(seasonals[i%self.slen])                
                result.append(series[:,0])

                continue

            #smoothing
            #its smaller here, so smoothing is only for one less than the input? 
            if i <series.shape[1]:

                val = series[:,i]

                last_smooth, smooth = smooth, self.sig(self.alpha)*(val-seasonals[i%self.slen]) + (1-self.sig(self.alpha))*(smooth)

                seasonals[i%self.slen] = self.sig(self.gamma)*(val-smooth) + (1-self.sig(self.gamma))*seasonals[i%self.slen]
                
                #we store values, used for normaizing in ES RNN 
                if rv==True:
                    value_list.append(smooth)
                    season_list.append(seasonals[i%self.slen])

                result.append(smooth+seasonals[i%self.slen])
            
            #forecasting would jsut select last smoothed value and the appropriate seasonal, we will do this seperately 
            #in the ES RNN implementation
            else:
    
                m = i - series.shape[1] + 1

                result.append((smooth ) + seasonals[i%self.slen])
                
                if rv==True:
                    value_list.append(smooth)
                    season_list.append(seasonals[i%self.slen])
                
            #If we want to return the actual, smoothed values or only the forecast
        if rv==False:
            return torch.stack(result,dim=1)[:,-n_preds:]
        else:
            return torch.stack(result,dim=1),torch.stack(value_list,dim=1),torch.stack(season_list,dim=1)
                
            
class es_rnn(torch.nn.Module):
    
    def __init__(self,hidden_size=16,slen=12,pred_len=12):
      
        super(es_rnn, self).__init__()
        
        self.hw=holt_winters_no_trend(init_a=0.1,init_g=0.1)
        self.RNN=nn.GRU(hidden_size=hidden_size,input_size=1,batch_first=True)
        self.lin=nn.Linear(hidden_size,pred_len)
        self.pred_len=pred_len
        self.slen=slen
        
    def forward(self,series,shifts):
        
        #Get Batch size
        batch_size=series.shape[0]
        result,smoothed_value,smoothed_season = self.hw(series,shifts,rv=True,n_preds=0)
        
        de_season=series-smoothed_season
        de_level=de_season-smoothed_value
        noise=torch.randn(de_level.shape[0],de_level.shape[1])
        noisy=de_level#+noise
        noisy=noisy.unsqueeze(2)
        #noisy=noisy.permute(1,0,2)
        #take the last element in the sequence t agg (can also use attn)
        feature=self.RNN(noisy)[1].squeeze()#[-1,:,:]
        pred=self.lin(feature)
        
        #äthe season forecast entail just taking the correct smooothed values 
        season_forecast=[]
        for i in range(self.pred_len):
            season_forecast.append(smoothed_season[:,i%self.slen])
        season_forecast=torch.stack(season_forecast,dim=1)
        
        #in the end we multiply it all together and we are done!
        #here additive seems to work a bit better, need to make that an if/else of the model
        return smoothed_value[:,-1].unsqueeze(1)+season_forecast+pred
       