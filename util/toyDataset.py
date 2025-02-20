import torch
import numpy as np
import os
import pandas as pd
from torch.nn import functional as F

def toy_dataset(dataset,num_sample=100):
    path = f"TimeSeries\\{dataset}"
    metadata = pd.read_csv(f'{path}\\'+'label.csv')[["subject","DX_GROUP"]]
    TimeSeries, labels = [], []
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith('cc200.1D'):continue
            timeSeries = torch.Tensor(np.loadtxt(root+'\\'+file,dtype=float).transpose(1,0)).to('cpu')
            length = timeSeries.shape[1]
            paddingsize = 360-length
            timeSeries = (timeSeries-timeSeries.mean(-1,keepdim=True))/(timeSeries.std(-1,keepdim=True)+1e-9)
            TimeSeries.append(timeSeries)
            ID = file[-19:-14]
            label = metadata.loc[metadata['subject']==int(ID)].iloc[0,1]-1
            labels.append(label)
            count+=1
            if count>= num_sample:break
    return TimeSeries, labels

def timeseries_split(t):
    split_point = t.shape[-1]//2
    # print(f"split time series with length: {t.shape[1]} from point: {split_point}")
    return t[:,0:split_point], t[:,split_point:], torch.tensor([split_point])

def pearson(t):
    t = (t-t.mean(-1,keepdim=True))/(t.std(-1,keepdim=True)+1e-9)
    return t@t.T/t.shape[1]