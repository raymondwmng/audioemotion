#!/usr/bin/python
import numpy as np
from torch.utils import data
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


### ---------------- data loader
class data_loader_npy(data.Dataset):
    def __init__(self, featurefile):
        # read in data
        self.fea = np.array([np.load(featurefile)])
        # prevent nan or inf
        for i in range(len(self.fea)):
            x = self.fea[i]
            if np.any(np.isnan(x)):
                print(i, "contains NaN")
                self.fea[i][np.isnan(self.fea[i])] = 0
            if np.any(np.isinf(x)):
                print(i, "contains -inf or inf")
                # using nan_to_num still resulted in nans in network
                self.fea[i][np.isinf(self.fea[i])] = 0
        print("Data loaded:", self.fea.shape) 


    def __len__(self):
        return len(self.fea)


    def __getitem__(self,index):
        x = self.fea[index]
        scaler = StandardScaler().fit(x)
        scaler.transform(x)
        return x

