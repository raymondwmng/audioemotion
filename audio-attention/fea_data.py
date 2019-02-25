# loading the data which is in npy format
import numpy as np
import h5py
from torch.utils import data
import sys
sys.path.append('/share/spandh.ami1/usr/rosanna/tools/htk')
#sys.path.append('/home/rosanna/work/htk')
from htkmfc_python3 import HTKFeat_read 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# things to consider about loading data:
#1/ features contained in single file or multiple files?
#2/ unnecessary times removed or not (scp file?)
#3/ data split into train/valid/test already or not?

### ---------------- train data loader
class fea_data_npy(data.Dataset):
    def __init__(self, file_feas, file_refs, BATCHSIZE, PADDING):
        # reading in data and reference
        print('Training features = ', file_feas)
        print('Training reference = ', file_refs)
        for i, file_fea in enumerate(file_feas):
            file_ref = file_refs[i]
            fea = np.load(file_fea)
            ref = np.load(file_ref)
            tsk = [0] * len(ref)		# classification
            if "mosei" in file_ref:
                ref = np.delete(ref,1,1)
                tsk = [1] * len(ref)	# regression
            if i == 0:
                self.fea, self.ref, self.tsk = fea, ref, tsk 
            else:
                self.fea = np.concatenate( (self.fea, fea) )
                self.ref = np.concatenate( (self.ref, ref) )
                self.tsk += tsk
        print("Data loaded: features", self.fea.shape, "and reference", self.ref.shape)
        # removing nan and inf
        for i in range(len(self.fea)):
            x = self.fea[i]
            if np.any(np.isnan(x)):
                print(i, "contains NaN")
                self.fea[i][np.isnan(self.fea[i])] = 0
            if np.any(np.isinf(x)):
                print(i, "contains -inf or inf")
                # using nan_to_num still resulted in nans in network
                self.fea[i][np.isinf(self.fea[i])] = 0


    def __len__(self):
        return len(self.fea)


    def __getitem__(self,index):
        x = self.fea[index]
        y = self.ref[index]
        t = self.tsk[index]
#        x_norm = preprocessing.scale(x)
        scaler = StandardScaler().fit(x)
        scaler.transform(x)
        return x, y, t
#        return x_norm, y, t


### ---------------- test and eval data loader
class fea_test_data_npy(data.Dataset):
    def __init__(self, file_feas, file_refs, dataset_name):
        # reading in data and reference
        print('Test features  = ', file_feas)
        print('Test reference = ', file_refs)
        for i, file_fea in enumerate(file_feas):
            file_ref = file_refs[i]
            if i == 0:
                 fea = np.load(file_fea)
                 ref = np.load(file_ref)
                 tsk = [0] * len(ref)           # classification
                 if "mosei" in file_ref:
                     ref = np.delete(ref,1,1)
                     tsk = [1] * len(ref)       # regression
            if i == 0:
                 self.fea, self.ref, self.tsk = fea, ref, tsk
            else:
                 self.fea = np.concatenate( (self.fea, fea) )
                 self.ref = np.concatenate( (self.ref, ref) )
                 self.tsk += tsk
        # remove inf and nan
        for i in range(len(self.fea)):
            x = self.fea[i]
            if np.any(np.isnan(x)):
                print(i, "contains NaN")
                self.fea[i][np.isnan(self.fea[i])] = 0
            if np.any(np.isinf(x)):
                print(i, "contains -inf or inf")
                # using nan_to_num still resulted in nans in network
                self.fea[i][np.isinf(self.fea[i])] = 0
        print("Test data loaded (%s): features (%d) reference (%d) task labels (%d)" % (dataset_name, len(self.fea), len(self.ref), len(self.tsk)))


    def __len__(self):
        return len(self.fea)


    def __getitem__(self,index):
        x = self.fea[index]
        y = self.ref[index]
        t = self.tsk[index]
        scaler = StandardScaler().fit(x)
        scaler.transform(x)
        return x, y, t
#        x_norm = preprocessing.scale(x)
#        return x_norm, y, t

