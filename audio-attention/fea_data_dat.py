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

fnames = {}
fnames["enterface"] = "ent05p2_t34v5t5_shoutclipped"
fnames["ravdess"] = "ravdess_t18v3t3_all1ne0caNA_shoutclipped"
fnames["iemocap"] = "iemocap_t1234t5_ne0frNAexNAotNA"
fnames["mosei"] = "MOSEI_acl2018"

### ---------------- train data loader
class fea_data_npy(data.Dataset):
    def __init__(self, file_feas, file_refs, BATCHSIZE, traindatalbl, MULTITASK):
        # reading in data and reference
        print('Training features = ', file_feas)
        print('Training reference = ', file_refs)
        for i, file_fea in enumerate(file_feas):
            file_ref = file_refs[i]
            fea = np.load(file_fea)
            ref = np.load(file_ref)
            filelbl = file_fea.split("/")[7]
            domain_ref = [[1 if lbl == fnames[filelbl] else 0 for lbl in traindatalbl]] * len(ref) # domain classification
            tsk = [0] * len(ref)		# classification
            if "mosei" in file_ref:
                ref = np.delete(ref,1,1)
#                ref = [r if r == max(r) else 0 for r in ref]	##### chekc this
                if MULTITASK:
                    tsk = [1] * len(ref)	# regression
            if i == 0:
                self.fea, self.ref, self.tsk, self.domain_ref = fea, ref, tsk, domain_ref
            else:
                self.fea = np.concatenate( (self.fea, fea) )
                self.ref = np.concatenate( (self.ref, ref) )
                self.tsk += tsk
                self.domain_ref += domain_ref
        self.domain_ref = np.array(self.domain_ref)
        print("Train data loaded: features (%d) reference (%d) task labels (%d) domain (%d)" % (len(self.fea), len(self.ref), len(self.tsk), len(self.domain_ref)))
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
        y2 = self.domain_ref[index]
#        x_norm = preprocessing.scale(x)
        scaler = StandardScaler().fit(x)
        scaler.transform(x)
        return x, y, t, y2
#        return x_norm, y, t


### ---------------- test and eval data loader
class fea_test_data_npy(data.Dataset):
    def __init__(self, file_feas, file_refs, dataset_name, traindatalbl, MULTITASK):
        # reading in data and reference
        print('Test features  = ', file_feas)
        print('Test reference = ', file_refs)
        for i, file_fea in enumerate(file_feas):
            file_ref = file_refs[i]
            if i == 0:
                 fea = np.load(file_fea)
                 ref = np.load(file_ref)
                 tsk = [0] * len(ref)           # classification
                 filelbl = file_fea.split("/")[7]
                 domain_ref = [[1 if lbl == fnames[filelbl] else 0 for lbl in traindatalbl]] * len(ref) # domain classification
                 if "mosei" in file_ref:
                     ref = np.delete(ref,1,1)
                     if MULTITASK:
                         tsk = [1] * len(ref)       # regression
            if i == 0:
                 self.fea, self.ref, self.tsk, self.domain_ref = fea, ref, tsk, domain_ref
            else:
                 self.fea = np.concatenate( (self.fea, fea) )
                 self.ref = np.concatenate( (self.ref, ref) )
                 self.tsk += tsk
                 self.domain_ref += domain_ref
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
        print("Test data loaded (%s): features (%d) reference (%d) task labels (%d) domain (%d)" % (dataset_name, len(self.fea), len(self.ref), len(self.tsk), len(self.domain_ref)))


    def __len__(self):
        return len(self.fea)


    def __getitem__(self,index):
        x = self.fea[index]
        y = self.ref[index]
        t = self.tsk[index]
        y2 = self.domain_ref[index]
        scaler = StandardScaler().fit(x)
        scaler.transform(x)
        return x, y, t, y2
#        x_norm = preprocessing.scale(x)
#        return x_norm, y, t

