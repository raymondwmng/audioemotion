# loading the data which is in npy format
import numpy as np
import h5py
from torch.utils import data
import sys
sys.path.append('/share/spandh.ami1/usr/rosanna/tools/htk')
from htkmfc_python3 import HTKFeat_read 
from sklearn import preprocessing

# things to consider about loading data:
#1/ features contained in single file or multiple files?
#2/ unnecessary times removed or not (scp file?)
#3/ data split into train/valid/test already or not?

#transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                         std=[0.229, 0.224, 0.225])
#])

### ---------------- train data loader
class fea_data_npy(data.Dataset):
    def __init__(self, file_feas, file_refs, dataset_name, BATCHSIZE, PADDING):
        # reading in data and reference
        print('Training features = ', file_feas)
        print('Training reference = ', file_refs)
        self.fea, self.ref, self.tsk = [], [], []
        fea = np.load(file_feas[0])
        ref = np.load(file_refs[0])
        if "mosei" in file_refs[0]:
            ref = np.delete(ref,1,1)
        if len(file_feas) > 1:
            for n in range(1,len(file_feas)):
                fea = np.concatenate( (fea, np.load(file_feas[n])) )
                if "mosei" in file_refs[n]:
                    tmpref = np.load(file_refs[n])
                    ref = np.concatenate( (ref, np.delete(tmpref, 1, 1)) )
                else:
                    ref = np.concatenate( (ref, np.load(file_refs[n])) )
        if BATCHSIZE > 1 and PADDING == True:
            # turn segments to list of frames
            self.lens = []
            for i,seg in enumerate(fea):
                # collect refs
                for j in range(len(seg)):
                    # collect lens
                    self.lens.append(len(seg))
                    if 'MOSEI' in dataset_name: # ref[i][0] = sentiment, ignore this
                        self.ref.append(ref[i])
                    else:
                        self.ref.append(ref[i])
            self.fea = np.concatenate(fea, axis=0)
#            print(len(fea), fea.shape, len(self.lens), len(self.ref))
#            # pad segments into equal length
#            longest_seg = max(self.lens)
#            padded_fea = np.ones((len(self.lens), longest_seg)) * 0
#            # copy over segments
##            for i, fea_len in enumerate(self.lens):
#                padded_fea[i,0:fea_len] = fea[i]
#            self.fea = padded_fea
#            print("Loaded: features (%d padded to %d) and reference (%d)" % (len(self.fea), longest_seg, len(self.ref)))
#            sys.exit()	###########
            print("Training data loaded: features (%d) and reference (%d)" % (len(self.fea), len(self.ref)))
        elif BATCHSIZE > 1:
            # chop into equal length
            for i, seg in enumerate(fea):
                beg, dur = 0, 50
                end = dur
                while end < len(seg):
                    self.fea.append(seg[beg:end])
                    if 'MOSEI' in dataset_name: # ref[i][0] = sentiment, ignore this
                        self.ref.append(ref[i])
                    else:
                        self.ref.append(ref[i])
                    beg, end = beg+dur, end+dur
            print("Training data loaded: features (%d chopped from %d) and reference (%d chopped from %d)" % (len(self.fea), len(fea), len(self.ref), len(ref)))
        else:
            for i, seg in enumerate(fea):
                self.fea.append(seg)
                self.ref.append( ref[i] )
                if 'MOSEI' in dataset_name: 
                    self.tsk.append(1)	#regression
                else:
                    self.tsk.append(0)	#classification
            self.fea = np.array(self.fea)
            self.ref = np.array(self.ref)
            print("Training data loaded: features", self.fea.shape, "and reference", self.ref.shape)
#        self.fea = np.array(self.fea)
#        self.ref = np.array(self.ref)
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
        x_norm = preprocessing.scale(x)
        return x_norm, y, t


### ---------------- test and eval data loader
class fea_test_data_npy(data.Dataset):
    def __init__(self, file_feas, file_refs, dataset_name):
        print('Testing features = ', file_feas)
        print('Testing reference = ', file_refs)
        self.fea, self.ref, self.tsk = [], [], []
        fea = np.load(file_feas[0])
        ref = np.load(file_refs[0])
        if "mosei" in file_refs[0]:
            ref = np.delete(ref,1,1)
        if len(file_feas) > 1:
            for n in range(1,len(file_feas)):
                fea = np.concatenate( (fea, np.load(file_feas[n])) )
                if "mosei" in file_refs[n]:
                    tmpref = np.load(file_refs[n])
                    ref = np.concatenate( (ref, np.delete(tmpref, 1, 1)) )
                else:
                    ref = np.concatenate( (ref, np.load(file_refs[n])) )
        self.fea = np.array(fea) 
        self.ref = np.array(ref)
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
        # always from the same dataset?
        if 'MOSEI' in dataset_name:
            self.tsk = [1] * len(self.fea)
        else:
            self.tsk = [0] * len(self.fea)
        print("Test data loaded (%s): features (%d) reference (%d) task labels (%d)" % (dataset_name, len(self.fea), len(self.ref), len(self.tsk)))

    def __len__(self):
        return len(self.fea)

    def __getitem__(self,index):
        x = self.fea[index]
        y = self.ref[index]
        t = self.tsk[index]
        x_norm = preprocessing.scale(x)
        return x_norm, y, t

