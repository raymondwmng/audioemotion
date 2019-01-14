# loading the data which is in npy format
import numpy as np
import h5py
from torch.utils import data
import sys
sys.path.append('/share/spandh.ami1/usr/rosanna/tools/htk')
from htkmfc_python3 import HTKFeat_read 

# things to consider about loading data:
#1/ features contained in single file or multiple files?
#2/ unnecessary times removed or not (scp file?)
#3/ data split into train/valid/test already or not?


### ---------------- train data loader
class fea_data_npy(data.Dataset):
    def __init__(self, file_feas, file_refs, dataset_name, BATCHSIZE, PADDING):
        print('Features = ', file_feas)
        print('Reference = ', file_refs)
        self.fea, self.ref = [], []
        for n in range(len(file_feas)):
            fea = np.load(file_feas[n])
            ref = np.load(file_refs[n])
            if BATCHSIZE > 1 and PADDING == True:
                # pad segments into equal length
                fea_lens = [len(seg) for seg in fea]
                print(fea_lens)
                longest_seg = max(fea_lens)
                print(longest_seg)
                padded_fea = np.ones((len(fea), longest_seg)) * 0
                print(padded_fea)
                # copy over segments
                for i, fea_len in enumerate(fea_lens):
                    sequence = fea[i]
                    padded_fea[i,0:fea_len] = sequence[:fea_len]
                print(padded_fea)
                # to continue...
            elif BATCHSIZE > 1:
                # chop into equal length
#                self.fea = []
#                self.ref = []
                for i in range(len(fea)):
                    fea_segment = fea[i]
                    ref_segment = ref[i]
                    beg, dur = 0, 50
                    end = dur
                    while end < len(fea_segment):
                        self.fea.append(fea_segment[beg:end])
                        # ignore first point in reference as it is sentiment for MOSEI
                        if 'MOSEI' in dataset_name:
                            self.ref.append(ref_segment[1:])
                        else:
                            self.ref.append(ref_segment)
                        beg, end = beg+dur, end+dur
#                self.fea = np.array(self.fea)
#                self.ref = np.array(self.ref)
                print("Loaded: features (%d chopped from %d) and reference (%d chopped from %d)" % (len(self.fea), len(fea), len(self.ref), len(ref)))
            else:
#                self.ref = []
                for seg in fea:
                    self.fea.append(seg)
                for r in ref:
                    
                    # ignore first point in reference as it is sentiment for MOSEI
                    if 'MOSEI' in dataset_name:
                        self.ref.append(r[1:])
                    else:
                        self.ref.append(r)
#                self.ref = np.array(self.ref)
#                self.fea = np.array(fea)
#                print("Loaded: features (%d) and reference (%d)" % (len(self.fea), len(self.ref)))
        self.fea = np.array(self.fea)
        self.ref = np.array(self.ref)
        print("Loaded: features (%d) and reference (%d)" % (len(self.fea), len(self.ref)))

    def __len__(self):
        return len(self.fea)

    def __getitem__(self,index):
        return self.fea[index], self.ref[index]


### ---------------- test and eval data loader
class fea_test_data_npy(data.Dataset):
    def __init__(self,file_feas, file_refs, dataset_name):
        self.fea, self.ref = [], []
        for n in range(len(file_feas)):
            fea = np.load(file_feas[n])
            ref = np.load(file_refs[n])
            for seg in fea:
                    self.fea.append(seg)
            for r in ref:
                # ignore first point in reference as it is sentiment for MOSEI
                if 'MOSEI' in dataset_name:
                    self.ref.append(r[1:])
                else:
                    self.ref.append(r)
        self.fea = np.array(self.fea)
        self.ref = np.array(self.ref)
        print("Loaded: features (%d) and reference (%d)" % (len(self.fea), len(self.ref)))

    def __len__(self):
        return len(self.fea)

    def __getitem__(self,index):
        return self.fea[index], self.ref[index]

