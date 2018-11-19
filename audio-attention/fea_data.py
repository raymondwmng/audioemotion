# loading the data which is in npy format
import numpy as np
import h5py
from torch.utils import data
import sys
sys.path.append('/share/spandh.ami1/usr/rosanna/tools/htk')
import htkmfc_python3


# things to consider about loading data:
#1/ features contained in single file or multiple files?
#2/ unnecessary times removed or not (scp file?)
#3/ data split into train/valid/test already or not?


class fea_data(data.Dataset):
    def __init__(self, file_fea, file_ref, dataset_name, dataset_split):
        # reading feature data
        self.ext = file_fea.split('.')[-1]
        if self.ext == 'npy':    # numpy features
            fea = np.load(file_fea)
        elif self.ext == 'mat':    # covarep features
            fea = h5py.File(file_fea)    # check this format
        elif self.ext in ['mfcc','mfc','plp','fbk']:    # htk features
            htk = HTKFeat_read(file_fea)
            fea = htk.getall(fea_file) # for a single file
        else:
            print("File is in format '%s' which dataloader cannot read yet" % self.ext)
            sys.exit()
        # reading reference data
        ref = np.load(file_ref)    # currently assuming ref always in numpy from (mosei so far)


        # data splits - or do this before?
        if 'MOSEI' in dataset_name:
            sys.path.append('/share/spandh.ami1/emotion/import/feat/converthdf5numpy/')
#            from ids_11875 import ids_11875     # think about how to implement this
#            from ids_11866 import ids_11866
            ids_npy = list(np.load('/share/spandh.ami1/emotion/import/feat/converthdf5numpy/ids_11866.npy'))
#            ids_npy = list(np.load('/share/spandh.ami1/emotion/import/feat/converthdf5numpy/ids_11875.npy'))
            # creating dataset split according to ACL2018
#            if dataset_name == 'MOSEI_acl2018':
            sys.path.append('/share/spandh.ami1/emotion//tools/audioemotion/preprocessing/CMU-MultimodalSDK/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/')
            from cmu_mosei_std_folds import standard_train_fold as train_ids
            from cmu_mosei_std_folds import standard_valid_fold as valid_ids
            from cmu_mosei_std_folds import standard_test_fold as test_ids
            # creating dataset split according to ACL2018 Edinburgh
            if dataset_name == 'MOSEI_edin':
                valid = list(valid_ids)
                train = list(train_ids)
                train_ids, valid_ids, test_ids = [], [], []
                total = len(valid)
                half = int(total / 2)
                valid_ids = valid[:half]
                test_ids = valid[half+1:]
                # 5 % of training into test data
                five_p = int(len(train) * 0.05)
                train_ids = train[:-five_p]
                test_ids = test_ids + train[-five_p:]
                # 10% of leftover training into valid data
                ten_p = int(len(train_ids) * 0.1)
                train_ids = train_ids[:-ten_p]
                valid_ids = valid_ids + train_ids[-ten_p:]
            # finding correct features for the splits
            if dataset_split == 'train':
                ids = train_ids
            if dataset_split == 'valid':
                ids = valid_ids
            if dataset_split == 'test':
                ids = test_ids

            self.fea, self.ref = [], []
            for idname in ids:
                for idname2 in ids_npy:
                    if idname == idname2.split('[')[0]:
                        self.fea.append(fea[ids_npy.index(idname2)])
                        self.ref.append(ref[ids_npy.index(idname2)])
            self.fea, self.ref = np.array(self.fea), np.array(self.ref)
            print("Loaded %s (%d) and reference (%d)" % (dataset_split, len(self.fea), len(self.ref)))




#	if ".csd" in file_fea:
#		self.fea = h5py.File(file_fea)
#		f = h5py.File(file_fea)
#		self.fea = []
#		for (
#	if ".csd" in file_ref:
#		self.ref = h5py.File(file_ref)

    def __len__(self):
        if self.ext == 'npy':
            return len(self.fea)
        elif self.ext == 'mat':
            return len(self.fea) # check this
        elif self.ext in ['csd','h5py']:
            return len(self.fea['COVAREP']['data'])


    def __getitem__(self,index):
        return self.fea[index], self.ref[index]


#class fea_test_data(data.Dataset):
#    def __init__(self,file_fea):
#        self.fea = np.load(file_fea)
#
#    def __len__(self):
#        return len(self.fea)
#
#    def __getitem__(self,index):
#        return self.fea[index]


##fea['COAVAREP']['data']['zx4W0Vuus-I']['features'].shape=(8171, 74)

