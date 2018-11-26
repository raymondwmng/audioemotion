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


class fea_data(data.Dataset):
    def __init__(self, file_fea, file_ref, dataset_name, dataset_split):
        # check for scp and list all feature files
        scptag=False
        ffiles = []
        if '.scp' in file_fea:
            scptag = True
            with open(file_fea) as f:
                [ffiles.append(line.strip().split(']')[0].split('[')) for line in f]
            f.close()
        else:
            ffiles = [[file_fea]]

        # load data from all feature files
        fea = {}
        idnames = []
        for ffile in ffiles:
            ff = ffile[0]
            if len(ffile) > 1:    # find start and end times
                [beg, end] = ffile[1].split(',')
                if int(end) <= int(beg):
                    print('Warning: %s has end time (%s) not larger than start end (%s)' % (ff, beg, end))
                    beg = end
            idname = ff.split('/')[-1].split('.')[0]
            idnames.append(idname)
            self.ext = ff.split('.')[-1]
            if self.ext == 'npy':    # numpy features
                feat = np.load(ff)
            elif self.ext == 'mat':    # covarep features
                feat = h5py.File(ff)    # check this format
            elif self.ext in ['mfcc','mfc','plp','fbk']:    # htk features
                htk = HTKFeat_read()
                feat = htk.getall(ff) # for a single file
            else:
                print("File is in format '%s' which dataloader cannot read yet" % self.ext)
                sys.exit()
            if len(ffile) > 1:    # find start and end times
                [beg, end] = ffile[1].split(',')
                fea[idname] = feat[int(beg):int(end)]
            else:
                fea[idname] = feat
        # not using scp files
        if scptag == False:
            fea = fea[idname]
            ref = np.load(file_ref)
        else:
            tmp_ref = np.load(file_ref)    # currently assuming ref always in numpy from
            ref = {}
            i = 0
            for idname in idnames:
                ref[idname] = tmp_ref[i]   # check this...
                i += 1
        print(len(fea), len(ref), len(idnames))


        # split dataset in train/valid/test
        if 'MOSEI' in dataset_name:
            sys.path.append('/share/spandh.ami1/emotion/import/feat/converthdf5numpy/')
#            from ids_11866 import ids_11866
            id_npy = list(np.load('/share/spandh.ami1/emotion/import/feat/converthdf5numpy/ids_11866.npy'))
            id_import = [idname.split('[')[0] for idname in id_npy]
            sys.path.append('/share/spandh.ami1/emotion//tools/audioemotion/preprocessing/CMU-MultimodalSDK/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/')
            from cmu_mosei_std_folds import standard_train_fold as train_ids
            from cmu_mosei_std_folds import standard_valid_fold as valid_ids
            from cmu_mosei_std_folds import standard_test_fold as test_ids
            # creating dataset split according to ACL2018 Edinburgh
            if 'edin' in dataset_name:
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
        else:
            # split into 85/5/10%
            total = len(idnames)
            trainlen = int(total*0.85)
            validlen = int(total*0.05)
            train_ids = idnames[:trainlen]
            valid_ids = idnames[trainlen:(trainlen+validlen)]
            test_ids = idnames[(trainlen+validlen):]
        # necessary split
        if dataset_split == 'train':
            ids = train_ids
        if dataset_split == 'valid':
            ids = valid_ids
        if dataset_split == 'test':
            ids = test_ids

        # find data in given splits
        self.fea, self.ref = [], []
        if scptag == True: #### FIX THIS ####
             for idname in idnames:
                 if idname in ids:
                     self.fea.append(fea[idname])
                     self.ref.append(ref[idname])
        else:
             for idname in id_import:
                if idname in ids:
                    self.fea.append(fea[id_import.index(idname)])
                    self.ref.append(ref[id_import.index(idname)])
        self.fea, self.ref = np.array(self.fea), np.array(self.ref)
        print("Dataset: %s, loaded: features (%d) and reference (%d)" % (dataset_split, len(self.fea), len(self.ref)))


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

