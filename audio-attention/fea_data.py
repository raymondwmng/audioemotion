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


class fea_data_npy(data.Dataset):
    def __init__(self, file_fea, file_ref, dataset_name):
        print('Features = ', file_fea)
        print('Reference = ', file_ref)
        self.fea = np.load(file_fea)
        ref = np.load(file_ref)
        self.ref = []
        for r in ref:
            if 'MOSEI' in dataset_name:
                self.ref.append(r[1:])
            else:
                self.ref.append(r)
        self.ref = np.array(self.ref)
        print("Loaded: features (%d) and reference (%d)" % (len(self.fea), len(self.ref)))

    def __len__(self):
        return len(self.fea)


    def __getitem__(self,index):
        return self.fea[index], self.ref[index]



class fea_data(data.Dataset):
    def __init__(self, file_fea, file_ref, dataset_name, dataset_split):
        print('Features = ', file_fea)
        print('Reference = ', file_ref)


        # check for scp and list all feature files
        scptag=False
        ffiles = []
#        ifnames = []
        if '.scp' in file_fea:
            scptag = True
            with open(file_fea) as f:
                [ffiles.append(line.strip().split(']')[0].split('[')) for line in f]
            self.ext = ffiles[0][0].split('[')[0].split('.')[-1]
        else:
            ffiles = [[file_fea]]
            self.ext = file_fea.split('.')[-1]
        print('scptag = ', scptag)


        # reading reference etm file
        print('reading reference...')
        if '.etm' in file_ref:
            ref = {}
            with open(file_ref) as f:
                for line in f:
                    if ';;' not in line:
                        l = line.split()
                        idname = l[0] + "_" + str(int(float(l[2])*100))
                        ref[idname] = l[4:]
        idnames = list(ref.keys())


        # split dataset in train/valid/test
        if 'MOSEI' in dataset_name and scptag == False:
            print('Using mosei_std_folds...')
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
                print('...with Edinburgh split')
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
            print('Using 85/5/10 split...')
            total = len(idnames)
            trainlen = int(total*0.85)
            validlen = int(total*0.05)
            train_ids = idnames[:trainlen]
            valid_ids = idnames[trainlen:(trainlen+validlen)]
            test_ids = idnames[(trainlen+validlen):]
            print("TOTALS: ref (%d) train (%d) valid (%d) test (%d)" % (total, len(train_ids), len(valid_ids), len(test_ids)))
        # necessary split
        if dataset_split == 'train':
            ids = train_ids
        if dataset_split == 'valid':
            ids = valid_ids
        if dataset_split == 'test':
            ids = test_ids


        # load data from necessary feature files
        print('reading features...')
        self.fea, self.ref = [], []
        if self.ext in ['mfcc','mfc','plp','fbk']:
            htk = HTKFeat_read()
        for ffile in ffiles:
            ff = ffile[0]
            print(ffile)
            beg = 0
            if len(ffile) > 1:    # find start and end times
                [beg, end] = ffile[1].split(',')
                if int(end) <= int(beg):
                    print('Warning: %s has end time (%s) not larger than start time (%s)' % (ff, beg, end))
                    beg = end
            idname = ff.split('/')[-1].split('.')[0] + "_" + str(beg)
#            if 'MOSEI' in dataset_name:
#                idname2 = ff.split('/')[-1].split('.')[0]
#            else:
#                idname2 = idname
#            print(idname, idname2)
            if idname in ids and idname in ref:
                print(ffile)
                if self.ext == 'npy':    # numpy features
                    feat = np.load(ff)
                elif self.ext == 'mat':    # covarep features
                    feat = h5py.File(ff)    # check this format
                elif self.ext in ['mfcc','mfc','plp','fbk']:    # htk features
#                    htk = HTKFeat_read()
#                    feat = htk.getall(ff) # for a single file
                    feat = htk.getsegment(ff, int(beg), int(end))
                    # no need to read all?



                else:
                    print("Error: File is in format '%s' which dataloader cannot read yet" % self.ext)
                    sys.exit()
                if len(ffile) > 1:    # find start and end times
#                    [beg, end] = ffile[1].split(',')
#                    self.fea.append(feat[int(beg):int(end)]) # beg, fea
                    self.fea.append(feat)
                    self.ref.append(ref[idname])
                else:
                    self.fea = feat # beg, fea
                    self.ref = np.load(file_ref)

        
        self.fea, self.ref = np.array(self.fea), np.array(self.ref)
        print("Dataset: %s, loaded: features (%d) and reference (%d)" % (dataset_split, len(self.fea), len(self.ref)))


    def __len__(self):
        if self.ext == 'npy':
            return len(self.fea)


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

