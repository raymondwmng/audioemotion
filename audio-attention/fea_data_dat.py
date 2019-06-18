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


### ---------------- data specifics
fnames = {}
fnames["enterface"] = ["ent05p2_t34v5t5_shoutclipped","acted"]
fnames["ravdess"] = ["ravdess_t17v2t5_all1neNAcaNA_shoutclipped","acted"]
fnames["iemocap"] = ["iemocap_t1234t5_neNAfrNAexNAotNA","elicited"]
fnames["mosei"] = ["MOSEI_acl2018_neNA","natural"]
datatypes = ["acted","elicited","natural"]

### ---------------- train data loader
class fea_data_npy(data.Dataset):
    def __init__(self, file_feas, file_refs, BATCHSIZE, traindatalbl, TASK):
        # reading in data and reference
        print('Training features = ', file_feas)
        print('Training reference = ', file_refs)
        for i, file_fea in enumerate(file_feas):
            file_ref = file_refs[i]
            fea = np.load(file_fea)
            ref = np.load(file_ref)
            ref2 = np.load(file_ref)

            print(file_fea) #fnames, traindatalbl)
            print("EMOREF=", sum(ref), sum(sum(ref)))

            # find domain ref
            filelbl = file_fea.split("/")[7]
            domain_ref = np.array([[1 if lbl == fnames[filelbl][0] else 0 for lbl in traindatalbl]] * len(ref)) # domain classification
            print("DOMAINREF=", sum(domain_ref), sum(sum(domain_ref)))
            # find datatype ref
            datatype_ref = np.array([[1 if lbl == fnames[filelbl][1] else 0 for lbl in datatypes]] * len(ref)) # datatype classification
            print("DATATYPEREF=", sum(datatype_ref), sum(sum(datatype_ref)))
#            sys.exit()
            # find tasks
            if "+" in TASK:
                print(TASK, TASK.split("+"))
                # two tasks and therefore DAT
                TASK2 = TASK.split("+")
                if TASK2[0] == "EMO": # emo is primary task
                    if TASK2[1] == "DOM":
                        ref2 = domain_ref 
                    elif TASK2[1] == "TYP":
                        ref2 = datatype_ref 
                    elif TASK2[1] == "EMO":
                        ref2 = ref
                    else:
                        print("Unknown task specified (%s)" % TASK2[1])
                        sys.exit()
                else:
                    print("Primary task must be EMO, not %s" % TASK2[0])
                    sys.exit()
            elif TASK == "DOM":
                ref = domain_ref
                ref2 = domain_ref 
            elif TASK == "TYP":
                ref = datatype_ref
                ref2 = datatype_ref
            else:
                print("Unknown task specified (%s)" % TASK)
                sys.exit()
            # find multitask label
            if "mosei" in file_ref and TASK[:3] == "EMO":
                ref = np.delete(ref,1,1)
                if "+" not in TASK:
                    ref2 = np.delete(ref2,1,1)
            if i == 0:
                self.fea, self.ref, self.ref2 = fea, ref, ref2
            else:
#                print("fea", self.fea.shape, fea.shape)
#                print("ref", self.ref.shape, ref.shape)
#                print("ref2", self.ref2.shape, ref2.shape)
                self.fea = np.concatenate( (self.fea, fea) )
                self.ref = np.concatenate( (self.ref, ref) )
                self.ref2 = np.concatenate( (self.ref2, ref2) )
        # DAT - two classifiers
        if "+" in TASK:
            print("Data loaded: features", self.fea.shape, "and [%s] reference" % TASK2[0], self.ref.shape, "with [%s] reference" % TASK2[1], self.ref2.shape)
        else:
            print("Data loaded: features", self.fea.shape, "and [%s] reference" % TASK, self.ref.shape)
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
        y2 = self.ref2[index]
#        x_norm = preprocessing.scale(x)
        scaler = StandardScaler().fit(x)
        scaler.transform(x)
        return x, y, y2
#        return x_norm, y, t


### ---------------- test and eval data loader
class fea_test_data_npy(data.Dataset):
    def __init__(self, file_feas, file_refs, dataset_name, traindatalbl, TASK):
        # reading in data and reference
        print('Test features  = ', file_feas)
        print('Test reference = ', file_refs)
        for i, file_fea in enumerate(file_feas):
            file_ref = file_refs[i]
            fea = np.load(file_fea)
            ref = np.load(file_ref)

            print(file_fea) #fnames, traindatalbl)
            print("EMOREF=", sum(ref), sum(sum(ref)))

            # find domain ref
            filelbl = file_fea.split("/")[7]
            domain_ref = np.array([[1 if lbl == fnames[filelbl][0] else 0 for lbl in traindatalbl]] * len(ref)) # domain classification
            print("DOMAINREF=", sum(domain_ref), sum(sum(domain_ref)))
            # find datatype ref
            datatype_ref = np.array([[1 if lbl == fnames[filelbl][1] else 0 for lbl in datatypes]] * len(ref)) # datatype classification
            print("DATATYPEREF=", sum(datatype_ref), sum(sum(datatype_ref)))

            # find tasks
            if "+" in TASK:
                # two tasks and therefore DAT
                TASK2 = TASK.split("+")
                if TASK2[0] == "EMO": # emo is primary task
                    if TASK2[1] == "DOM":
                        ref2 = domain_ref
                    elif TASK2[1] == "TYP":
                        ref2 = datatype_ref
                    elif TASK2[1] == "EMO":
                        ref2 = ref
                    else:
                        print("Unknown task specified (%s)" % TASK2[1])
                        sys.exit()
                else:
                    print("Primary task must be EMO, not %s" % TASK2[0])
                    sys.exit()
            elif TASK == "DOM":
                ref = domain_ref
                ref2 = domain_ref
            elif TASK == "TYP":
                ref = datatype_ref
                ref2 = datatype_ref
            if "mosei" in file_ref and TASK[:3] == "EMO":
                ref = np.delete(ref,1,1)
                if "+" not in TASK:
                    ref2 = np.delete(ref2,1,1)
            if i == 0:
                self.fea, self.ref, self.ref2 = fea, ref, ref2
            else:
#                print("fea", self.fea.shape, fea.shape)
#                print("ref", self.ref.shape, ref.shape)
#                print("ref2", self.ref2.shape, ref2.shape)
                self.fea = np.concatenate( (self.fea, fea) )
                self.ref = np.concatenate( (self.ref, ref) )
                self.ref2 = np.concatenate( (self.ref2, ref2) )
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
        # DAT - two classifiers
        self.ref = np.array(self.ref)
        self.ref2 = np.array(self.ref2)
        if "+" in TASK:
            print("Test date loaded: features", self.fea.shape, "and [%s] reference" % TASK2[0], self.ref.shape, "with [%s] reference" % TASK2[1], self.ref2.shape)
        else:
            print("Test data loaded: features", self.fea.shape, "and [%s] reference" % TASK, self.ref.shape)


    def __len__(self):
        return len(self.fea)


    def __getitem__(self,index):
        x = self.fea[index]
        y = self.ref[index]
        y2 = self.ref2[index]
        scaler = StandardScaler().fit(x)
        scaler.transform(x)
        return x, y, y2
#        x_norm = preprocessing.scale(x)
#        return x_norm, y, t

