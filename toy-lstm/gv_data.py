import numpy as np
from torch.utils import data

class gv_data(data.Dataset):
    def __init__(self,file_gv,file_ref):
        self.gv = np.load(file_gv)
        self.ref = np.load(file_ref)

    def __len__(self):
        return len(self.gv)

    def __getitem__(self,index):
        return self.gv[index], self.ref[index]


class gv_test_data(data.Dataset):
    def __init__(self,file_gv):
        self.gv = np.load(file_gv)

    def __len__(self):
        return len(self.gv)

    def __getitem__(self,index):
        return self.gv[index]


