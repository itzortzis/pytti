import torch
import pytti.loaders as loaders
import numpy as np

class DataPreprocesing:
    def __init__(self, comps):
        self.comps = comps
        self.train_set_x = self.comps.sets['training_set_x']
        self.train_set_y = self.comps.sets['training_set_y']
        self.valid_set_x = self.comps.sets['validation_set_x']
        self.valid_set_y = self.comps.sets['validation_set_y']
        self.test_set_x = self.comps.sets['test_set_x']
        self.test_set_y = self.comps.sets['test_set_y']
        
        
    def run(self):
        # self.normalize_sets()
        self.build_loaders()
        
    def normalize(self, s, max_, min_):

        for i in range(len(s)):
            s[i, :, :, 0] = (s[i, :, :, 0] - min_) / (max_ - min_)

        return s


    def normalize_sets(self):
        max_ = np.max(self.train_set[:, :, :, 0])
        min_ = np.min(self.train_set[:, :, :, 0])
        print(max_, min_)
        self.train_set = self.normalize(self.train_set, max_, min_)
        self.valid_set = self.normalize(self.valid_set, max_, min_)
        self.set_set = self.normalize(self.test_set, max_, min_)


    def build_loaders(self):
        train_set      = loaders.Dataset(self.train_set_x, self.train_set_y, self.comps.classes)
        params         = {'batch_size': 10, 'shuffle': True}
        self.train_ldr = torch.utils.data.DataLoader(train_set, **params)
        valid_set      = loaders.Dataset(self.valid_set_x, self.valid_set_y, self.comps.classes)
        params         = {'batch_size': 10, 'shuffle': False}
        self.valid_ldr = torch.utils.data.DataLoader(valid_set, **params)
        test_set       = loaders.Dataset(self.test_set_x, self.test_set_y, self.comps.classes)
        params         = {'batch_size': 10, 'shuffle': False}
        self.test_ldr  = torch.utils.data.DataLoader(test_set, **params)
