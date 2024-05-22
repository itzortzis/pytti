import numpy as np
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, set_x, set_y):
        self.set_x = set_x
        self.set_y = set_y
        self.num_of_classes = 5

    def __len__(self):
        return len(self.set_y)

    def __getitem__(self, index):

        y = self.create_y(self.set_y[index])
        # y = np.zeros((2), dtype=np.float16)
        # if self.set_y[index] <= 3:
        #     y[0] = 1
        # else:
        #     y[1] = 1

        x = torch.from_numpy(self.set_x[index, :, :])
        y = torch.from_numpy(y)
        
        return x, y
    
    def create_y(self, v):
        value = int(v)
        y = np.zeros((self.num_of_classes), dtype=np.float16)
        y[value-1] = 1
        return y