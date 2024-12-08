import numpy as np
import random
import torch


# set_x shape: (number_of_patients, channels, width, height, depth)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, set_x):
        self.set_x = set_x

    def __len__(self):
        return len(self.set_x)

    def __getitem__(self, index):
        
        slice = random.randint(0, self.set_x.shape[4])
        
        x_np = self.set_x[index, :, :, :, :].copy()
        x_np[:, :, :, slice] = 0
        
        y_np = np.zeros(x_np.shape)
        y_np[:, :, :, slice] = self.set_x[index, :, :, :, slice].copy()
        
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        
        return x, y
