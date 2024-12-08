
import torch
import numpy as np
import torch.optim as optim

from pytti.training import Training
from pytti.training import CNNInterpolator3D



train_set_x = np.zeros((50, 1, 512, 512, 40))
valid_set_x = np.zeros((20, 256, 256))
test_set_x = np.zeros((10, 256, 256))


model = CNNInterpolator3D(in_channels=1, out_channels=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


comps = {
    'model': model,
    'opt': optimizer,
    'loss_fn': criterion,
    'sets': {
        'training_set_x': train_set_x,
        'validation_set_x': valid_set_x,
        'test_set_x': test_set_x,
    }
}

params = {
    'classes': 2,
    'epochs': 100,
    'epoch_thresh': 0,
    'score_thresh': 0.0,
    'device': 'cuda',
    'batch_size': 10,
}

t = Training(comps, params)
t.main_training()