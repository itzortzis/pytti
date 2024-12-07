
import torch
import numpy as np

from pytti.training import Training
from pytti.models import ConvNeuralNet_b
from pytti.models import AlexNet



train_set_x = np.zeros((200, 256, 256))
train_set_y = np.zeros((200))
valid_set_x = np.zeros((50, 256, 256))
valid_set_y = np.zeros((50))
test_set_x = np.zeros((20, 256, 256))
test_set_y = np.zeros((20))

learning_rate = 0.00000001
classes = 5
# model = ConvNeuralNet_b(classes)
model = AlexNet(classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr = learning_rate)

comps = {
    'model': model,
    'opt': opt,
    'loss_fn': loss_fn,
    'sets': {
        'training_set_x': train_set_x,
        'training_set_y': train_set_y,
        'validation_set_x': valid_set_x,
        'validation_set_y': valid_set_y,
        'test_set_x': test_set_x,
        'test_set_y': test_set_y
    }
}

params = {
    'classes': classes,
    'epochs': 100,
    'epoch_thresh': 40,
    'score_thresh': 0.75,
    'device': 'cuda',
    'batch_size': 10,
}

t = Training(comps, params)
t.main_training()