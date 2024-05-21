
import torch
import numpy as np

from pti.training import Training
from pti.models import ConvNeuralNet_b
from pti.models import AlexNet



train_set_x = np.zeros((500, 256, 256))
train_set_y = np.zeros((500))
valid_set_x = np.zeros((100, 256, 256))
valid_set_y = np.zeros((100))
test_set_x = np.zeros((50, 256, 256))
test_set_y = np.zeros((50))

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
    'epochs': 100,
    'exp_name': 'INBreast_bad_0_1000',
    'epoch_thresh': 40,
    'score_thresh': 0.75,
    'device': 'cuda',
    'batch_size': 10,
    'inf_model_name': 'INBreast_bad_0_1000_1713349397.pth',
}
tr_paths = {
    'trained_models': './generated/trained_models/',
    'metrics': './generated/metrics/',
    'figures': './generated/figures/'
}
t = Training(comps, params, tr_paths)
t.main_training()