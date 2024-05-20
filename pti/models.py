from torch import nn
import torch.nn.functional as F

class ConvNeuralNet_b(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNet_b, self).__init__()

        self.init_conv_layer_1()
        self.init_conv_layer_2()
        self.fc   = nn.Linear(32*64*64, num_classes) # 256x256
        

    def init_conv_layer_1(self):
        self.conv1  = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride = 1, padding=2)
        self.conv12 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride = 1, padding=2)
        self.batch1 = nn.BatchNorm2d(16)
        self.relu1  = nn.ReLU()
        self.pool1  = nn.MaxPool2d(kernel_size=2)
        self.drop_1 = nn.Dropout2d(0.50)


    def init_conv_layer_2(self):
        self.conv2  = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride = 1, padding=2)
        self.conv22 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride = 1, padding=2)
        self.batch2 = nn.BatchNorm2d(32)
        self.relu2  = nn.ReLU()
        self.pool2  = nn.MaxPool2d(kernel_size=2)
        self.drop_2 = nn.Dropout2d(0.25)


    def exec_conv_layer_1(self, x):
        out = self.conv1(x)
        out = self.conv12(out)
        out = self.batch1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.drop_1(out)

        return out


    def exec_conv_layer_2(self, x):
        out = self.conv2(x)
        out = self.conv22(out)
        out = self.batch2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.drop_2(out)

        return out


    def forward(self, x):
        out = self.exec_conv_layer_1(x)
        out = self.exec_conv_layer_2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        out = F.softmax(out, dim = 1)

        return out