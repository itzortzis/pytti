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
    
    
    
    class AlexNet(nn.Module):
        def __init__(self, num_classes=10):
            super(AlexNet, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=0),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2))
            self.layer3 = nn.Sequential(
                nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU())
            self.layer4 = nn.Sequential(
                nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU())
            self.layer5 = nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2))
            self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(9216, 4096),
                nn.ReLU())
            self.fc1 = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(4096, 2048),
                nn.ReLU())
            self.fc2= nn.Sequential(
                nn.Linear(2048, num_classes))
            
        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            out = self.fc1(out)
            out = self.fc2(out)
            return out