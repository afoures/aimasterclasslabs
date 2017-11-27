import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.minibatch_size = 1
        self.conv0 = nn.Conv2d(in_channels = 1,
                                out_channels = 16,
                                kernel_size = (3, 3),
                                stride = 1,
                                padding = 1)
        self.bn0 = nn.BatchNorm2d(16 * self.minibatch_size)
        self.conv1 = nn.Conv2d(in_channels = 16,
                                out_channels = 16,
                                kernel_size = (3, 3),
                                stride = 1,
                                padding = 1)
        self.bn1 = nn.BatchNorm2d(16 * self.minibatch_size)
        self.maxpool0 = nn.MaxPool2d(kernel_size = 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        self.fc0 = nn.Linear(14*14*16, 42)
        self.fc1 = nn.Linear(42, 27)


    def forward(self, x):
        #[[conv-relu-bn] * n] - maxpool - fc - fc
        conv_relu0 = self.conv0(x).clamp(min=0)
        bn0 = self.bn0(conv_relu0)

        conv_relu1 = self.conv1(bn0).clamp(min=0)
        bn1 = self.bn1(conv_relu1)
        maxpool0 = self.maxpool0(bn1)

        out_fc0 = self.fc0(maxpool0.view(maxpool0.size(0), -1))
        out_fc1 = self.fc1(out_fc0)
        return F.log_softmax(out_fc1)
