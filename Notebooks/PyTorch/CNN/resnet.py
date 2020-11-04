import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import CIFAR10
from torch.autograd import Variable
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random




class _Block(nn.Module):
    def __init__(self, kernel_dim, n_filters, in_channels, stride, padding):
        super(_Block, self).__init__()
        self._layer_one = nn.Conv2d(in_channels=in_channels, out_channels=n_filters, 
                                  kernel_size=kernel_dim, stride=stride[0], padding=padding, bias=False)
        self._layer_two = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, 
                                  kernel_size=kernel_dim, stride=stride[1], padding=padding, bias=False)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.bn2 = nn.BatchNorm2d(n_filters)

    def forward(self, X, shortcut = None):
        output = self._layer_one(X)
        output = self.bn1(output)
        output = self.relu(output)
        output = self._layer_two(output)
        output = self.bn2(output)
        output = self.relu(output)
        if isinstance(shortcut, torch.Tensor):
            return output + shortcut
        return output + X


class ResNet(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(ResNet, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(3, 64, 4, 2)
        self.block1 = _Block(5, 64, 64, (1,1), 2)
        self.block2 = _Block(5, 64, 64, (1,1), 2)
        self.block3 = _Block(5, 64, 64, (1,1), 2)
        self.transition1 = nn.Conv2d(64, 128, 1, 2, 0, bias=False)
        self.block4 = _Block(3, 128, 64, (2,1), 1)
        self.block5 = _Block(3, 128, 128, (1,1), 1)
        self.block6 = _Block(3, 128, 128, (1,1), 1)
        self.transition2 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        self.block7 = _Block(3, 256, 128, (2,1), 1)
        self.block8 = _Block(3, 256, 256, (1,1), 1)
        self.block9 = _Block(3, 256, 256, (1,1), 1)
        self.transition3 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)
        self.block10 = _Block(3, 512, 256, (2,1), 1)
        self.block11 = _Block(3, 512, 512, (1,1), 1)
        self.block12 = _Block(3, 512, 512, (1,1), 1)
        self.linear1 = nn.Linear(2048, n_classes)


    def forward(self, X):
        output = self.conv1(X)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        shortcut1 = self.transition1(output)
        output = self.block4(output, shortcut1)
        output = self.block5(output)
        output = self.block6(output)
        shortcut2 = self.transition2(output)
        output = self.block7(output, shortcut2)
        output = self.block8(output)
        output = self.block9(output)
        shortcut3 = self.transition3(output)
        output = self.block10(output, shortcut3)
        output = self.block11(output)
        output = self.block12(output)
        output = output.view(-1, 2048)
        output = self.linear1(output)
        return output



if __name__ == "__main__":
    X = torch.ones((1,3,32,32))
    model = ResNet(32, 10)
    model(X)



