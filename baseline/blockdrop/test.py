import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
from torch.autograd import Variable
import numpy as np



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

if __name__ == "__main__":

    inputs = torch.randn(1,3,32, 32).cuda()
    model = conv3x3(3,16).cuda()
    print(inputs.shape)
    print(model(inputs).shape)