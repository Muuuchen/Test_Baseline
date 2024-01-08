"""
 *@description: blockdrop
 *@author: haozhaoyang
 *@date: 2023-10-30
 *@idea: 
 """

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
from torch.autograd import Variable
import numpy as np

n_warmup = 100
n_run = 100

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--overhead_test', action='store_true')
parser.add_argument('--unroll', dest='unroll', action='store_true')
parser.add_argument('--fix', dest='unroll', action='store_false')
parser.set_defaults(unroll=False)
parser.add_argument('--rate', type=int, default=-1)
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--use_profile', type=bool,default=True)

arguments = parser.parse_args()
use_profile = arguments.use_profile


import sys
from time import time
sys.path.append('../utils')
from benchmark_timer import Timer
from nsysprofile import profile_start, profile_stop, enable_profile
enable_profile(use_profile)

cuda_device = torch.device("cuda:0")

# torch.manual_seed(196)
# # 如果使用GPU，还需要设置 GPU 的随机数种子
# torch.cuda.manual_seed_all(196)
# # 设置CUDNN以确保一致性
# torch.backends.cudnn.deterministic = True
np.random.seed(196)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return out


class DownsampleB(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.avg = nn.AvgPool2d(stride)
        self.expand_ratio = nOut // nIn

    def forward(self, x):
        x = self.avg(x)
        x = torch.cat((x, torch.zeros_like(x)), dim=1)
        return x


class FlatResNet32(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(FlatResNet32, self).__init__()

        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.avgpool = nn.AvgPool2d(8)

        strides = [1, 2, 2]
        filt_sizes = [16, 32, 64]
        self.blocks, self.ds = [], []
        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc_dim = 64 * block.expansion


        self.layer_config = layers

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def seed(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        return x


    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = nn.Sequential()
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleB(self.inplanes, planes * block.expansion, stride)

        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1))

        return layers, downsample


class FlatResNet32Policy(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(FlatResNet32Policy, self).__init__()

        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.avgpool = nn.AvgPool2d(8)

        strides = [1, 2, 2]
        filt_sizes = [16, 32, 64]
        self.blocks, self.ds = [], []
        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc_dim = 64 * block.expansion

        self.layer_config = layers

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = nn.Sequential()
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleB(self.inplanes, planes * block.expansion, stride)

        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1))

        return layers, downsample


    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        # layer 00
        residual0 = self.ds[0](x)
        b0 = self.blocks[0][0](x)
        x0 = torch.relu(residual0 + b0)
        # layer 10
        residual1 = self.ds[1](x0)
        b1 = self.blocks[1][0](x0)
        x1 = torch.relu(residual1 + b1)
        # layer 20
        residual2 = self.ds[2](x1)
        b2 = self.blocks[2][0](x1)
        x2 = torch.relu(residual2 + b2)
        # postprocessing
        x = self.avgpool(x2)
        x = x.view(x.size(0), 64)
        return x


class Policy32(nn.Module):

    def __init__(self, layer_config=[1,1,1], num_blocks=15):
        super(Policy32, self).__init__()
        self.features = FlatResNet32Policy(BasicBlock, layer_config, num_classes=10)
        self.feat_dim = self.features.fc.weight.data.shape[1]
        self.features.fc = nn.Sequential()
        self.num_layers = sum(layer_config)

        self.logit = nn.Linear(self.feat_dim, num_blocks)
        self.vnet = nn.Linear(self.feat_dim, 1)

    def load_state_dict(self, state_dict):
        # support legacy models
        state_dict = {k:v for k,v in state_dict.items() if not k.startswith('features.fc')}
        return super(Policy32, self).load_state_dict(state_dict)


    def forward(self, x):
        y = self.features(x)
        value = self.vnet(y)
        probs = torch.sigmoid(self.logit(y))
        return probs, value


class BlockDrop(nn.Module):
    def __init__(self, rnet, agent) -> None:
        super().__init__()
        self.rnet = rnet
        self.agent = agent
    
    def forward_resnet(self, inputs):
        # FlatResNet
        x = self.rnet.seed(inputs)

        # layer 00
        residual = self.rnet.ds[0](x)
        fx = self.rnet.blocks[0][0](x)
        x = torch.relu(residual + fx)
        
        # layer 01
        residual = x
        fx = self.rnet.blocks[0][1](x)
        x = torch.relu(residual + fx)
        
        # layer 02
        residual = x
        fx = self.rnet.blocks[0][2](x)
        x = torch.relu(residual + fx)
        
        # layer 03
        residual = x
        fx = self.rnet.blocks[0][3](x)
        x = torch.relu(residual + fx)

        # layer 04
        residual = x
        fx = self.rnet.blocks[0][4](x)
        x = torch.relu(residual + fx)

        # layer 10
        residual = self.rnet.ds[1](x)
        fx = self.rnet.blocks[1][0](x)
        x = torch.relu(residual + fx)

        # layer 11
        residual = x
        fx = self.rnet.blocks[1][1](x)
        x = torch.relu(residual + fx)

        # layer 12
        residual = x
        fx = self.rnet.blocks[1][2](x)
        x = torch.relu(residual + fx)

        # layer 13
        residual = x
        fx = self.rnet.blocks[1][3](x)
        x = torch.relu(residual + fx)

        # layer 14
        residual = x
        fx = self.rnet.blocks[1][4](x)
        x = torch.relu(residual + fx)

        # layer 20
        residual = self.rnet.ds[2](x)
        fx = self.rnet.blocks[2][0](x)
        x = torch.relu(residual + fx)
        
        # layer 21
        residual = x
        fx = self.rnet.blocks[2][1](x)
        x = torch.relu(residual + fx)

        # layer 22
        residual = x
        fx = self.rnet.blocks[2][2](x)
        x = torch.relu(residual + fx)

        # layer 23
        residual = x
        fx = self.rnet.blocks[2][3](x) 
        x = torch.relu(residual + fx)

        # layer 24
        residual = x
        fx = self.rnet.blocks[2][4](x)
        x = torch.relu(residual + fx)

        x = self.rnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.rnet.fc(x)
        return x
    
    def forward_real(self, inputs, probs):
        # probs, _ = self.agent(inputs)

        cond = torch.lt(probs, torch.full_like(probs, 0.5))
        policy = torch.where(cond, torch.zeros_like(probs), torch.ones_like(probs))
        policy = policy.transpose(0, 1)

        # FlatResNet
        x = self.rnet.seed(inputs)

        # layer 00
        action = policy[0]
        residual = self.rnet.ds[0](x)
        if torch.sum(action) > 0.0:
            action_mask = action.view(-1,1,1,1)
            fx = self.rnet.blocks[0][0](x)
            print("$$$$$$$$",fx.shape)
            # print("$$$$$$$$",residual.shape)
            fx = torch.relu(residual + fx)


            x = fx*action_mask + residual*(1.0-action_mask)
        else:
            x = residual

        # layer 01
        action = policy[1]
        residual = x
        if torch.sum(action) > 0.0:
            action_mask = action.view(-1,1,1,1)
            fx = self.rnet.blocks[0][1](x)

            fx = torch.relu(residual + fx)
            x = fx*action_mask + residual*(1.0-action_mask)
        

        # layer 02
        action = policy[2]
        residual = x
        if torch.sum(action) > 0.0:
            action_mask = action.view(-1,1,1,1)
            fx = self.rnet.blocks[0][2](x)
            fx = torch.relu(residual + fx)
            x = fx*action_mask + residual*(1.0-action_mask)

        # layer 03
        action = policy[3]
        residual = x
        if torch.sum(action) > 0.0:
            action_mask = action.view(-1,1,1,1)
            fx = self.rnet.blocks[0][3](x)
            fx = torch.relu(residual + fx)
            x = fx*action_mask + residual*(1.0-action_mask)

        # layer 04
        action = policy[4]
        residual = x
        if torch.sum(action) > 0.0:
            action_mask = action.view(-1,1,1,1)
            fx = self.rnet.blocks[0][4](x)
            fx = torch.relu(residual + fx)
            x = fx*action_mask + residual*(1.0-action_mask)

        # layer 10
        action = policy[5]
        residual = self.rnet.ds[1](x)
        if torch.sum(action) > 0.0:
            action_mask = action.view(-1,1,1,1)
            fx = self.rnet.blocks[1][0](x)
            fx = torch.relu(residual + fx)
            x = fx*action_mask + residual*(1.0-action_mask)
        else:
            x = residual
        
        # layer 11
        action = policy[6]
        residual = x
        if torch.sum(action) > 0.0:
            action_mask = action.view(-1,1,1,1)
            fx = self.rnet.blocks[1][1](x)
            fx = torch.relu(residual + fx)
            x = fx*action_mask + residual*(1.0-action_mask)

        # layer 12
        action = policy[7]
        residual = x
        if torch.sum(action) > 0.0:
            action_mask = action.view(-1,1,1,1)
            fx = self.rnet.blocks[1][2](x)
            fx = torch.relu(residual + fx)
            x = fx*action_mask + residual*(1.0-action_mask)

        # layer 13
        action = policy[8]
        residual = x
        if torch.sum(action) > 0.0:
            action_mask = action.view(-1,1,1,1)
            fx = self.rnet.blocks[1][3](x)
            fx = torch.relu(residual + fx)
            x = fx*action_mask + residual*(1.0-action_mask)

        # layer 14
        action = policy[9]
        residual = x
        if torch.sum(action) > 0.0:
            action_mask = action.view(-1,1,1,1)
            fx = self.rnet.blocks[1][4](x)
            fx = torch.relu(residual + fx)
            x = fx*action_mask + residual*(1.0-action_mask)

        # layer 20
        action = policy[10]
        residual = self.rnet.ds[2](x)
        if torch.sum(action) > 0.0:
            action_mask = action.view(-1,1,1,1)
            fx = self.rnet.blocks[2][0](x)
            fx = torch.relu(residual + fx)
            x = fx*action_mask + residual*(1.0-action_mask)
        else:
            x = residual
        
        # layer 21
        action = policy[11]
        residual = x
        if torch.sum(action) > 0.0:
            action_mask = action.view(-1,1,1,1)
            fx = self.rnet.blocks[2][1](x)
            fx = torch.relu(residual + fx)
            x = fx*action_mask + residual*(1.0-action_mask)

        # layer 22
        action = policy[12]
        residual = x
        if torch.sum(action) > 0.0:
            action_mask = action.view(-1,1,1,1)
            fx = self.rnet.blocks[2][2](x)
            fx = torch.relu(residual + fx)
            x = fx*action_mask + residual*(1.0-action_mask)

        # layer 23
        action = policy[13]
        residual = x
        if torch.sum(action) > 0.0:
            action_mask = action.view(-1,1,1,1)
            fx = self.rnet.blocks[2][3](x) 
            fx = torch.relu(residual + fx)
            x = fx*action_mask + residual*(1.0-action_mask)

        # layer 24
        action = policy[14]
        residual = x
        if torch.sum(action) > 0.0:
            action_mask = action.view(-1,1,1,1)
            fx = self.rnet.blocks[2][4](x)
            fx = torch.relu(residual + fx)
            x = fx*action_mask + residual*(1.0-action_mask)

        x = self.rnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.rnet.fc(x)
        return x

    def forward_skip(self, inputs, probs):
        # probs, _ = self.agent(inputs)

        cond = torch.lt(probs, torch.full_like(probs, 0.5))
        policy = torch.where(cond, torch.zeros_like(probs), torch.ones_like(probs))
        policy = policy.transpose(0, 1)

        # FlatResNet
        x = self.rnet.seed(inputs)

        # layer 00
        residual = self.rnet.ds[0](x)
        x = residual

        # layer 01
        action = policy[1]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[0][1](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 02
        residual = x

        # layer 03
        action = policy[3]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[0][3](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 04
        residual = x

        # layer 10
        action = policy[5]
        residual = self.rnet.ds[1](x)
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[1][0](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 11
        residual = x

        # layer 12
        action = policy[7]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[1][2](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 13
        residual = x

        # layer 14
        action = policy[9]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[1][4](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 20
        residual = self.rnet.ds[2](x)
        x = residual
        
        # layer 21
        action = policy[11]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[2][1](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 22
        residual = x

        # layer 23
        action = policy[13]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[2][3](x) 
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 24
        residual = x

        x = self.rnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.rnet.fc(x)
        return x

    def forward_unroll_0(self, inputs, probs):
        cond = torch.lt(probs, torch.full_like(probs, 0.5))
        policy = torch.where(cond, torch.zeros_like(probs), torch.ones_like(probs))
        policy = policy.transpose(0, 1)

        # FlatResNet
        x = self.rnet.seed(inputs)

        # layer 00
        residual = self.rnet.ds[0](x)
        x = residual

        # layer 10
        residual = self.rnet.ds[1](x)
        x = residual
        
        # layer 20
        residual = self.rnet.ds[2](x)
        x = residual
        
        x = self.rnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.rnet.fc(x)
        return x, policy

    def forward_unroll_25(self, inputs, probs):
        cond = torch.lt(probs, torch.full_like(probs, 0.5))
        policy = torch.where(cond, torch.zeros_like(probs), torch.ones_like(probs))
        policy = policy.transpose(0, 1)

        # FlatResNet
        x = self.rnet.seed(inputs)

        # layer 00
        residual = self.rnet.ds[0](x)
        x = residual

        # layer 02
        action = policy[2]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[0][2](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 10
        residual = self.rnet.ds[1](x)
        x = residual
        
        # layer 11
        action = policy[6]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[1][1](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 13
        action = policy[8]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[1][3](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 20
        action = policy[10]
        residual = self.rnet.ds[2](x)
        x = residual
        
        # layer 21
        action = policy[11]
        residual = x

        # layer 22
        action = policy[12]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[2][2](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 23
        action = policy[13]
        residual = x

        # layer 24
        action = policy[14]
        residual = x

        x = self.rnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.rnet.fc(x)
        return x

    def forward_unroll_50(self, inputs, probs):
        cond = torch.lt(probs, torch.full_like(probs, 0.5))
        policy = torch.where(cond, torch.zeros_like(probs), torch.ones_like(probs))
        policy = policy.transpose(0, 1)

        # FlatResNet
        x = self.rnet.seed(inputs)

        # layer 00
        action = policy[0]
        residual = self.rnet.ds[0](x)
        x = residual

        # layer 01
        action = policy[1]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[0][1](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 03
        action = policy[3]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[0][3](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 04
        action = policy[4]
        residual = x

        # layer 10
        action = policy[5]
        residual = self.rnet.ds[1](x)
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[1][0](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)
        
        # layer 12
        action = policy[7]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[1][2](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 14
        action = policy[9]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[1][4](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 20
        action = policy[10]
        residual = self.rnet.ds[2](x)
        x = residual
        
        # layer 21
        action = policy[11]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[2][1](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 23
        action = policy[13]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[2][3](x) 
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        x = self.rnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.rnet.fc(x)
        return x

    def forward_unroll_75(self, inputs, probs):
        cond = torch.lt(probs, torch.full_like(probs, 0.5))
        policy = torch.where(cond, torch.zeros_like(probs), torch.ones_like(probs))
        policy = policy.transpose(0, 1)

        # FlatResNet
        x = self.rnet.seed(inputs)

        # layer 00
        action = policy[0]
        residual = self.rnet.ds[0](x)
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[0][0](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)
        
        # layer 01
        action = policy[1]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[0][1](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 03
        action = policy[3]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[0][3](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 04
        action = policy[4]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[0][4](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 10
        action = policy[5]
        residual = self.rnet.ds[1](x)
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[1][0](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)
        
        # layer 12
        action = policy[7]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[1][2](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 14
        action = policy[9]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[1][4](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 20
        action = policy[10]
        residual = self.rnet.ds[2](x)
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[2][0](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)
        
        # layer 21
        action = policy[11]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[2][1](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 23
        action = policy[13]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[2][3](x) 
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 24
        action = policy[14]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[2][4](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        x = self.rnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.rnet.fc(x)
        return x

    def forward_unroll_100(self, inputs, probs):
        # probs, _ = self.agent(inputs)

        cond = torch.lt(probs, torch.full_like(probs, 0.5))
        policy = torch.where(cond, torch.zeros_like(probs), torch.ones_like(probs))
        policy = policy.transpose(0, 1)

        # FlatResNet
        x = self.rnet.seed(inputs)

        # layer 00
        action = policy[0]
        residual = self.rnet.ds[0](x)
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[0][0](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)
        
        # layer 01
        action = policy[1]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[0][1](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 02
        action = policy[2]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[0][2](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 03
        action = policy[3]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[0][3](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 04
        action = policy[4]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[0][4](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 10
        action = policy[5]
        residual = self.rnet.ds[1](x)
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[1][0](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)
        
        # layer 11
        action = policy[6]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[1][1](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 12
        action = policy[7]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[1][2](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 13
        action = policy[8]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[1][3](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 14
        action = policy[9]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[1][4](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 20
        action = policy[10]
        residual = self.rnet.ds[2](x)
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[2][0](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)
        
        # layer 21
        action = policy[11]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[2][1](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 22
        action = policy[12]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[2][2](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 23
        action = policy[13]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[2][3](x) 
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        # layer 24
        action = policy[14]
        residual = x
        action_mask = action.view(-1,1,1,1)
        fx = self.rnet.blocks[2][4](x)
        fx = torch.relu(residual + fx)
        x = fx*action_mask + residual*(1.0-action_mask)

        x = self.rnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.rnet.fc(x)
        return x

def build_model(batch_size, probs, unroll, rate):
    layer_config = [5, 5, 5]
    rnet = FlatResNet32(BasicBlock, layer_config, num_classes=10)
    agent = Policy32([1,1,1], num_blocks=15)
    rnet.eval().cuda()
    agent.eval().cuda()
    torch.manual_seed(0)
    # load_checkpoint(rnet, agent, os.path.expanduser('../../data/blockdrop/ckpt_E_730_A_0.913_R_2.73E-01_S_6.92_#_53.t7'))
    model = BlockDrop(rnet, agent).eval()
    # preprocess(model)
    len_dataset = 10000
    if unroll:
        if rate == -1:
            model.forward = model.forward_skip
            print("run model.forward_skip")
        else:
            model.forward = getattr(model, f"forward_unroll_{rate}")
            print(f"run model.forward_unroll_{rate}")
    else:
        model.forward = model.forward_real
        print("run model.forward_real", probs)
    model = torch.jit.script(model)
    inp_np = np.random.randn(batch_size, 3, 32, 32)
    inp = torch.from_numpy(inp_np).to(cuda_device).to(torch.float32)
    ## paramemters replace here

    for parameter in model.parameters():  
        random_array = np.random.uniform(-0.5,0.5,parameter.data.shape)
        # if parameter.data.dim() > 1:
        #     xavier_uniform_(random_array)
        # if parameter.data.dim() > 1:
        #     nn.init.xavier_uniform_(parameter.data)
        tensor = torch.from_numpy(random_array).to(cuda_device).to(torch.float32)
        parameter.data = tensor
    return model, inp, probs


def test_all():
    if arguments.rate == -1:
        actions = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    elif arguments.rate == 0:
        actions = [0] * 15
    elif arguments.rate == 25:
        actions = [
            0, 0, 1, 0, 0,
            0, 1, 0, 1, 0,
            0, 0, 1, 0, 0,
        ]
    elif arguments.rate == 50:
        actions = [
            0, 1, 0, 1, 0,
            1, 0, 1, 0, 1,
            0, 1, 0, 1, 0,
        ]
    elif arguments.rate == 75:
        actions = [
            1, 1, 0, 1, 1,
            1, 0, 1, 0, 1,
            1, 1, 0, 1, 1,
        ]
    elif arguments.rate == 100:
        actions = [1] * 15
    actions = torch.tensor(actions, dtype=torch.float32).reshape(-1, 15).cuda()
    model, inp,probs = build_model(1, actions, arguments.unroll, arguments.rate)
    test_with_grad(model,inp.clone(),probs.clone())
    test_with_backward(model,inp.clone(),probs.clone())
    test_res_accuracy(model, inp.clone(),probs.clone())
    test_grad_accuracy(model,inp.clone(),probs.clone())

def test_with_grad(model,inp,probs):
    torch.set_grad_enabled(True)
    print("\033[91m----test-with-grad----\033[0m")
    for i in range(0, 10000, 1):
        if i >= n_warmup * 1: break
        torch.cuda.synchronize()
        t0 = time()
        _ = model(inp, probs)
        torch.cuda.synchronize()
        t1 = time()
        # print("time", t1 - t0)
    # run
    print("[run]",end="")
    timer = Timer("ms")
    profile_start(use_profile)
    for i in range(0, 10000, 1):
        if i >= n_run * 1: break
        torch.cuda.synchronize()
        timer.start()
        _ = model(inp, probs)
        torch.cuda.synchronize()
        timer.log()
    profile_stop(use_profile)
    timer.report()


def test_with_backward(model,inp,probs):
    torch.set_grad_enabled(True)
    print("\033[91m----test-with-backward----\033[0m")
    for i in range(0, 10000, 1):
        if i >= n_warmup * 1: break
        torch.cuda.synchronize()
        t0 = time()
        out = model(inp, probs)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        t1 = time()
        # print("time", t1 - t0)
    # run
    print("[run]",end="")
    timer = Timer("ms")
    profile_start(use_profile)
    for i in range(0, 10000, 1):
        if i >= n_run * 1: break
        torch.cuda.synchronize()
        timer.start()
        out = model(inp, probs)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        timer.log()
    profile_stop(use_profile)
    timer.report()

def test_res_accuracy(model,inp,probs):
    with torch.no_grad():
        print("\033[92m----test-res-accuracy----\033[0m")
        #warm_up

        torch.cuda.synchronize()
        out = model(inp, probs)
        res = out.cpu().numpy().mean().reshape(-1, 1)
        print("res:",res)
        np.savetxt('res.txt',res)

def test_grad_accuracy(model,inp,probs):
    print("\033[92m----test-grad-accuracy----\033[0m")
    out = model(inp, probs)
    out = out.sum()
    out.backward(retain_graph=True)
    grad_res = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad = torch.autograd.grad(out, param, create_graph=True,allow_unused=True)
            if isinstance(grad[0], type(None)):
                grad_res.append(0)
            else:
                grad_res.append(grad[0].detach().cpu().numpy().sum())
            # print(f'Gradient check for {name}')
    print(grad_res)
    np.savetxt('res_grad.txt',grad_res)

if __name__ == "__main__":
    test_all()
