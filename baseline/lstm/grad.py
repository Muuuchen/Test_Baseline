"""
 *@description: check grad
 *@author: haozhaoyang
 *@date: 2023-10-30
 *@idea:  model just test roll
 """

from typing import Tuple
import torch
import torch.nn as nn
from time import time
import os
import numpy as np
from lstm_pytorch_unroll import LSTMUnroll

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--use_profile', type=bool,default=True)
parser.add_argument('--overhead_test', action='store_true')
parser.add_argument('--unroll', dest='unroll', action='store_true')
parser.add_argument('--fix', dest='unroll', action='store_false')
parser.set_defaults(unroll=False)
arguments = parser.parse_args()
use_profile = arguments.use_profile

import sys
sys.path.append('../utils')
from benchmark_timer import Timer
from nsysprofile import profile_start, profile_stop, enable_profile
enable_profile(use_profile)

n_warmup = 100
n_run = 100
cuda_device = torch.device("cuda:0")

# # 设置全局随机数种子
# torch.manual_seed(196)
# # 如果使用GPU，还需要设置 GPU 的随机数种子
# torch.cuda.manual_seed_all(196)
# # 设置CUDNN以确保一致性
# torch.backends.cudnn.deterministic = True
np.random.seed(196)

def xavier_uniform_(arr):
    fan_in, fan_out = arr.shape[0], arr.shape[1]
    limit = np.sqrt(6 / (fan_in + fan_out))
    arr.uniform_(-limit, limit)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.LSTMCell(input_size, hidden_size))
        for i in range(num_layers - 1):
            self.layers.append(nn.LSTMCell(hidden_size, hidden_size))
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, inputs):  # seq_len, batch, input_size
        batch_size = inputs.shape[1]
        state_c = [torch.zeros(batch_size, self.hidden_size, device='cuda') for _ in range(10)] # hardcode for ts compile
        state_h = [torch.zeros(batch_size, self.hidden_size, device='cuda') for _ in range(10)]
        for i in range(inputs.size()[0]):
            cur_input = inputs[i]
            for j, layer in enumerate(self.layers):
                c = state_c[j]
                h = state_h[j]
                c, h = layer(cur_input, (c, h))

                state_c[j] = c
                state_h[j] = h
                cur_input = h
        return state_h[self.num_layers - 1]

def build_model(enable_torch, batch_size, *params):
    input_size = 256
    hidden_size = 256
    num_layers = 10
    seq_len = 64

    model = LSTM(input_size, hidden_size, num_layers).to(cuda_device)
    if enable_torch:
        model = torch.jit.script(model)
    inp_np = np.random.randn(seq_len, batch_size, input_size)
    inp = torch.from_numpy(inp_np).to(cuda_device).to(torch.float32)
    ## quanhzong tihuan here
    for parameter in model.parameters():  
        random_array = np.random.uniform(-1,1,parameter.data.shape)
        # if parameter.data.dim() > 1:
        #     xavier_uniform_(random_array)
        # if parameter.data.dim() > 1:
        #     nn.init.xavier_uniform_(parameter.data)
        tensor = torch.from_numpy(random_array).to(cuda_device).to(torch.float32)
        parameter.data = tensor

        # print(parameter.data)
    return model, inp

def test_res_accuracy(model,inp):
    with torch.no_grad():
        print("\033[92m----test-res-accuracy----\033[0m")
        #warm_up

        torch.cuda.synchronize()
        out = model(inp)
        res = out.cpu().numpy().mean().reshape(-1, 1)
        print("res:",res)
        np.savetxt('res.txt',res)



def test_grad_accuracy(model,inp):
    print("\033[92m----test-grad-accuracy----\033[0m")
    out = model(inp)
    out = out.sum()
    out.backward(retain_graph=True)
    grad_res = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad = torch.autograd.grad(out, param, create_graph=True)
            grad_res.append(grad[0].detach().cpu().numpy().sum())
            # print(f'Gradient check for {name}')
    print(grad_res)
    np.savetxt('res_grad.txt',grad_res)


def test_with_grad (model,inp):
    torch.set_grad_enabled(True)
    print("\033[91m----test-with-grad----\033[0m")
    #warm_up
    torch.cuda.synchronize()
    for i in range(n_warmup):
        t0 = time()
        inp_ = model(inp)
        torch.cuda.synchronize()
        # print("[warm up] Time {} ms".format((time() - t0) * 1000))
    
    print("[run]",end="")
    timer = Timer("ms")
    torch.cuda.synchronize()
    for i in range(n_run):
        timer.start()
        _ = model(inp)
        torch.cuda.synchronize()
        timer.log()
    timer.report()

def test_with_backward(model,inp):
    torch.set_grad_enabled(True)
    print("\033[91m----test-with-backward----\033[0m")

    torch.cuda.synchronize()
    #warm_up
    for i in range(n_warmup):
        t0 = time()
        out = model(inp)
        #loss 及反向传播
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        # print("[warm up] Time {} ms".format(((time() - t0) * 1000)))

    print("[run]",end="")
    timer = Timer("ms")
    torch.cuda.synchronize()
    for i in range(n_run):
        timer.start()
        out = model(inp)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        timer.log()
    timer.report()


def test_all():

    model, inp = build_model(True, 1)
    test_with_grad(model,inp.clone())
    test_with_backward(model,inp.clone())
    test_res_accuracy(model, inp.clone())
    test_grad_accuracy(model,inp.clone())


if __name__ == "__main__":
    test_all()