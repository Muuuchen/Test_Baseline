"""
 *@description: Seq2seq 无法处理outputall 因为inplace操作因此考虑对h的梯度在正确性的检查下
 *@author: haozhaoyang
 *@date: 2023-11-1
 *@idea: accuracy - same seed
 """
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import os
from time import time


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--use_profile', type=bool,default=True)
parser.add_argument('--unroll', dest='unroll', action='store_true')
parser.add_argument('--fix', dest='unroll', action='store_false')
parser.set_defaults(unroll=False)
parser.add_argument('--rate', type=int, default=-1)
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--platform', type=str)
arguments = parser.parse_args()
use_profile = arguments.use_profile

import sys
sys.path.append('../utils')
from benchmark_timer import Timer
from nsysprofile import profile_start, profile_stop, enable_profile
enable_profile(use_profile)


n_warmup = 100
n_run = 100


MAX_LENGTH = 50
OUTPUT_SIZE = 3797
HIDDEN_SIZE = 256

cuda_device = torch.device('cuda:0')
# torch.manual_seed(196)
# # 如果使用GPU，还需要设置 GPU 的随机数种子
# torch.cuda.manual_seed_all(196)
# # 设置CUDNN以确保一致性
# torch.backends.cudnn.deterministic = True
np.random.seed(196)


class LSTMCell(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.weight_ih_l0_t = nn.Parameter(torch.randn(4, input_size, hidden_size, dtype=torch.float32))
        self.weight_hh_l0_t = nn.Parameter(torch.randn(4, input_size, hidden_size, dtype=torch.float32))
        self.bias_ih_0 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_hh_0 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_ih_1 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_hh_1 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_ih_2 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_hh_2 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_ih_3 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_hh_3 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.hidden_size = hidden_size
        self.input_size = input_size
        nn.init.xavier_uniform_(self.weight_ih_l0_t)
        nn.init.xavier_uniform_(self.weight_hh_l0_t)
def forward(self, x, h, c):
        ih = torch.matmul(x, self.weight_ih_l0_t)
        hh = torch.matmul(h, self.weight_hh_l0_t)
        ih0 = ih[0] + self.bias_ih_0
        hh0 = hh[0] + self.bias_hh_0
        ih1 = ih[1] + self.bias_ih_1
        hh1 = hh[1] + self.bias_hh_1
        ih2 = ih[2] + self.bias_ih_2
        hh2 = hh[2] + self.bias_hh_2
        ih3 = ih[3] + self.bias_ih_3
        hh3 = hh[3] + self.bias_hh_3

        ingate = torch.sigmoid(ih0 + hh0)
        forgetgate = torch.sigmoid(ih1 + hh1)
        cellgate = torch.tanh(ih2 + hh2)
        outgate = torch.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * torch.tanh(c)
        return h, c


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.gru = LSTMCell(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # self.EOS_token = nn.Parameter(torch.full((1,), 0, dtype=torch.long, device=device), requires_grad=False)
        self.EOS_token = 0
        self.SOS_token = 1

    def forward(self, encoder_output, std, h, c):
        # hidden: (1, bs, hidden_size)
        # encoder_outputs: (max_length, bs, hidden_size)
        batch_size = encoder_output.size()[1]
        output_all = torch.zeros(self.max_length, batch_size, dtype=torch.int64, device='cuda') + 0 # Hack for bug in ScatterND on Constant
        output = torch.full((batch_size,), self.SOS_token, dtype=torch.int64, device='cuda')
        cond = True
        # when disable cf
        # id = torch.zeros((), dtype=torch.int64, device='cuda')
        id = 0
        while cond:
            x = self.embedding(output)
            # h = torch.reshape(h, (batch_size, self.hidden_size))
            # lstm start
            ih = torch.matmul(x, self.gru.weight_ih_l0_t)
            hh = torch.matmul(h, self.gru.weight_hh_l0_t)
            ih0 = ih[0] + self.gru.bias_ih_0
            hh0 = hh[0] + self.gru.bias_hh_0
            ih1 = ih[1] + self.gru.bias_ih_1
            hh1 = hh[1] + self.gru.bias_hh_1
            ih2 = ih[2] + self.gru.bias_ih_2
            hh2 = hh[2] + self.gru.bias_hh_2
            ih3 = ih[3] + self.gru.bias_ih_3
            hh3 = hh[3] + self.gru.bias_hh_3

            ingate = torch.sigmoid(ih0 + hh0)
            forgetgate = torch.sigmoid(ih1 + hh1)
            cellgate = torch.tanh(ih2 + hh2)
            outgate = torch.sigmoid(ih3 + hh3)

            c = (forgetgate * c) + (ingate * cellgate)
            h = outgate * torch.tanh(c)
            # lstm end
            output = self.out(h) + std[id]
            output = output.argmax(1)
            output_all[id] = output
            id = id + 1
            cond = bool((torch.max(output) > self.EOS_token).item()) & (id < self.max_length) # when testing torchscript
            # cond = (torch.max(output) > self.EOS_token) & (id < self.max_length)
        return output_all, h


def gen_mask_from_sequence(std):
    bs = std.shape[0]
    padded_std = torch.zeros((bs, MAX_LENGTH), dtype=std.dtype, device=cuda_device)
    padded_std[:, :std.shape[1]] = std
    mask = torch.zeros(bs, MAX_LENGTH, OUTPUT_SIZE, device=cuda_device)
    mask[torch.arange(bs).unsqueeze(1), torch.arange(MAX_LENGTH).unsqueeze(0), padded_std] = 1000000.0
    mask = mask.transpose(0, 1).contiguous().clone()
    return mask


def build_model():
    batch_size =1
    model = AttnDecoderRNN(HIDDEN_SIZE, OUTPUT_SIZE, dropout_p=0.1).to(cuda_device)
    #model 
    model = torch.jit.script(model)
    #data
    std = []
    MAX_LENGTH = 50

    h_np = np.random.randn(batch_size, HIDDEN_SIZE)
    h = torch.from_numpy(h_np).to(cuda_device).to(torch.float32)

    c_np = np.random.randn(batch_size, HIDDEN_SIZE)
    c = torch.from_numpy(c_np).to(cuda_device).to(torch.float32)

    # h = torch.randn(batch_size, HIDDEN_SIZE, device=device)
    # c = torch.randn(batch_size, HIDDEN_SIZE, device=device)
    sos = torch.full((batch_size,), model.SOS_token, dtype=torch.int64, device=cuda_device)
    for i in range(batch_size):
        l = 10
        lst = list(range(1, l))
        lst.append(0)
        assert(len(lst) <= MAX_LENGTH)
        # pad to MAX_LENGTH
        lst = lst + [0] * (MAX_LENGTH - len(lst))
        std.append(lst)
    std = torch.tensor(std, device=cuda_device)
    mask = gen_mask_from_sequence(std)

    encoder_output_np = np.random.randn(MAX_LENGTH, batch_size, HIDDEN_SIZE)
    encoder_output = torch.from_numpy(encoder_output_np).to(cuda_device).to(torch.float32)
    args = (encoder_output, mask, h, c)
    for parameter in model.parameters():  
        random_array = np.random.uniform(-0.1,0.1,parameter.data.shape)
        # if parameter.data.dim() > 1:
        #     xavier_uniform_(random_array)
        # if parameter.data.dim() > 1:
        #     nn.init.xavier_uniform_(parameter.data)
        tensor = torch.from_numpy(random_array).to(cuda_device).to(torch.float32)
        parameter.data = tensor


    return model,args


def test_with_grad (model,args):
    torch.set_grad_enabled(True)
    print("\033[91m----test-with-grad----\033[0m")
    #warm_up
    torch.cuda.synchronize()
    for i in range(n_warmup):
        t0 = time()
        _ = model(*args)
        torch.cuda.synchronize()
        # print("[warm up] Time {} ms".format((time() - t0) * 1000))

    print("[run]",end="")
    timer = Timer("ms")
    torch.cuda.synchronize()
    for i in range(n_run):
        timer.start()
        _ = model(*args)
        torch.cuda.synchronize()
        timer.log()
    timer.report()



def test_with_backward(model,args):
    torch.set_grad_enabled(True)
    print("\033[91m----test-with-backward----\033[0m")
    #warm_up
    torch.cuda.synchronize()
    for i in range(n_warmup):
        t0 = time()
        _,out = model(*args)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        # print("[warm up] Time {} ms".format((time() - t0) * 1000))

    print("[run]",end="")
    timer = Timer("ms")
    torch.cuda.synchronize()
    for i in range(n_run):
        timer.start()
        _,out = model(*args)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        timer.log()
    timer.report()



def test_res_accuracy(model,args):
    with torch.no_grad():
        print("\033[92m----test-res-accuracy----\033[0m")
        #warm_up

        torch.cuda.synchronize()
        _,out = model(*args)
        res = out.cpu().numpy().mean().reshape(-1, 1)
        print("res:",res)
        np.savetxt('res.txt',res)


def test_grad_accuracy(model,args):
    print("\033[92m----test-grad-accuracy----\033[0m")
    _, out = model(*args)
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

            # grad_res.append(grad[0].detach().cpu().numpy().sum())
            # print(f'Gradient check for {name}')
    print(grad_res)
    np.savetxt('res_grad.txt',grad_res)

def test_all():

    model, args = build_model()
    # test_with_grad(model,args)
    # test_with_backward(model,args)hei
    test_res_accuracy(model,args)
    test_grad_accuracy(model,args)

if __name__ == '__main__':
    test_all()
