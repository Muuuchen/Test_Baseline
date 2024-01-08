"""
 *@description: Nasrnn
 *@author: haozhaoyang
 *@date: 2023-11-1
 *@idea: accuracy - same seed
 """
import numpy as np
import torch
import torch.nn as nn
from time import time
from nas_pytorch_unroll import NasRNNUnroll
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

cuda_device = torch.device("cuda:0")
n_warmup = 100
n_run = 100
# torch.manual_seed(196)
# # 如果使用GPU，还需要设置 GPU 的随机数种子
# torch.cuda.manual_seed_all(196)
# # 设置CUDNN以确保一致性
# torch.backends.cudnn.deterministic = True
np.random.seed(196)


class NasRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NasRNN, self).__init__()
        self.weight_ih = nn.Parameter(torch.randn(
            8, input_size, hidden_size, dtype=torch.float32))
        self.weight_hh = nn.Parameter(torch.randn(
            8, hidden_size, hidden_size, dtype=torch.float32))
        self.hidden_size = hidden_size
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)

    def forward(self, inputs):  # seq_len, batch, input_size
        batch_size = inputs.shape[1]
        state_c = torch.ones(batch_size, self.hidden_size, device='cuda')
        state_m = torch.ones(batch_size, self.hidden_size, device='cuda')
        for i in range(inputs.size()[0]):
            inp = inputs[i]

            ih = torch.matmul(inp, self.weight_ih)
            hh = torch.matmul(state_m, self.weight_hh)

            i0 = ih[0]
            i1 = ih[1]
            i2 = ih[2]
            i3 = ih[3]
            i4 = ih[4]
            i5 = ih[5]
            i6 = ih[6]
            i7 = ih[7]

            h0 = hh[0]
            h1 = hh[1]
            h2 = hh[2]
            h3 = hh[3]
            h4 = hh[4]
            h5 = hh[5]
            h6 = hh[6]
            h7 = hh[7]

            layer1_0 = torch.sigmoid(i0 + h0)
            layer1_1 = torch.relu(i1 + h1)
            layer1_2 = torch.sigmoid(i2 + h2)
            layer1_3 = torch.relu(i3 + h3)
            layer1_4 = torch.tanh(i4 + h4)
            layer1_5 = torch.sigmoid(i5 + h5)
            layer1_6 = torch.tanh(i6 + h6)
            layer1_7 = torch.sigmoid(i7 + h7)

            l2_0 = torch.tanh(layer1_0 * layer1_1)
            l2_1 = torch.tanh(layer1_2 + layer1_3)
            l2_2 = torch.tanh(layer1_4 * layer1_5)
            l2_3 = torch.sigmoid(layer1_6 + layer1_7)

            # Inject the cell
            l2_0_v2 = torch.tanh(l2_0 + state_c)

            # Third layer
            state_c = l2_0_v2 * l2_1
            l3_1 = torch.tanh(l2_2 + l2_3)

            # Final layer
            state_m = torch.tanh(state_c * l3_1)

        return state_m
    

def build_model():
    input_size = 256
    hidden_size = 256
    seq_len = 1000

    batch_size = 1
    model = NasRNN(input_size, hidden_size).to(cuda_device)
    model = torch.jit.script(model)
    inp_np = np.random.randn(seq_len, batch_size, input_size)
    inp = torch.from_numpy(inp_np).to(cuda_device).to(torch.float32)
    for parameter in model.parameters():  
        random_array = np.random.uniform(-0.1,0.1,parameter.data.shape)
        # if parameter.data.dim() > 1:
        #     xavier_uniform_(random_array)
        # if parameter.data.dim() > 1:
        #     nn.init.xavier_uniform_(parameter.data)
        tensor = torch.from_numpy(random_array).to(cuda_device).to(torch.float32)
        parameter.data = tensor
    return model, inp

def test_res_accuracy(model,inp):
    with torch.no_grad():
        print("\033[92m----test-res-accuracy----\033[0m")

        torch.cuda.synchronize()
        out = model(inp)
        res = out.cpu().numpy().mean().reshape(-1, 1)
        print("res:",res)
        np.savetxt('res.txt',res)

def test_with_grad (model,inp):
    torch.set_grad_enabled(True)
    print("\033[91m----test-with-grad----\033[0m")
    #warm_up
    torch.cuda.synchronize()
    for i in range(n_warmup):
        t0 = time()
        _ = model(inp)
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





def test_all():

    model, inp = build_model()
    test_with_grad(model,inp.clone())
    test_with_backward(model,inp.clone())
    test_res_accuracy(model, inp.clone())
    test_grad_accuracy(model,inp.clone())


if __name__ == "__main__":
    test_all()