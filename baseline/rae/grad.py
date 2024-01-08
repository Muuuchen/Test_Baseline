"""
 *@description: no script !!!
 *@author: haozhaoyang
 *@date: 2023-10-30
 *@idea:  model just test roll
 """
import torch
import torch.nn as nn
from rae_pytorch_unroll import RAEUnroll
import numpy as np
from time import time


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
# torch.manual_seed(196)
# # 如果使用GPU，还需要设置 GPU 的随机数种子
# torch.cuda.manual_seed_all(196)
# # 设置CUDNN以确保一致性
# torch.backends.cudnn.deterministic = True
np.random.seed(196)


n_warmup = 100
n_run = 100

depth = 7
n = 2 ** depth - 1


class RAE(nn.Module):
    def __init__(self):
        super(RAE, self).__init__()
        self.encoder = nn.Linear(1024, 512)

    def forward(self, left, right, is_leaf, inp, root):
        if is_leaf[root]:
            output = inp[root] # (h,)
        else:
            a = self.forward(left, right, is_leaf, inp, left[root].item()) # (h,)
            b = self.forward(left, right, is_leaf, inp, right[root].item()) # (h,)
            ab = torch.cat((a, b)) # (2h,)
            e = self.encoder(ab)
            output = torch.tanh(e)
        # print(root, output)
        return output
    

class RAECell(nn.Module):
    def __init__(self):
        super(RAECell, self).__init__()
        self.encoder = nn.Linear(1024, 512)

    def forward(self, a, b):
        ab = torch.cat((a, b)) # (2h,)
        e = self.encoder(ab)
        output = torch.tanh(e)
        return output

class RAEScript(nn.Module):
    def __init__(self):
        super(RAEScript, self).__init__()
        self.cell = torch.jit.script(RAECell())
    
    def forward(self, left, right, is_leaf, inp, root):
        if is_leaf[root]:
            output = inp[root] # (h,)
        else:
            a = self.forward(left, right, is_leaf, inp, left[root].item()) # (h,)
            b = self.forward(left, right, is_leaf, inp, right[root].item()) # (h,)
            output = self.cell(a, b)
        return output 

def build_model():
    batch_size = 1
    # model = RAEScript().to(cuda_device)

    root = 64
    left = torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 30, 28, 27, 26, 33, 25, 24, 23, 22, 21, 20, 32, 18, 17, 15, 14, 13, 12, 11, 46, 8, 6, 5, 53, 10, 52, 2, 1, 0, 4, 58, 44])
    right = torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 31, 29, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 19, 45, 16, 47, 48, 49, 50, 51, 9, 7, 54, 55, 56, 57, 3, 59, 60, 61, 62, 63])
    is_leaf = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    x = torch.ones([n, 512], device=cuda_device)
    left = left.to(cuda_device)
    right = right.to(cuda_device)
    is_leaf = is_leaf.to(cuda_device)
    # model = torch.jit.script(model)
    model = RAE().to(cuda_device)

    x = torch.ones([n, 512], device=cuda_device)
    args = (left, right, is_leaf, x, root)
    for parameter in model.parameters():  
        random_array = np.random.uniform(-0.1,0.1,parameter.data.shape)
        # if parameter.data.dim() > 1:
        #     xavier_uniform_(random_array)
        # if parameter.data.dim() > 1:
        #     nn.init.xavier_uniform_(parameter.data)
        tensor = torch.from_numpy(random_array).to(cuda_device).to(torch.float32)
        parameter.data = tensor
    return model, args



def test_with_grad (model,args):
    torch.set_grad_enabled(True)
    print("\033[91m----test-with-grad----\033[0m")
    #warm_up
    torch.cuda.synchronize()
    for i in range(n_warmup):
        t0 = time()
        inp_ = model(*args)
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

    torch.cuda.synchronize()
    #warm_up
    for i in range(n_warmup):
        t0 = time()
        out = model(*args)
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
        out = model(*args)
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
        out = model(*args)
        res = out.cpu().numpy().mean().reshape(-1, 1)
        print("res:",res)
        np.savetxt('res.txt',res)


def test_grad_accuracy(model,args):
    print("\033[92m----test-grad-accuracy----\033[0m")
    out = model(*args)
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
    model,args = build_model()
    test_with_grad(model, args)
    test_with_backward(model,args)
    test_res_accuracy(model, args)
    test_grad_accuracy(model,args)       


if __name__ == "__main__":
    test_all()

