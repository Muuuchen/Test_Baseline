"""
 *@description: Attention forward backward accuracy
 *@author: haozhaoyang
 *@date: 2023-10-24
 *@idea: accuracy - same seed
1. 推理时间 √  torch script / tf function graph (no eager) /  jit jax
2. 带梯度推理时间 √ 同样在图优化的结果下， 加一个接口实现不同测试（带梯度，不代梯度)
3. 推理结果准确性 <相同随机种子，推理结果>  Cong np zhuanhuan
4. 梯度计算准确性 
正确性验证文件中是否我们不需要重新构建一次

 """
import numpy as np
import torch
import torch.nn as nn
from time import time
from attention_pytorch_unroll import AttentionUnroll
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--use_profile', type=bool,default=True)
parser.add_argument('--script', type=bool,default=True)
parser.add_argument('--overhead_test', action='store_true')
parser.add_argument('--unroll', dest='unroll', action='store_true')
parser.add_argument('--fix', dest='unroll', action='store_false')
parser.set_defaults(unroll=False)
arguments = parser.parse_args()
use_profile = arguments.use_profile
enable_torch = arguments.script

import sys
sys.path.append('../utils')
from benchmark_timer import Timer
from nsysprofile import profile_start, profile_stop, enable_profile
enable_profile(use_profile)

START_LEN = 32
SEQ_LEN = 64
NUM_HEAD = 12
SIZE_PER_HEAD = 64

cuda_device = torch.device("cuda:0")
# print(cuda_device)
n_warmup = 100
n_run = 100

import torch

# 设置全局随机数种子
# torch.manual_seed(196)
# # 如果使用GPU，还需要设置 GPU 的随机数种子
# torch.cuda.manual_seed_all(196)
# # 设置CUDNN以确保一致性
# torch.backends.cudnn.deterministic = True
np.random.seed(196)

class Attention(nn.Module):
    def __init__(self, num_head, size_per_head):
        super().__init__()
        np_q = np.random.randn(num_head, size_per_head, size_per_head).astype(np.float32)
        np_k = np.random.randn(num_head, size_per_head, size_per_head).astype(np.float32)
        np_v = np.random.randn(num_head, size_per_head, size_per_head).astype(np.float32)
        np_o = np.random.randn(num_head, size_per_head, size_per_head).astype(np.float32)

        self.weight_q =  nn.Parameter(torch.from_numpy(np_q))
        self.weight_k =  nn.Parameter(torch.from_numpy(np_k))
        self.weight_v =  nn.Parameter(torch.from_numpy(np_v))
        self.weight_o =  nn.Parameter(torch.from_numpy(np_o))
        nn.init.xavier_uniform_(self.weight_q)
        nn.init.xavier_uniform_(self.weight_k)
        nn.init.xavier_uniform_(self.weight_v)
        nn.init.xavier_uniform_(self.weight_o)
        self.num_head = num_head
        self.size_per_head = size_per_head
        self.start_len = START_LEN
        self.seq_len = SEQ_LEN

    def forward(self, x, k, v): # (batch_size, num_head, 1, size_per_head)
        k = k + 0.0
        v = v + 0.0
        batch_size = x.size()[0]
        gen_id = self.start_len
        attn = torch.zeros(batch_size, self.num_head, 1, self.seq_len, device='cuda')
        for i in range(self.seq_len - self.start_len):
            q = torch.matmul(x, self.weight_q)
            # inplace在梯度计算的时候会有问题 需要修改,在反向传播的过程中
            # k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
            # v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
            k_new = k.clone()
            v_new = v.clone()
            k_new[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
            v_new[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
            k = k_new
            v = v_new
            
            attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
            attn = attn * 0.125
            attn = torch.softmax(attn, dim=3)
            x = torch.matmul(attn, v)
            x = torch.matmul(x, self.weight_o)
            gen_id = gen_id + 1
        
        return k, v, x


def build_model():
    model = Attention(NUM_HEAD, SIZE_PER_HEAD).cuda()
    if enable_torch:
        model = torch.jit.script(model)
    batch_size = arguments.bs
    # x = torch.randn(batch_size, NUM_HEAD, 1, SIZE_PER_HEAD).cuda()
    inp_np = np.random.randn(batch_size, NUM_HEAD, 1, SIZE_PER_HEAD)
    x = torch.from_numpy(inp_np).to(cuda_device).to(torch.float32)

    k = torch.zeros(batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    temp_k = np.random.randn(batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD)
    k[:, :, :START_LEN, :] = torch.from_numpy(temp_k).to(cuda_device).to(torch.float32)
    v = torch.zeros(batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    temp_v = np.random.randn(batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD)
    v[:, :, :START_LEN, :] =torch.from_numpy(temp_v).to(cuda_device).to(torch.float32)
    
    for parameter in model.parameters():  
        random_array = np.random.uniform(-0.1,0.1,parameter.data.shape)
        # if parameter.data.dim() > 1:
        #     xavier_uniform_(random_array)
        # if parameter.data.dim() > 1:
        #     nn.init.xavier_uniform_(parameter.data)
        tensor = torch.from_numpy(random_array).to(cuda_device).to(torch.float32)
        parameter.data = tensor

    
    return model, (x,k,v)

def test_all():
    model,(x,k,v) = build_model()
    test_with_grad(model,x.clone(),k.clone(),v.clone())
    test_with_backward(model,x.clone(),k.clone(),v.clone())
    test_res_accuracy(model,x.clone(),k.clone(),v.clone())
    test_grad_accuracy(model,x.clone(),k.clone(),v.clone())


def test_grad_accuracy(model,x,k,v):
    print("\033[92m----test-grad-accuracy----\033[0m")
    _,_,_x = model(x,k,v)
    loss = _x.sum()
    loss.backward(retain_graph=True)
    grad_res = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad = torch.autograd.grad(loss, param, create_graph=True)
            grad_res.append(grad[0].detach().cpu().numpy().reshape(-1,1))
            # print(f'Gradient check for {name}')
    print(np.asanyarray(grad_res).mean(axis=1))
    np.savetxt('res_grad.txt',np.asanyarray(grad_res).mean(axis=1))


def test_res_accuracy(model,x,k,v):
    with torch.no_grad():
        print("\033[92m----test-res-accuracy----\033[0m")
        #warm_up

        torch.cuda.synchronize()
        k,v,x = model(x,k,v)
        res = x.cpu().numpy().mean().reshape(-1, 1)
        print("res:",res)
        np.savetxt('res.txt',res)



def test_with_grad (model,x,k,v):
    torch.set_grad_enabled(True)
    print("\033[91m----test-with-grad----\033[0m")
    #warm_up
    torch.cuda.synchronize()
    for i in range(n_warmup):
        t0 = time()
        _ = model(x,k,v)
        torch.cuda.synchronize()
        # print("[warm up] Time {} ms".format((time() - t0) * 1000))

    print("[run]",end="")
    timer = Timer("ms")
    torch.cuda.synchronize()
    for i in range(n_run):
        timer.start()
        _ = model(x,k,v)
        torch.cuda.synchronize()
        timer.log()
    timer.report()


def test_with_backward(model,x,k,v):
    torch.set_grad_enabled(True)
    print("\033[91m----test-with-backward----\033[0m")

    torch.cuda.synchronize()
    #warm_up
    for i in range(n_warmup):
        t0 = time()
        _,_,_x = model(x,k,v)
        #loss 及反向传播
        loss = _x.sum()
        loss.backward()
        torch.cuda.synchronize()
        # print("[warm up] Time {} ms".format(((time() - t0) * 1000)))

    print("[run]",end="")
    timer = Timer("ms")
    torch.cuda.synchronize()
    for i in range(n_run):
        timer.start()
        _,_,_x = model(x,k,v)
        loss = _x.sum()
        loss.backward()
        torch.cuda.synchronize()
        timer.log()
    timer.report()

# 应该在保证权重没有被更新前进行第一次梯度计算的正确性检验
# def test_accuracy(model,x,k,v):
#     x_,k_,v_ = x,k,v
#     _,_,_x = model(x_,k_,v_)
#     loss = _x.sum()
#     loss.backward(retain_graph=True)
    
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             numerical_grad = torch.autograd.grad(loss, param, create_graph=True)
#             numerical_grad = numerical_grad[0].data

#             backward_grad = param.grad.data

#             # 检查数值梯度和反向传播计算得到的梯度是否接近
#             is_close = torch.allclose(numerical_grad, backward_grad, rtol=1e-3, atol=1e-4)
#             print(numerical_grad)
#             print(backward_grad)
#             # print(f'Gradient check for {name}: {"Passed" if is_close else "Failed"}')


if __name__ == '__main__':
    test_all()

