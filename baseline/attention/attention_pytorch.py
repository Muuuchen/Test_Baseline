"""
 *@description: Attention forward backward accuracy
 *@author: haozhaoyang
 *@date: 2023-10-24
 *@idea: accuracy - same seed
 """
import torch
import torch.nn as nn
from time import time
from attention_pytorch_unroll import AttentionUnroll
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

START_LEN = 32
SEQ_LEN = 64
NUM_HEAD = 12
SIZE_PER_HEAD = 64

cuda_device = torch.device("cuda:0")
# print(cuda_device)
n_warmup = 100
n_run = 100


class Attention(nn.Module):
    def __init__(self, num_head, size_per_head):
        super().__init__()
        self.weight_q = nn.Parameter(torch.randn(num_head, size_per_head, size_per_head, dtype=torch.float32))
        self.weight_k = nn.Parameter(torch.randn(num_head, size_per_head, size_per_head, dtype=torch.float32))
        self.weight_v = nn.Parameter(torch.randn(num_head, size_per_head, size_per_head, dtype=torch.float32))
        self.weight_o = nn.Parameter(torch.randn(num_head, size_per_head, size_per_head, dtype=torch.float32))
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
            # inplace 需要修改在反向传播的过程中
            k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
            v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
            # k_new = k.clone()
            # v_new = v.clone()
            # k_new[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
            # v_new[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
            # k = k_new
            # v = v_new
            attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
            attn = attn * 0.125
            attn = torch.softmax(attn, dim=3)
            x = torch.matmul(attn, v)
            x = torch.matmul(x, self.weight_o)
            gen_id = gen_id + 1
        
        return k, v, x

def test_model(enable_torch, batch_size, unroll):
    if not unroll:
        model = Attention(NUM_HEAD, SIZE_PER_HEAD).cuda()
    else:
        model = AttentionUnroll(NUM_HEAD, SIZE_PER_HEAD, START_LEN, SEQ_LEN).cuda()
    model.eval()
    if enable_torch:
        model = torch.jit.script(model)
    x = torch.randn(batch_size, NUM_HEAD, 1, SIZE_PER_HEAD).cuda()
    k = torch.zeros(batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    k[:, :, :START_LEN, :] = torch.randn(batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    v = torch.zeros(batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    v[:, :, :START_LEN, :] = torch.randn(batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    print("----batch_size={}---torchscript={}----".format(batch_size, enable_torch))
    print("[warmup]")
    with torch.no_grad():
        torch.cuda.synchronize()
        for i in range(n_warmup):
            t0 = time()
            _ = model(x, k, v)
            torch.cuda.synchronize()
            print("Time {} ms".format((time() - t0) * 1000))

        timer = Timer("ms")
        torch.cuda.synchronize()
        print("[run]")
        for i in range(n_run):
            timer.start()
            _ = model(x, k, v)
            torch.cuda.synchronize()
            timer.log()
        timer.report()

    # os.system("mkdir -p onnx")
    # if unroll:
    #     torch.onnx.export(model, (x, k, v), f'onnx/attention.b{batch_size}.unroll.onnx', verbose=True, opset_version=11)
    # else:
    #     torch.onnx.export(model, (x, k, v), f'onnx/attention.b{batch_size}.onnx', verbose=True, opset_version=11)


if __name__ == '__main__':
    if not arguments.overhead_test:
        test_model(True, arguments.bs, False)
    else:
        if arguments.unroll:
            test_model(True, 1, True)
        else:
            test_model(True, 1, False)
        # test_model(False, 1, False)
        # test_model(True, 1, False)
        # test_model(False, 64, False)
        # test_model(True, 64, False)

        # test_model(True, 1, True)
