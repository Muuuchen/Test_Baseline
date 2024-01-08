import torch
# import tenosorflow as tf
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import os

MAX_LENGTH = 50
OUTPUT_SIZE = 3797
HIDDEN_SIZE = 256
batch_size = 1

device = torch.device('cuda')


def gen_mask_from_sequence(std):
    bs = std.shape[0]
    padded_std = torch.zeros((bs, MAX_LENGTH), dtype=std.dtype, device=device)
    padded_std[:, :std.shape[1]] = std
    # print(std)
    # print(padded_std)
    mask = torch.zeros(bs, MAX_LENGTH, OUTPUT_SIZE, device=device)
    # print(mask)
    # print(torch.arange(bs).unsqueeze(1), torch.arange(
        # MAX_LENGTH).unsqueeze(0))
    mask[torch.arange(bs).unsqueeze(1), torch.arange(
        MAX_LENGTH).unsqueeze(0), padded_std] = 1000000.0
    # print(mask)
    mask = mask.transpose(0, 1).contiguous().clone()
    np.savetxt('mask.txt', mask.cpu().numpy().flatten())

    return mask


def test_torch():
    std = []
    for i in range(batch_size):
        l = 10
        lst = list(range(1, l))
        lst.append(0)
        assert (len(lst) <= MAX_LENGTH)
        # pad to MAX_LENGTH
        lst = lst + [0] * (MAX_LENGTH - len(lst))
        std.append(lst)
    std = torch.tensor(std, device=device)
    mask = gen_mask_from_sequence(std)
    return mask


if __name__ == "__main__":
    mask = test_torch().cpu().numpy()
    array  = np.loadtxt('mask.txt').reshape((50,1,3797))
    print(np.array_equal(mask, array))
    print(array)