import torch 
from torchviz import make_dot


import sys
from attention_pytorch import Attention
# from lstm.lstm_pytorch import LSTMWrapper
# from nasrnn.nas_pytorch import NasRNN
# from rae.rae_pytorch import RAE
# from seq2seq.seq2seq_pytorch import AttnDecoderRNN
# from skipnet.skipnet_pytorch import RecurrentGatedRLResNet
# from blockdrop.blockdrop_pytorch import BlockDrop

if __name__ == "__main__":
    START_LEN = 32
    SEQ_LEN = 64
    NUM_HEAD = 12
    SIZE_PER_HEAD = 64
    batch_size = 1
    model = Attention(NUM_HEAD, SIZE_PER_HEAD).cuda()
    # Attention_inp
    x = torch.randn(batch_size, NUM_HEAD, 1, SIZE_PER_HEAD).cuda()
    k = torch.zeros(batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    k[:, :, :START_LEN, :] = torch.randn(batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    v = torch.zeros(batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    v[:, :, :START_LEN, :] = torch.randn(batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')

    outputs = model(x, k, v)

    g =make_dot(outputs)  
    g.render('attention_model', view=False)