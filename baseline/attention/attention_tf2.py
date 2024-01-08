"""
 *@description: tf
 *@author: haozhaoyang
 *@date: 2023-10-25
 *@idea: 
 """
import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from time import time
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--use_profile', type=bool,default=True)
parser.add_argument('--unroll', dest='unroll', action='store_true')
parser.add_argument('--fix', dest='unroll', action='store_false')
parser.add_argument('--platform', type=str)
parser.add_argument('--bs', type=int, default=1)
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

n_warmup = 100
n_run = 100

class Attention(tf.keras.layers.Layer):
    def __init__(self, num_head, size_per_head):
        super().__init__()
        self.num_head = num_head
        self.size_per_head = size_per_head
        self.start_len = START_LEN
        self.seq_len = SEQ_LEN
        
        self.weight_q = tf.Variable(initial_value=tf.random.normal((num_head, size_per_head, size_per_head)), dtype=tf.float32)
        self.weight_k = tf.Variable(initial_value=tf.random.normal((num_head, size_per_head, size_per_head)), dtype=tf.float32)
        self.weight_v = tf.Variable(initial_value=tf.random.normal((num_head, size_per_head, size_per_head)), dtype=tf.float32)
        self.weight_o = tf.Variable(initial_value=tf.random.normal((num_head, size_per_head, size_per_head)), dtype=tf.float32)
        
    def call(self, x, k, v):
        # k = k + 0.0 #从varibale 变成tensor
        # v = v + 0.0
        batch_size = tf.shape(x)[0]
        gen_id = self.start_len
        attn = tf.zeros((batch_size, self.num_head, 1, self.seq_len))
        
        for i in range(self.seq_len - self.start_len):
            q = tf.matmul(x, self.weight_q)
            k[:, :, gen_id, :].assign(tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head)))
            v[:, :, gen_id, :].assign(tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head)))
            attn = tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2]))
            attn = tf.transpose(attn, perm=[0,1,3,2])
            attn = attn * 0.125
            attn = tf.nn.softmax(attn, axis=3)
            x = tf.matmul(attn, v)
            x = tf.matmul(x, self.weight_o)
            gen_id = gen_id + 1
        return k, v, x

# class AttentionUnroll

class AttentionUnroll(tf.keras.layers.Layer):
    def __init__(self, num_head, size_per_head, start_len, seq_len):
        super(AttentionUnroll, self).__init__()
        self.num_head = num_head
        self.size_per_head = size_per_head
        self.start_len = start_len
        self.seq_len = seq_len

        self.weight_q = tf.Variable(initial_value=tf.random.normal((num_head, size_per_head, size_per_head)), dtype=tf.float32)
        self.weight_k = tf.Variable(initial_value=tf.random.normal((num_head, size_per_head, size_per_head)), dtype=tf.float32)
        self.weight_v = tf.Variable(initial_value=tf.random.normal((num_head, size_per_head, size_per_head)), dtype=tf.float32)
        self.weight_o = tf.Variable(initial_value=tf.random.normal((num_head, size_per_head, size_per_head)), dtype=tf.float32)

    def call(self, x, k, v): # (batch_size, num_head, 1, size_per_head)
        batch_size = tf.shape(x)[0]
        gen_id = self.start_len
        attn = tf.zeros((batch_size, self.num_head, 1, self.seq_len), dtype=tf.float32)
        
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = tf.matmul(x, self.weight_q)
        k = tf.reshape(tf.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v = tf.reshape(tf.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = tf.transpose(tf.matmul(k, tf.transpose(q, perm=[0, 1, 3, 2])), perm=[0, 1, 3, 2])
        attn = attn * 0.125
        attn = tf.nn.softmax(attn, axis=3)
        x = tf.matmul(attn, v)
        x = tf.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        
        return k,v,x

def test_model(enable_tf, batch_size, unroll):
    if not unroll:
        model = Attention(NUM_HEAD, SIZE_PER_HEAD)
    else:
        model = AttentionUnroll(NUM_HEAD, SIZE_PER_HEAD, START_LEN, SEQ_LEN)
        
    if enable_tf:
        model = tf.function(model)
    x = tf.Variable(initial_value=tf.random.normal((batch_size, NUM_HEAD, 1, SIZE_PER_HEAD)))
    k = tf.Variable(initial_value=tf.zeros((batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD)), dtype=tf.float32)
    k[:, :, :START_LEN, :].assign(tf.random.normal((batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD), dtype=tf.float32))
    v = tf.Variable(initial_value=tf.zeros((batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD)), dtype=tf.float32)
    v[:, :, :START_LEN, :].assign(tf.random.normal((batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD), dtype=tf.float32))
    
    print("----batch_size={}---tf_function={}----".format(batch_size, enable_tf))
    print("[warmup]")
    tf.summary.trace_on(graph=True)
    for i in range(n_warmup):
        t0 = time()
        _ = model(x, k, v,training=False)
        print("Time {} ms".format((time() - t0) * 1000))
    # tf.summary.trace_export(name="trace", step=0)
    
    timer = Timer("ms")
    profile_start(use_profile)
    print("[run]")
    for i in range(n_run):
        timer.start()
        _ = model(x, k, v,training=False)
        timer.log()
    profile_stop(use_profile)
    timer.report()

if __name__ == "__main__":
    if arguments.unroll:
        test_model(True, 1, True)
    else:
        test_model(True, 1, False)