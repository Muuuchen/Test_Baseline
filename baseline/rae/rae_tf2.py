import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate
import numpy as np
import os
from time import time
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import os
import numpy as np



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
depth = 7
n = 2 ** depth - 1



class RAE(tf.keras.Model):
    def __init__(self):
        super(RAE, self).__init__()
        self.encoder = Dense(512)
    
    def call(self, left, right, is_leaf, inp, root):
        if tf.cast(is_leaf[root], dtype=tf.bool):
            output = inp[root]  # (h,)
        else:
            a = self.call(left, right, is_leaf, inp, left[root].numpy())  # (h,)
            b = self.call(left, right, is_leaf, inp, right[root].numpy())  # (h,)
            ab = tf.concat((a, b),axis=0)  # (2h,)
            ab = tf.expand_dims(ab, axis=0)
            e = self.encoder(ab)
            e = tf.squeeze(e, axis=0)
            
            output = tf.nn.tanh(e)
        return output  
    

def test_model(enable_tf, enable_unroll, batch_size):
    if enable_unroll:
        model = RAEUnroll()
    else:
        model = RAE()

    root = 64
    left = tf.constant([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 30, 28, 27, 26, 33, 25, 24, 23, 22, 21, 20, 32, 18, 17, 15, 14, 13, 12, 11, 46, 8, 6, 5, 53, 10, 52, 2, 1, 0, 4, 58, 44])
    right = tf.constant([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 31, 29, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 19, 45, 16, 47, 48, 49, 50, 51, 9, 7, 54, 55, 56, 57, 3, 59, 60, 61, 62, 63])
    is_leaf = tf.constant([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    x = tf.ones([n, 512])
    # if enable_unroll:
    # args = (x,)

    args = (left, right, is_leaf, x, root)

    print("----batch_size={}---tf_function={}----".format(batch_size, enable_tf))
    print("[warmup]")
    for _ in range(n_warmup):
        t0 = time()
        _ = model(*args)
        print("Time {} ms".format((time() - t0) * 1000))

    timer = Timer("ms")
    profile_start(use_profile)

    for _ in range(n_run):
        timer.start()
        _ = model(*args)
        timer.log()
    profile_stop(use_profile)

    timer.report()

if __name__ == '__main__':
    test_model(True, arguments.unroll, arguments.bs)