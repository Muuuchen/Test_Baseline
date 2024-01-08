import tensorflow as tf
import numpy as np
import os
from time import time
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from nas_tf2_unroll import NasRNNUnroll

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

class NasRNN(tf.keras.Model):
    def __init__(self, input_size, hidden_size):
        super(NasRNN, self).__init__()
        self.weight_ih = tf.Variable(tf.random.normal(
            [8, input_size, hidden_size], dtype=tf.float32))
        self.weight_hh = tf.Variable(tf.random.normal(
            [8, hidden_size, hidden_size], dtype=tf.float32))
        self.hidden_size = hidden_size
        tf.keras.initializers.GlorotUniform(self.weight_ih)
        tf.keras.initializers.GlorotUniform(self.weight_hh)

    def call(self, inputs):  # seq_len, batch, input_size
        print(tf.executing_eagerly())
        batch_size = inputs.shape[1]
        state_c = tf.ones((batch_size, self.hidden_size), dtype=tf.float32)
        state_m = tf.ones((batch_size, self.hidden_size), dtype=tf.float32)
        for i in range(inputs.shape[0]):
            inp = inputs[i]

            ih = tf.matmul(inp, self.weight_ih)
            hh = tf.matmul(state_m, self.weight_hh)

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

            layer1_0 = tf.sigmoid(i0 + h0)
            layer1_1 = tf.nn.relu(i1 + h1)
            layer1_2 = tf.sigmoid(i2 + h2)
            layer1_3 = tf.nn.relu(i3 + h3)
            layer1_4 = tf.tanh(i4 + h4)
            layer1_5 = tf.sigmoid(i5 + h5)
            layer1_6 = tf.tanh(i6 + h6)
            layer1_7 = tf.sigmoid(i7 + h7)

            l2_0 = tf.tanh(layer1_0 * layer1_1)
            l2_1 = tf.tanh(layer1_2 + layer1_3)
            l2_2 = tf.tanh(layer1_4 * layer1_5)
            l2_3 = tf.sigmoid(layer1_6 + layer1_7)

            # Inject the cell
            l2_0_v2 = tf.tanh(l2_0 + state_c)

            # Third layer
            state_c = l2_0_v2 * l2_1
            l3_1 = tf.tanh(l2_2 + l2_3)

            # Final layer
            state_m = tf.tanh(state_c * l3_1)

        return state_m
    

def test_model(enable_tf, batch_size, unroll, *params):
    input_size, hidden_size, seq_len = params
    if not unroll:
        model = NasRNN(input_size, hidden_size)
    else:
        model = NasRNNUnroll(input_size, hidden_size)
    model.compile()
    if enable_tf:
        model = tf.function(model)

    embed = tf.random.normal([seq_len,batch_size, input_size])
    print("----batch_size={}---tf_function={}----".format(batch_size, enable_tf))
    print("[warmup]")
    tf.summary.trace_on(graph=True)
    for i in range(n_warmup):
        t0 = time()
        _ = model(embed)
        print("Time {} ms".format((time() - t0) * 1000))

    timer = Timer("ms")
    profile_start(use_profile)
    for i in range(n_run):
        timer.start()
        _ = model(embed)
        timer.log()
    profile_stop(use_profile)
    timer.report()


if __name__ == '__main__':
    input_size = 256
    hidden_size = 256
    seq_len = 1000

    if arguments.unroll:
        test_model(True, arguments.bs, True, input_size, hidden_size, seq_len)
    else:
        test_model(True, arguments.bs, False, input_size, hidden_size, seq_len)
