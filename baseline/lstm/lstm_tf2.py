import tensorflow as tf
import numpy as np
import os
from time import time
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from lstm_tf2_unroll import LSTMUnroll

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


class LSTM(tf.keras.Model):
    def __init__(self, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.lstmlayers = []
        self.lstmlayers.append(tf.keras.layers.LSTMCell(hidden_size))
        for _ in range(num_layers - 1):
            self.lstmlayers.append(tf.keras.layers.LSTMCell(hidden_size))
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def call(self, inputs):  # seq_len, batch, input_size
        batch_size = tf.shape(inputs)[1]
        state_c =tf.TensorArray(tf.float32, size=10)
        state_h =tf.TensorArray(tf.float32, size=10)
        for i in range(10):
            state_c.write(i,tf.zeros((batch_size, self.hidden_size))) # hardcode for ts compile
            state_h.write(i,tf.zeros((batch_size, self.hidden_size))) # hardcode for ts compile

        for i in range(tf.shape(inputs)[0]):
            cur_input = inputs[i]
            for j, layer in enumerate(self.lstmlayers):
                c = state_c.read(j)
                h = state_h.read(j)
                _, (c, h) = layer(cur_input,(c,h))
                # state_c[j] = c
                # state_h[j] = h
                state_c = state_c.write(j,c)
                state_h = state_h.write(j,h)
                cur_input = h
        return state_h.read(self.num_layers - 1)

# class LSTMUnroll



def test_model(enable_tf, batch_size, impl, *params):
    input_size, hidden_size, num_layers, seq_len = params
    
    # if impl == 'cudnn':
    #     model = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True)
    if impl == 'loop':
        model = LSTM(hidden_size, num_layers)
    elif impl == 'unroll':
        model = LSTMUnroll(hidden_size, num_layers)
    else:
        raise NotImplementedError
    
    # model.build((None, seq_len, input_size))
    # model.summary()
        # Convert to TensorFlow GraphDef
    if enable_tf:
        model = tf.function(model)
    
    inp = tf.random.normal([seq_len,batch_size, input_size])
    print("----batch_size={}---tf_function={}----".format(batch_size, enable_tf))
    print("[warmup]")
    tf.summary.trace_on(graph=True)

    for i in range(n_warmup):
        t0 = time()
        _ = model(inp)
        print("Time {} ms".format((time() - t0) * 1000))
    
    timer = Timer("ms")
    profile_start(use_profile)
    for i in range(n_run):
        timer.start()
        _ = model(inp)
        timer.log()
    profile_stop(use_profile)
    timer.report()



if __name__ == '__main__':
    input_size = 256
    hidden_size = 256
    num_layers = 10
    seq_len = 64

    if arguments.unroll:
        test_model(True, 1, 'unroll', input_size, hidden_size, num_layers, seq_len)
    else:
        test_model(True, 1, 'loop', input_size, hidden_size, num_layers, seq_len)

