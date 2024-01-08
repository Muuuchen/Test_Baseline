
import sys
import argparse
import tensorflow as tf
import numpy as np
import os
from time import time
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--use_profile', type=bool, default=True)
parser.add_argument('--overhead_test', action='store_true')
parser.add_argument('--unroll', dest='unroll', action='store_true')
parser.add_argument('--fix', dest='unroll', action='store_false')
parser.set_defaults(unroll=False)
arguments = parser.parse_args()
use_profile = arguments.use_profile

sys.path.append('../utils')
from nsysprofile import profile_start, profile_stop, enable_profile
from benchmark_timer import Timer
enable_profile(use_profile)

n_warmup = 100
n_run = 100

MAX_LENGTH = 50
OUTPUT_SIZE = 3797
HIDDEN_SIZE = 256


class LSTMCell(tf.keras.layers.Layer):
    def __init__(self, hidden_size, input_size):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.weight_ih_l0_t = self.add_weight(shape=(
            4, input_size, hidden_size), initializer='random_normal', dtype=tf.float32)
        self.weight_hh_l0_t = self.add_weight(shape=(
            4, hidden_size, hidden_size), initializer='random_normal', dtype=tf.float32)
        self.bias_ih_0 = self.add_weight(
            shape=(hidden_size,), initializer='random_normal', dtype=tf.float32)
        self.bias_hh_0 = self.add_weight(
            shape=(hidden_size,), initializer='random_normal', dtype=tf.float32)
        self.bias_ih_1 = self.add_weight(
            shape=(hidden_size,), initializer='random_normal', dtype=tf.float32)
        self.bias_hh_1 = self.add_weight(
            shape=(hidden_size,), initializer='random_normal', dtype=tf.float32)
        self.bias_ih_2 = self.add_weight(
            shape=(hidden_size,), initializer='random_normal', dtype=tf.float32)
        self.bias_hh_2 = self.add_weight(
            shape=(hidden_size,), initializer='random_normal', dtype=tf.float32)
        self.bias_ih_3 = self.add_weight(
            shape=(hidden_size,), initializer='random_normal', dtype=tf.float32)
        self.bias_hh_3 = self.add_weight(
            shape=(hidden_size,), initializer='random_normal', dtype=tf.float32)

    def call(self, x, h, c):
        ih = tf.matmul(x, self.weight_ih_l0_t)
        hh = tf.matmul(h, self.weight_hh_l0_t)
        ih0 = ih[0] + self.bias_ih_0
        hh0 = hh[0] + self.bias_hh_0
        ih1 = ih[1] + self.bias_ih_1
        hh1 = hh[1] + self.bias_hh_1
        ih2 = ih[2] + self.bias_ih_2
        hh2 = hh[2] + self.bias_hh_2
        ih3 = ih[3] + self.bias_ih_3
        hh3 = hh[3] + self.bias_hh_3

        ingate = tf.sigmoid(ih0 + hh0)
        forgetgate = tf.sigmoid(ih1 + hh1)
        cellgate = tf.tanh(ih2 + hh2)
        outgate = tf.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * tf.tanh(c)
        return h, c


class AttnDecoderRNN(tf.keras.Model):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.gru = LSTMCell(self.hidden_size, self.hidden_size)
        self.out = tf.keras.layers.Dense(self.output_size)
        self.embedding = tf.keras.layers.Embedding(
            self.output_size, self.hidden_size)
        self.EOS_token = 0
        self.SOS_token = 1

   

    def call(self, encoder_output, std, h, c):
        batch_size = encoder_output.shape[1]
        # Hack for bug in ScatterND on Constant
        # output_all = tf.zeros(
        #     (self.max_length, batch_size), dtype=tf.int64) + 0
        output_all =tf.TensorArray(tf.float32, size=self.max_length)

        output = tf.fill((batch_size,), self.SOS_token)
        output= tf.cast(output, dtype=tf.int64)
        cond = tf.constant(True)
        id = 0
        while cond:
            x = self.embedding(output)
            ih = tf.matmul(x, self.gru.weight_ih_l0_t)
            hh = tf.matmul(h, self.gru.weight_hh_l0_t)
            ih0 = ih[0] + self.gru.bias_ih_0
            hh0 = hh[0] + self.gru.bias_hh_0
            ih1 = ih[1] + self.gru.bias_ih_1
            hh1 = hh[1] + self.gru.bias_hh_1
            ih2 = ih[2] + self.gru.bias_ih_2
            hh2 = hh[2] + self.gru.bias_hh_2
            ih3 = ih[3] + self.gru.bias_ih_3
            hh3 = hh[3] + self.gru.bias_hh_3

            ingate = tf.sigmoid(ih0 + hh0)
            forgetgate = tf.sigmoid(ih1 + hh1)
            cellgate = tf.tanh(ih2 + hh2)
            outgate = tf.sigmoid(ih3 + hh3)

            c = (forgetgate * c) + (ingate * cellgate)
            h = outgate * tf.tanh(c)

            output = self.out(h) + std[id]
            output = tf.argmax(output, axis=1)
            output_all.write(id, output)
            id = id + 1
            cond = tf.math.logical_and(tf.reduce_max(
                output) > self.EOS_token, id < self.max_length)
        return output_all.stack(), h


class AttnDecoderRNNUnroll(tf.keras.Model):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNNUnroll, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.gru = LSTMCell(self.hidden_size, self.hidden_size)
        self.out = tf.keras.layers.Dense(self.output_size)
        self.embedding = tf.keras.layers.Embedding(
            self.output_size, self.hidden_size)
        self.EOS_token = 0
        self.SOS_token = 1

    def call(self, encoder_output, std, h, c,output):
        batch_size = encoder_output.shape[1]
        # Hack for bug in ScatterND on Constant
        # output_all = tf.zeros(
        #     (self.max_length, batch_size), dtype=tf.int64) + 0
        output_all =tf.TensorArray(tf.float32, size=self.max_length)

        cond = tf.constant(True)
        id = 0
        x = self.embedding(output)
        ih = tf.matmul(x, self.gru.weight_ih_l0_t)
        hh = tf.matmul(h, self.gru.weight_hh_l0_t)
        ih0 = ih[0] + self.gru.bias_ih_0
        hh0 = hh[0] + self.gru.bias_hh_0
        ih1 = ih[1] + self.gru.bias_ih_1
        hh1 = hh[1] + self.gru.bias_hh_1
        ih2 = ih[2] + self.gru.bias_ih_2
        hh2 = hh[2] + self.gru.bias_hh_2
        ih3 = ih[3] + self.gru.bias_ih_3
        hh3 = hh[3] + self.gru.bias_hh_3

        ingate = tf.sigmoid(ih0 + hh0)
        forgetgate = tf.sigmoid(ih1 + hh1)
        cellgate = tf.tanh(ih2 + hh2)
        outgate = tf.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * tf.tanh(c)

        output = self.out(h) + std[id]
        output = tf.argmax(output, axis=1)
        output_all.write(id, output)
        id = id + 1 # id =1

        x = self.embedding(output)
        ih = tf.matmul(x, self.gru.weight_ih_l0_t)
        hh = tf.matmul(h, self.gru.weight_hh_l0_t)
        ih0 = ih[0] + self.gru.bias_ih_0
        hh0 = hh[0] + self.gru.bias_hh_0
        ih1 = ih[1] + self.gru.bias_ih_1
        hh1 = hh[1] + self.gru.bias_hh_1
        ih2 = ih[2] + self.gru.bias_ih_2
        hh2 = hh[2] + self.gru.bias_hh_2
        ih3 = ih[3] + self.gru.bias_ih_3
        hh3 = hh[3] + self.gru.bias_hh_3

        ingate = tf.sigmoid(ih0 + hh0)
        forgetgate = tf.sigmoid(ih1 + hh1)
        cellgate = tf.tanh(ih2 + hh2)
        outgate = tf.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * tf.tanh(c)

        output = self.out(h) + std[id]
        output = tf.argmax(output, axis=1)
        output_all.write(id, output)
        id = id + 1 # id =2

        x = self.embedding(output)
        ih = tf.matmul(x, self.gru.weight_ih_l0_t)
        hh = tf.matmul(h, self.gru.weight_hh_l0_t)
        ih0 = ih[0] + self.gru.bias_ih_0
        hh0 = hh[0] + self.gru.bias_hh_0
        ih1 = ih[1] + self.gru.bias_ih_1
        hh1 = hh[1] + self.gru.bias_hh_1
        ih2 = ih[2] + self.gru.bias_ih_2
        hh2 = hh[2] + self.gru.bias_hh_2
        ih3 = ih[3] + self.gru.bias_ih_3
        hh3 = hh[3] + self.gru.bias_hh_3

        ingate = tf.sigmoid(ih0 + hh0)
        forgetgate = tf.sigmoid(ih1 + hh1)
        cellgate = tf.tanh(ih2 + hh2)
        outgate = tf.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * tf.tanh(c)

        output = self.out(h) + std[id]
        output = tf.argmax(output, axis=1)
        output_all.write(id, output)
        id = id + 1 # id =3

        x = self.embedding(output)
        ih = tf.matmul(x, self.gru.weight_ih_l0_t)
        hh = tf.matmul(h, self.gru.weight_hh_l0_t)
        ih0 = ih[0] + self.gru.bias_ih_0
        hh0 = hh[0] + self.gru.bias_hh_0
        ih1 = ih[1] + self.gru.bias_ih_1
        hh1 = hh[1] + self.gru.bias_hh_1
        ih2 = ih[2] + self.gru.bias_ih_2
        hh2 = hh[2] + self.gru.bias_hh_2
        ih3 = ih[3] + self.gru.bias_ih_3
        hh3 = hh[3] + self.gru.bias_hh_3

        ingate = tf.sigmoid(ih0 + hh0)
        forgetgate = tf.sigmoid(ih1 + hh1)
        cellgate = tf.tanh(ih2 + hh2)
        outgate = tf.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * tf.tanh(c)

        output = self.out(h) + std[id]
        output = tf.argmax(output, axis=1)
        output_all.write(id, output)
        id = id + 1 # id =4

        x = self.embedding(output)
        ih = tf.matmul(x, self.gru.weight_ih_l0_t)
        hh = tf.matmul(h, self.gru.weight_hh_l0_t)
        ih0 = ih[0] + self.gru.bias_ih_0
        hh0 = hh[0] + self.gru.bias_hh_0
        ih1 = ih[1] + self.gru.bias_ih_1
        hh1 = hh[1] + self.gru.bias_hh_1
        ih2 = ih[2] + self.gru.bias_ih_2
        hh2 = hh[2] + self.gru.bias_hh_2
        ih3 = ih[3] + self.gru.bias_ih_3
        hh3 = hh[3] + self.gru.bias_hh_3

        ingate = tf.sigmoid(ih0 + hh0)
        forgetgate = tf.sigmoid(ih1 + hh1)
        cellgate = tf.tanh(ih2 + hh2)
        outgate = tf.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * tf.tanh(c)

        output = self.out(h) + std[id]
        output = tf.argmax(output, axis=1)
        output_all.write(id, output)
        id = id + 1 # id =5

        x = self.embedding(output)
        ih = tf.matmul(x, self.gru.weight_ih_l0_t)
        hh = tf.matmul(h, self.gru.weight_hh_l0_t)
        ih0 = ih[0] + self.gru.bias_ih_0
        hh0 = hh[0] + self.gru.bias_hh_0
        ih1 = ih[1] + self.gru.bias_ih_1
        hh1 = hh[1] + self.gru.bias_hh_1
        ih2 = ih[2] + self.gru.bias_ih_2
        hh2 = hh[2] + self.gru.bias_hh_2
        ih3 = ih[3] + self.gru.bias_ih_3
        hh3 = hh[3] + self.gru.bias_hh_3

        ingate = tf.sigmoid(ih0 + hh0)
        forgetgate = tf.sigmoid(ih1 + hh1)
        cellgate = tf.tanh(ih2 + hh2)
        outgate = tf.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * tf.tanh(c)

        output = self.out(h) + std[id]
        output = tf.argmax(output, axis=1)
        output_all.write(id, output)
        id = id + 1 # id =6

        x = self.embedding(output)
        ih = tf.matmul(x, self.gru.weight_ih_l0_t)
        hh = tf.matmul(h, self.gru.weight_hh_l0_t)
        ih0 = ih[0] + self.gru.bias_ih_0
        hh0 = hh[0] + self.gru.bias_hh_0
        ih1 = ih[1] + self.gru.bias_ih_1
        hh1 = hh[1] + self.gru.bias_hh_1
        ih2 = ih[2] + self.gru.bias_ih_2
        hh2 = hh[2] + self.gru.bias_hh_2
        ih3 = ih[3] + self.gru.bias_ih_3
        hh3 = hh[3] + self.gru.bias_hh_3

        ingate = tf.sigmoid(ih0 + hh0)
        forgetgate = tf.sigmoid(ih1 + hh1)
        cellgate = tf.tanh(ih2 + hh2)
        outgate = tf.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * tf.tanh(c)

        output = self.out(h) + std[id]
        output = tf.argmax(output, axis=1)
        output_all.write(id, output)
        id = id + 1 # id =7

        x = self.embedding(output)
        ih = tf.matmul(x, self.gru.weight_ih_l0_t)
        hh = tf.matmul(h, self.gru.weight_hh_l0_t)
        ih0 = ih[0] + self.gru.bias_ih_0
        hh0 = hh[0] + self.gru.bias_hh_0
        ih1 = ih[1] + self.gru.bias_ih_1
        hh1 = hh[1] + self.gru.bias_hh_1
        ih2 = ih[2] + self.gru.bias_ih_2
        hh2 = hh[2] + self.gru.bias_hh_2
        ih3 = ih[3] + self.gru.bias_ih_3
        hh3 = hh[3] + self.gru.bias_hh_3

        ingate = tf.sigmoid(ih0 + hh0)
        forgetgate = tf.sigmoid(ih1 + hh1)
        cellgate = tf.tanh(ih2 + hh2)
        outgate = tf.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * tf.tanh(c)

        output = self.out(h) + std[id]
        output = tf.argmax(output, axis=1)
        output_all.write(id, output)
        id = id + 1 # id =8

        x = self.embedding(output)
        ih = tf.matmul(x, self.gru.weight_ih_l0_t)
        hh = tf.matmul(h, self.gru.weight_hh_l0_t)
        ih0 = ih[0] + self.gru.bias_ih_0
        hh0 = hh[0] + self.gru.bias_hh_0
        ih1 = ih[1] + self.gru.bias_ih_1
        hh1 = hh[1] + self.gru.bias_hh_1
        ih2 = ih[2] + self.gru.bias_ih_2
        hh2 = hh[2] + self.gru.bias_hh_2
        ih3 = ih[3] + self.gru.bias_ih_3
        hh3 = hh[3] + self.gru.bias_hh_3

        ingate = tf.sigmoid(ih0 + hh0)
        forgetgate = tf.sigmoid(ih1 + hh1)
        cellgate = tf.tanh(ih2 + hh2)
        outgate = tf.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * tf.tanh(c)

        output = self.out(h) + std[id]
        output = tf.argmax(output, axis=1)
        output_all.write(id, output)
        id = id + 1 # id =9

        x = self.embedding(output)
        ih = tf.matmul(x, self.gru.weight_ih_l0_t)
        hh = tf.matmul(h, self.gru.weight_hh_l0_t)
        ih0 = ih[0] + self.gru.bias_ih_0
        hh0 = hh[0] + self.gru.bias_hh_0
        ih1 = ih[1] + self.gru.bias_ih_1
        hh1 = hh[1] + self.gru.bias_hh_1
        ih2 = ih[2] + self.gru.bias_ih_2
        hh2 = hh[2] + self.gru.bias_hh_2
        ih3 = ih[3] + self.gru.bias_ih_3
        hh3 = hh[3] + self.gru.bias_hh_3

        ingate = tf.sigmoid(ih0 + hh0)
        forgetgate = tf.sigmoid(ih1 + hh1)
        cellgate = tf.tanh(ih2 + hh2)
        outgate = tf.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * tf.tanh(c)

        output = self.out(h) + std[id]
        output = tf.argmax(output, axis=1)
        output_all.write(id, output)
        id = id + 1 # id =10
        return output_all.stack(), h


    
        


def gen_mask_from_sequence(std):
    bs = std.shape[0]
    padded_std = tf.zeros((bs, MAX_LENGTH), dtype=std.dtype)
    padded_std = std
    mask = tf.zeros((bs, MAX_LENGTH, OUTPUT_SIZE))
    mask = tf.constant(np.loadtxt('mask.txt').reshape((MAX_LENGTH,bs,OUTPUT_SIZE)),dtype=tf.float32)
    return mask


def run_fix_policy(batch_size, unroll):
    print("----batch_size={}---unroll={}----".format(batch_size, unroll))
    if unroll:
        model = AttnDecoderRNNUnroll(HIDDEN_SIZE, OUTPUT_SIZE, dropout_p=0.1)
    else:
        model = AttnDecoderRNN(HIDDEN_SIZE, OUTPUT_SIZE, dropout_p=0.1)
    std = []
    MAX_LENGTH = 50
    h = tf.random.normal((batch_size, HIDDEN_SIZE))
    c = tf.random.normal((batch_size, HIDDEN_SIZE))
    sos = tf.fill((batch_size,), model.SOS_token)
    for i in range(batch_size):
        l = 10
        lst = list(range(1, l))
        lst.append(0)
        assert (len(lst) <= MAX_LENGTH)
        # pad to MAX_LENGTH
        lst = lst + [0] * (MAX_LENGTH - len(lst))
        std.append(lst)
    std = tf.constant(std)
    mask = gen_mask_from_sequence(std)

    encoder_output = tf.random.normal((MAX_LENGTH, batch_size, HIDDEN_SIZE))
    script_model = tf.function(model)
    if unroll:
        args = (encoder_output, mask, h, c,sos)
    else:
        args = (encoder_output, mask, h, c)
    # warmup
    for i in range(0, n_warmup * batch_size, batch_size):
        _ = script_model(*args)
    # run
    timer = Timer("ms")
    profile_start(use_profile)
    for i in range(0, n_run * batch_size, batch_size):
        timer.start()
        _ = script_model(*args)
        timer.log()
    profile_stop(use_profile)
    timer.report()


if __name__ == '__main__':
    run_fix_policy(1, arguments.unroll)
    # std = []
    # MAX_LENGTH = 50
    # for i in range(1):
    #     l = 10
    #     lst = list(range(1, l))
    #     lst.append(0)
    #     assert (len(lst) <= MAX_LENGTH)
    #     # pad to MAX_LENGTH
    #     lst = lst + [0] * (MAX_LENGTH - len(lst))
    #     std.append(lst)
    # std = tf.constant(std)
    # print(std)
    # mask = gen_mask_from_sequence(std)
    # print(mask)
