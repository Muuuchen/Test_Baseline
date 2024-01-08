import tensorflow as tf
import numpy as np
import os
from time import time

class LSTMUnroll(tf.keras.Model):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
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

        cur_input = inputs[0]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        
        cur_input = inputs[1]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h

        cur_input = inputs[2]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
            
        cur_input = inputs[3]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[4]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[5]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[6]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[7]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[8]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[9]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[10]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[11]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[12]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[13]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[14]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[15]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[16]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[17]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[18]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[19]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[20]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[21]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[22]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[23]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[24]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[25]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[26]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[27]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[28]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[29]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[30]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[31]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[32]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[33]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[34]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[35]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[36]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[37]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[38]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[39]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[40]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[41]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[42]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[43]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[44]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[45]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[46]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[47]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[48]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[49]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[50]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[51]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[52]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[53]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[54]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[55]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[56]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[57]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[58]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[59]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[60]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[61]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[62]
        for j, layer in enumerate(self.lstmlayers):
            c = state_c.read(j)
            h = state_h.read(j)
            _, (c, h) = layer(cur_input,(c,h))
            # state_c[j] = c
            # state_h[j] = h
            state_c = state_c.write(j,c)
            state_h = state_h.write(j,h)
            cur_input = h
        cur_input = inputs[63]
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

