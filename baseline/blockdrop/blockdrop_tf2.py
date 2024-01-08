# import tensorflow as tf
# import numpy as np
# import os
# import time
# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# num_iter = 100
# warmup = 100

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--overhead_test', action='store_true')
# parser.add_argument('--unroll', dest='unroll', action='store_true')
# parser.add_argument('--fix', dest='unroll', action='store_false')
# parser.add_argument('--use_profile', type=bool,default=True)
# parser.set_defaults(unroll=False)
# parser.add_argument('--rate', type=int, default=-1)
# parser.add_argument('--bs', type=int, default=1)
# arguments = parser.parse_args()
# use_profile = arguments.use_profile

# import sys
# sys.path.append('../utils')
# from benchmark_timer import Timer
# from nsysprofile import profile_start, profile_stop, enable_profile
# enable_profile(use_profile)


# import ctypes

# def load_model(batch_size, unroll=False):
#     import onnx
#     from onnx_tf.backend import prepare
#     if unroll:
#         model_path = f"onnx/blockdrop.b{batch_size}.unroll.onnx"
#     else:
#         model_path = f"onnx/blockdrop.b{batch_size}.onnx"
#     model = onnx.load(model_path)
#     tf_model = prepare(model)
#     return tf_model.graph.as_graph_def()

# len_dataset = 10000

# def export_model(batch_size, unroll):
#     from tensorflow.python.framework import graph_util
#     model = load_model(batch_size, unroll)
#     output_names = tuple([model.tensor_dict[onnx_name].name for onnx_name in model.outputs])
#     print(output_names)
#     output_names = [o.split(":")[0] for o in output_names]
#     session_conf = tf.ConfigProto(
#         allow_soft_placement=True,
#         log_device_placement=False,
#         graph_options=tf.GraphOptions(infer_shapes=True),
#         inter_op_parallelism_threads=0
#     )
#     # not enable xla
#     session_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.L1
#     with tf.Graph().as_default(), tf.Session(config=session_conf) as session:
#         tf.import_graph_def(model.graph.as_graph_def(), name="")

#         constant_graph = graph_util.convert_variables_to_constants(
#                 session, session.graph_def, output_names)
#         with tf.gfile.GFile(f"blockdrop.b{batch_size}.tfgraph", "wb") as f:
#             f.write(constant_graph.SerializeToString())

# # def test_model(batch_size, enable_xla): 
# #     print("----batch_size={}---xla={}----".format(batch_size, enable_xla))

# #     session_conf = tf.ConfigProto(
# #         allow_soft_placement=True,
# #         log_device_placement=False,
# #         graph_options=tf.GraphOptions(infer_shapes=True),
# #         inter_op_parallelism_threads=0
# #     )

# #     if enable_xla:
# #         session_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
# #     else:
# #         session_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.L1

# #     # exit(0)
# #     model = load_model(batch_size,)
# #     with tf.Graph().as_default(), tf.Session(config=session_conf) as session:
# #         tf.import_graph_def(model, name="")
# #         # warm up
# #         for i in range(0, len_dataset, batch_size):
# #             if i >= warmup * batch_size: break
# #             inputs = inputs_all[i: i + batch_size]
# #             probs = probs_all[i: i + batch_size]
# #             outputs = session.run('add_18:0', {
# #                 'inputs.1:0': inputs,
# #                 'probs.1:0': probs
# #             })
# #         # run
# #         profile_start(use_profile)
# #         iter_times = []
# #         for i in range(0, len_dataset, batch_size):
# #             if i >= num_iter * batch_size: break
# #             inputs = inputs_all[i: i + batch_size]
# #             probs = probs_all[i: i + batch_size]
# #             start_time = time.time()
# #             outputs = session.run('add_18:0', {
# #                 'inputs.1:0': inputs,
# #                 'probs.1:0': probs
# #             })
# #             iter_time = (time.time() - start_time) * 1000
# #             iter_times.append(iter_time)
# #         profile_stop(use_profile)

# #         print("\033[31mSummary: [min, max, mean] = [%f, %f, %f] ms\033[m" % (
# #             min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))

# def test_fix_policy(batch_size, unroll, probs, rate): 
#     print("----batch_size={}---unroll={}----".format(batch_size, unroll))
#     import onnx
#     from onnx_tf.backend import prepare

#     rate_tag = "skip" if rate == -1 else f"{rate}"
#     if unroll:
#         model = onnx.load(f'onnx/blockdrop.b1.unroll.{rate_tag}.onnx')

#     else:
#         model = onnx.load(f'./onnx/testblockdrop.b1.fix.{rate_tag}.onnx')
#         print(f"onnx/blockdrop.b1.fix.{rate_tag}.onnx")
#         # tf.print(model)
#     session_conf = tf.compat.v1.ConfigProto(
#         allow_soft_placement=True,
#         log_device_placement=False,
#         graph_options=tf.compat.v1.GraphOptions(infer_shapes=True),
#         inter_op_parallelism_threads=0
#     )
#     # inputs = np.random.rand(batch_size, 3, 32, 32).astype(np.float32)
#     inputs =tf.Variable(tf.random.normal((batch_size,3, 32, 32)))  # Assuming channel-last format

#     session_conf.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.L1
    
#     model = prepare(model, gen_tensor_dict=True,training_mode=False)
#    # print("inputs:")
#     # for onnx_name in model.inputs:
#     #     print(onnx_name, model.tensor_dict[onnx_name])
#     # print("outputs:")
#     # for onnx_name in model.outputs:
#     #     print(onnx_name, model.tensor_dict[onnx_name])
#     # if unroll:
#     #     out_name = 'add_30:0'
#     # else:
#     #     out_name = 'add_18:0'
#     # print(model)
#     output_names = tuple([model.tensor_dict[onnx_name].name for onnx_name in model.outputs])
#     print("asdasdasdasd",model.graph)

#     with tf.Graph().as_default(), tf.compat.v1.Session(config=session_conf) as session:
#         tf.import_graph_def(model.graph.as_graph_def(), name="")
#         # warm up
#         for i in range(0, len_dataset, batch_size):
#             if i >= warmup * batch_size: break
#             outputs = session.run(output_names, {
#                 'inputs.1:0': inputs,
#                 'probs.1:0': probs
#             })
#         # run
#         profile_start(use_profile)
#         iter_times = []
#         for i in range(0, len_dataset, batch_size):
#             if i >= num_iter * batch_size: break
#             start_time = time.time()
#             outputs = session.run(output_names, {
#                 'inputs.1:0': inputs,
#                 'probs.1:0': probs
#             })
#             iter_time = (time.time() - start_time) * 1000
#             iter_times.append(iter_time)
#         profile_stop(use_profile)

#         print("\033[31mSummary: [min, max, mean] = [%f, %f, %f] ms\033[m" % (
#             min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))


# if arguments.rate == -1:
#     actions = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
# elif arguments.rate == 0:
#     actions = [0] * 15
# elif arguments.rate == 25:
#     actions = [
#         0, 0, 1, 0, 0,
#         0, 1, 0, 1, 0,
#         0, 0, 1, 0, 0,
#     ] 
# elif arguments.rate == 50:
#     actions = [
#         0, 1, 0, 1, 0,
#         1, 0, 1, 0, 1,
#         0, 1, 0, 1, 0,
#     ]
# elif arguments.rate == 75:
#     actions = [
#         1, 1, 0, 1, 1,
#         1, 0, 1, 0, 1,
#         1, 1, 0, 1, 1,
#     ]
# elif arguments.rate == 100:
#     actions = [1] * 15
# else:
#     raise NotImplementedError

# actions = np.array(actions, dtype=np.float32).reshape(-1, 15)
# test_fix_policy(1, arguments.unroll, actions, arguments.rate)
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers
from time import time
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


import os
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--rate', type=int, default=-1)
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


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3卷积，带有填充
    return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, padding= 'same', use_bias=False,input_shape=(32,32))

class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = conv3x3(planes, planes,stride)
        self.bn2 = layers.BatchNormalization()

    def call(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = tf.nn.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        return out


class DownsampleB(tf.keras.Model):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.avg = layers.AveragePooling2D(pool_size=stride)
        self.expand_ratio = nOut // nIn

    def call(self, x):
        x = self.avg(x)
        x = tf.concat((x, tf.zeros_like(x)), axis=1)
        return x

class FlatResNet32(tf.keras.Model):
    def __init__(self, block, layers, num_classes=10):
        super(FlatResNet32, self).__init__()

        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.avgpool = tf.keras.layers.AveragePooling2D(8)

        strides = [1, 2, 2]
        filt_sizes = [16, 32, 64]
        self.blocks, self.ds = [], []
        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(tf.keras.Sequential(blocks))
            if ds is not None:
                self.ds.append(ds)


        self.blocks = tf.keras.Sequential(self.blocks)
        self.ds = tf.keras.Sequential(self.ds)
        self.fc = tf.keras.layers.Dense(num_classes)
        self.fc_dim = 64 * block.expansion

        self.layer_config = layers

        # 这里应该也有一步初始化模-型参数的代码


    def seed(self, x):
        x = tf.nn.relu(self.bn1(self.conv1(x)))
        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleB(self.inplanes, planes * block.expansion, stride)
        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1))

        return layers, downsample

class FlatResNet32Policy(tf.keras.Model):
    def __init__(self, block, layers, num_classes=10):
        super(FlatResNet32Policy, self).__init__()

        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.avgpool = tf.keras.layers.AveragePooling2D(8)

        strides = [1, 2, 2]
        filt_sizes = [16, 32, 64]
        self.blocks, self.ds = [], []
        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(tf.keras.Sequential(blocks))
            if ds is not None:
                self.ds.append(ds)

        self.blocks = tf.keras.Sequential(self.blocks)
        self.ds = tf.keras.Sequential(self.ds)
        self.fc = tf.keras.layers.Dense(num_classes)
        self.fc_dim = 64 * block.expansion

        self.layer_config = layers
        # 这里应该也有一步初始化模型参数的代码



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleB(self.inplanes, planes * block.expansion, stride)

        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1))

        return layers, downsample

    def call(self, inputs):
        x = tf.nn.relu(self.bn1(self.conv1(inputs)))
        # layer 00
        residual0 = self.ds[0](x)
        b0 = self.blocks[0][0](x)  
        x0 = tf.nn.relu(residual0 + b0)
        # layer 10
        residual1 = self.ds[1](x0)
        b1 = self.blocks[1][0](x0)
        x1 = tf.nn.relu(residual1 + b1)
        # layer 20
        residual2 = self.ds[2](x1)
        b2 = self.blocks[2][0](x1)
        x2 = tf.nn.relu(residual2 + b2)
        # postprocessing
        x = self.avgpool(x2)
        x = tf.reshape(x, (x.shape[0], -1))
        return x


class Policy32(tf.keras.Model):
    def __init__(self, layer_config=[1,1,1], num_blocks=15):
        super(Policy32, self).__init__()
        self.features = FlatResNet32Policy(BasicBlock, layer_config, num_classes=10)
        self.feat_dim = self.features.fc_dim
        self.features.fc = tf.keras.Sequential()
        self.num_layers = sum(layer_config)

        self.logit = tf.keras.layers.Dense(num_blocks, activation=None)
        self.vnet = tf.keras.layers.Dense(1, activation=None)

    ## load dict  部分的


    def call(self, x):
        y = self.features(x)
        value = self.vnet(y)
        probs = tf.sigmoid(self.logit(y))
        return probs, value


class BlockDrop(tf.keras.Model):
    def __init__(self, rnet, agent):
        super(BlockDrop, self).__init__()
        self.rnet = rnet
        self.agent = agent

    def call_resnet(self, inputs):
        # FlatResNet
        x = self.rnet.seed(inputs)

        # layer 00
        residual = self.rnet.ds[0](x)
        fx = self.rnet.blocks[0][0](x)
        x = tf.nn.relu(residual + fx)

        # layer 01
        residual = x
        fx = self.rnet.blocks[0][1](x)
        x = tf.nn.relu(residual + fx)

        # layer 02
        residual = x
        fx = self.rnet.blocks[0][2](x)
        x = tf.nn.relu(residual + fx)

        # layer 03
        residual = x
        fx = self.rnet.blocks[0][3](x)
        x = tf.nn.relu(residual + fx)

        # layer 04
        residual = x
        fx = self.rnet.blocks[0][4](x)
        x = tf.nn.relu(residual + fx)

        # layer 10
        residual = self.rnet.ds[1](x)
        fx = self.rnet.blocks[1][0](x)
        x = tf.nn.relu(residual + fx)

        # layer 11
        residual = x
        fx = self.rnet.blocks[1][1](x)
        x = tf.nn.relu(residual + fx)

        # layer 12
        residual = x
        fx = self.rnet.blocks[1][2](x)
        x = tf.nn.relu(residual + fx)

        # layer 13
        residual = x
        fx = self.rnet.blocks[1][3](x)
        x = tf.nn.relu(residual + fx)

        # layer 14
        residual = x
        fx = self.rnet.blocks[1][4](x)
        x = tf.nn.relu(residual + fx)

        # layer 20
        residual = self.rnet.ds[2](x)
        fx = self.rnet.blocks[2][0](x)
        x = tf.nn.relu(residual + fx)

        # layer 21
        residual = x
        fx = self.rnet.blocks[2][1](x)
        x = tf.nn.relu(residual + fx)

        # layer 22
        residual = x
        fx = self.rnet.blocks[2][2](x)
        x = tf.nn.relu(residual + fx)

        # layer 23
        residual = x
        fx = self.rnet.blocks[2][3](x)
        x = tf.nn.relu(residual + fx)

        # layer 24
        residual = x
        fx = self.rnet.blocks[2][4](x)
        x = tf.nn.relu(residual + fx)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.rnet.fc(x)
        return x
    def call_real(self, inputs, probs):
        # probs, _ = self.agent(inputs)

        cond = tf.math.less(probs, tf.fill(tf.shape(probs), 0.5))
        policy = tf.where(cond, tf.zeros_like(probs), tf.ones_like(probs))
        policy = tf.transpose(policy)

        # FlatResNet
        x = self.rnet.seed(inputs)

        # layer 00
        action = policy[0]

        residual = self.rnet.ds.layers[0](x)
        if tf.reduce_sum(action) > 0.0:
            action_mask = tf.reshape(action, (-1, 1, 1, 1))
            fx = self.rnet.blocks.layers[0].layers[0](x)
            fx = tf.nn.relu(residual + fx)
            x = fx * action_mask + residual * (1.0 - action_mask)
        else:
            x = residual

# layer 01
        action = policy[1]
        residual = x
        if tf.reduce_sum(action) > 0.0:
            action_mask = tf.reshape(action, (-1, 1, 1, 1))
            fx = self.rnet.blocks[0][1](x)
            fx = tf.nn.relu(residual + fx)
            x = fx * action_mask + residual * (1.0 - action_mask)
        
# layer 02
        action = policy[2]
        residual = x
        if tf.reduce_sum(action) > 0.0:
            action_mask = tf.reshape(action, (-1, 1, 1, 1))
            fx = self.rnet.blocks[0][2](x)
            fx = tf.nn.relu(residual + fx)
            x = fx * action_mask + residual * (1.0 - action_mask)
        
# layer 03
        action = policy[3]
        residual = x
        if tf.reduce_sum(action) > 0.0:
            action_mask = tf.reshape(action, (-1, 1, 1, 1))
            fx = self.rnet.blocks[0][3](x)
            fx = tf.nn.relu(residual + fx)
            x = fx * action_mask + residual * (1.0 - action_mask)
        
# layer 04
        action = policy[4]
        residual = x
        if tf.reduce_sum(action) > 0.0:
            action_mask = tf.reshape(action, (-1, 1, 1, 1))
            fx = self.rnet.blocks[0][4](x)
            fx = tf.nn.relu(residual + fx)
            x = fx * action_mask + residual * (1.0 - action_mask)
        
# layer 10
        action = policy[5]
        residual = x
        if tf.reduce_sum(action) > 0.0:
            action_mask = tf.reshape(action, (-1, 1, 1, 1))
            fx = self.rnet.blocks[1][0](x)
            fx = tf.nn.relu(residual + fx)
            x = fx * action_mask + residual * (1.0 - action_mask)
        else:
            x = residual   
# layer 11
        action = policy[6]
        residual = x
        if tf.reduce_sum(action) > 0.0:
            action_mask = tf.reshape(action, (-1, 1, 1, 1))
            fx = self.rnet.blocks[1][1](x)
            fx = tf.nn.relu(residual + fx)
            x = fx * action_mask + residual * (1.0 - action_mask)
        
# layer 12
        action = policy[7]
        residual = x
        if tf.reduce_sum(action) > 0.0:
            action_mask = tf.reshape(action, (-1, 1, 1, 1))
            fx = self.rnet.blocks[1][2](x)
            fx = tf.nn.relu(residual + fx)
            x = fx * action_mask + residual * (1.0 - action_mask)
        
# layer13
        action = policy[8]
        residual = x
        if tf.reduce_sum(action) > 0.0:
            action_mask = tf.reshape(action, (-1, 1, 1, 1))
            fx = self.rnet.blocks[1][3](x)
            fx = tf.nn.relu(residual + fx)
            x = fx * action_mask + residual * (1.0 - action_mask)
        
# layer 09
        action = policy[9]
        residual = x
        if tf.reduce_sum(action) > 0.0:
            action_mask = tf.reshape(action, (-1, 1, 1, 1))
            fx = self.rnet.blocks[1][4](x)
            fx = tf.nn.relu(residual + fx)
            x = fx * action_mask + residual * (1.0 - action_mask)
        
# layer 20
        action = policy[10]
        residual = x
        if tf.reduce_sum(action) > 0.0:
            action_mask = tf.reshape(action, (-1, 1, 1, 1))
            fx = self.rnet.blocks[2][0](x)
            fx = tf.nn.relu(residual + fx)
            x = fx * action_mask + residual * (1.0 - action_mask)
        else:
            x = residual
# layer 21
        action = policy[11]
        residual = x
        if tf.reduce_sum(action) > 0.0:
            action_mask = tf.reshape(action, (-1, 1, 1, 1))
            fx = self.rnet.blocks[2][1](x)
            fx = tf.nn.relu(residual + fx)
            x = fx * action_mask + residual * (1.0 - action_mask)
        
# layer 22
        action = policy[12]
        residual = x
        if tf.reduce_sum(action) > 0.0:
            action_mask = tf.reshape(action, (-1, 1, 1, 1))
            fx = self.rnet.blocks[2][2](x)
            fx = tf.nn.relu(residual + fx)
            x = fx * action_mask + residual * (1.0 - action_mask)
        
# layer 23
        action = policy[13]
        residual = x
        if tf.reduce_sum(action) > 0.0:
            action_mask = tf.reshape(action, (-1, 1, 1, 1))
            fx = self.rnet.blocks[2][3](x)
            fx = tf.nn.relu(residual + fx)
            x = fx * action_mask + residual * (1.0 - action_mask)
        
# layer 24
        action = policy[14]
        residual = x
        if tf.reduce_sum(action) > 0.0:
            action_mask = tf.reshape(action, (-1, 1, 1, 1))
            fx = self.rnet.blocks[2][4](x)
            fx = tf.nn.relu(residual + fx)
            x = fx * action_mask + residual * (1.0 - action_mask)
        
        x = self.rnet.avgpool(x)
        x = tf.reshape(x, (x.shape[0], -1))
        x = self.rnet.fc(x)
        return x




layer_config = [5, 5, 5]
rnet = FlatResNet32(BasicBlock, layer_config, num_classes=10)
agent = Policy32([1,1,1], num_blocks=15)
len_dataset = 10000

model = BlockDrop(rnet, agent)



def run_fix_policy(batch_size, probs, unroll, rate):
    print("----batch_size={}---tensorflow={}----".format(batch_size, True))
    
    if unroll:
        if rate == -1:
            model.call = model.call_skip
            print("run model.call_skip")
        else:
            model.call = getattr(model, f"call_unroll_{rate}")
            print(f"run model.call_unroll_{rate}")
    else:
        model.call = model.call_real
        print("run model.call_real", probs)
    
    script_model = tf.function(model.call)
    inputs = tf.random.normal((batch_size,3, 32, 32)) 
    inputs = tf.transpose(inputs, perm=[0, 2, 3,1]) # Assuming channel-last format
    # Warmup
    print("[warmup]")
    tf.summary.trace_on(graph=True)

    for i in range(0, len_dataset, batch_size):
        if i >= n_warmup * batch_size:
            break
        t0 = time()
        _ = script_model(inputs, probs)
        t1 = time()
        print("time", t1 - t0)
    
    # Run
    timer = Timer("ms")
    print("[run]")
    
    profile_start(use_profile)
    for i in range(0, len_dataset, batch_size):
        if i >= n_run * batch_size:
            break
        timer.start()
        _ = script_model(inputs, probs)
        timer.log()
    profile_stop(use_profile)
    timer.report()


if __name__ == "__main__":
    if arguments.rate == -1:
        actions = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    elif arguments.rate == 0:
        actions = [0] * 15
    elif arguments.rate == 25:
        actions = [
            0, 0, 1, 0, 0,
            0, 1, 0, 1, 0,
            0, 0, 1, 0, 0,
        ]
    elif arguments.rate == 50:
        actions = [
            0, 1, 0, 1, 0,
            1, 0, 1, 0, 1,
            0, 1, 0, 1, 0,
        ]
    elif arguments.rate == 75:
        actions = [
            1, 1, 0, 1, 1,
            1, 0, 1, 0, 1,
            1, 1, 0, 1, 1,
        ]
    elif arguments.rate == 100:
        actions = [1] * 15
    actions = tf.reshape(tf.constant(actions, dtype=tf.float32),(-1, 15))
    print(arguments)
    run_fix_policy(1, actions, arguments.unroll, arguments.rate)
    # inputs = tf.random.normal((1, 3,32, 32))# Assuming channel-last format
    # inputs = tf.transpose(inputs, perm=[0, 2, 3,1])
    # modelconv  =conv3x3(3,16)
    # print(inputs)
    # print(modelconv(inputs))