"""
(C) Copyright [2024] Hewlett Packard Enterprise Development LP

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the Software),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import time
import sys
import os

from tool import *
from util import *

if len(sys.argv) == 14:
    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    output_path = sys.argv[2]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    iterations = int(sys.argv[3])
    conv2d_stride = int(sys.argv[4])
    conv2d_padding = int(sys.argv[5])
    conv2d_bias = float(sys.argv[6])
    pooling_size = int(sys.argv[7])
    pooling_stride = int(sys.argv[8])
    input_size = (int(sys.argv[9]), int(sys.argv[10]))
    conv_kernel_size = (int(sys.argv[11]), int(sys.argv[12]))

    conv_output_height = (input_size[0] - conv_kernel_size[0]  + 2 * conv2d_padding) // conv2d_stride + 1
    conv_output_width = (input_size[1] - conv_kernel_size[1] + 2 * conv2d_padding) // conv2d_stride + 1
    pool_output_height = (conv_output_height - pooling_size) // pooling_stride + 1
    pool_output_width = (conv_output_width - pooling_size) // pooling_stride + 1

    flattened_output_size = pool_output_height * pool_output_width

    full_connect_layer_size = (flattened_output_size, int(sys.argv[13]))
    transpose = False
else:
    print("Did not provide command line arguments correctly. Using default values from util.py")
    from util import *
    input_path = data_set_path
    if not os.path.exists(data_set_path):
        os.makedirs(data_set_path)
    output_path = data_set_path
    if not os.path.exists(data_set_path):
        os.makedirs(data_set_path)

def cnn_forward(network, input_image, iterations):
    print("=======================================")
    print("Running cnn benchmark Python")
    print("=======================================")
    
    def relu(x):
        return np.maximum(0, x)
    
    def softmax(input_0):
        # input_0 = input_0 - np.max(input_0)
        exp_input_0 = np.exp(input_0)
        sum_total_0 = np.sum(exp_input_0)
        output_0 = exp_input_0 / sum_total_0
        return output_0
    
    def pad_input(input, padding):
        if padding == 0:
            return input
        padded_input = np.zeros((input.shape[0] + 2*padding, input.shape[1] + 2*padding))
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                padded_input[i + padding][j + padding] = input[i][j]
        return padded_input
    
    def conv2d(input, kernel, bias, stride, padding):
        input_padded = pad_input(input, padding)
        kernel_height, kernel_width = kernel.shape
        output_height = (input_padded.shape[0] - kernel_height) // stride + 1
        output_width = (input_padded.shape[1] - kernel_width) // stride + 1
        conv2d_output = np.zeros((output_height, output_width))
        for i in range(0, output_height):
            for j in range(0, output_width):
                region = input_padded[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
                conv2d_output[i, j] = np.sum(region * kernel) + bias
        return conv2d_output

    def max_pooling(input, pooling_size, pooling_stride=2):
        output_height = (input.shape[0] - pooling_size) // pooling_stride + 1
        output_width = (input.shape[1] - pooling_size) // pooling_stride + 1
        output = np.zeros((output_height, output_width))
        for i in range(0, output_height):
            for j in range(0, output_width):
                region = input[i*pooling_stride:i*pooling_stride+pooling_size, j*pooling_stride:j*pooling_stride+pooling_size]
                output[i, j] = np.max(region)
        return output
    
    def dot_add(x, W, b):
        # mm = np.dot(x, W) + b
        # use forloops to create a fair comparison with the c++ code
        x_h = x.shape[0]
        x_w = x.shape[1]
        W_h = W.shape[0]
        W_w = W.shape[1]
        mm = np.zeros((x_h, W_w))
        for i in range(x_h):
            for j in range(W_w):
                for k in range(x_w):
                    mm[i, j] += x[i, k] * W[k, j]
                mm[i, j] += b[j]
        return mm
    
    # 1 warm up iteration
    print("Running 1 warm up iteration ...")
    W_conv, W_fc = network['weights'][0], network['weights'][1]
    b_fc = network['bias'][0] #, network['bias'][1]

    # Convolution layer
    conv_output = conv2d(input_image, W_conv, conv2d_bias, conv2d_stride, conv2d_padding)
    # print(f'conv_output = {conv_output}')
    save_bin(conv_output, f'{output_path}/conv_output.bin', np.float64, transpose)
    
    # ReLU activation
    relu_output = relu(conv_output)
    # print(f'relu_output = {relu_output}')
    save_bin(relu_output, f'{output_path}/relu_output.bin', np.float64, transpose)
    
    # Max Pooling layer
    pooled_output = max_pooling(relu_output, pooling_size, pooling_stride)
    # print(f'pooled_output = {pooled_output}')
    save_bin(pooled_output, f'{output_path}/pooled_output.bin', np.float64, transpose)

    # Flatten the pooled output
    # flattened_output = pooled_output.flatten()
    flattened_output = np.reshape(pooled_output, (1, flattened_output_size))
    # print(f'flattened_output = {flattened_output}')
    save_bin(flattened_output, f'{output_path}/flattened_output.bin', np.float64, transpose)

    # Fully connected layer
    fc_output = dot_add(flattened_output, W_fc, b_fc)
    fc_output = fc_output / 15000
    # print(f'fc_output = {fc_output}')
    save_bin(fc_output, f'{output_path}/fc_output.bin', np.float64, transpose)
    
    # Softmax output
    softmax_output = softmax(fc_output)
    softmax_output = softmax_output * 10000
    # print(f'softmax_output = {softmax_output}')
    save_bin(softmax_output, f'{output_path}/softmax_output.bin', np.float64, transpose)
    print("Done")

    # multi iterations
    print(f"Running {iterations} iterations ...")
    start_whole_time = time.time()

    conv_time = 0
    relu_time = 0
    pool_time = 0
    flat_time = 0
    fc_time = 0
    softmax_time = 0

    for i in range(iterations):
        start_iteration_time = time.time()
        conv_output = conv2d(input_image, W_conv, conv2d_bias, conv2d_stride, conv2d_padding)
        conv_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        relu_output = relu(conv_output)
        relu_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        pooled_output = max_pooling(relu_output, pooling_size, pooling_stride)
        pool_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        flattened_output = np.reshape(pooled_output, (1, flattened_output_size))
        flat_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        fc_output = dot_add(flattened_output, W_fc, b_fc)
        fc_time += time.time() - start_iteration_time
        fc_output = fc_output / 15000

        start_iteration_time = time.time()
        softmax_output = softmax(fc_output)
        softmax_time += time.time() - start_iteration_time
        softmax_output = softmax_output * 10000

    print("Done")

    run_whole_time = time.time() - start_whole_time
    print("1 warm up iteration and", iterations, "iterations")
    print("Single iteration time:", (run_whole_time / iterations) * 1000, "ms")
    print("conv2d kernel time:", (conv_time / iterations) * 1000, "ms")
    print("relu kernel time:", (relu_time / iterations) * 1000, "ms")
    print("max_pooling kernel time:", (pool_time / iterations) * 1000, "ms")
    # print("flat time:", (flat_time / iterations) * 1000, "ms")
    print("dot_add kernel time time:", (fc_time / iterations) * 1000, "ms")
    print("softmax kernel time:", (softmax_time / iterations) * 1000, "ms")

    del conv_output, relu_output, pooled_output, flattened_output, fc_output, softmax_output

if __name__ == "__main__":

    network = init_network(input_path, input_size, conv_kernel_size, full_connect_layer_size, transpose)
    input_image = np.random.random((input_size))
    # input_image = np.ones((input_size))

    save_bin(input_image, f'{input_path}/input_image.bin', np.float64, transpose)
    print(f'input_image = {input_image}')

    cnn_forward(network, input_image, iterations)
