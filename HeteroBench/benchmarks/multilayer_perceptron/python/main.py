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

if len(sys.argv) == 16:
    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    output_path = sys.argv[2]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    iterations = int(sys.argv[3])
    layer_num = 4
    layer_info = [  (int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])), \
                    (int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9])), \
                    (int(sys.argv[10]), int(sys.argv[11]), int(sys.argv[12])), \
                    (int(sys.argv[13]), int(sys.argv[14]), int(sys.argv[15]))]
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

def mlp_forward(network, a0):
    print("=======================================")
    print("Running mlp benchmark Python")
    print("=======================================")
    
    def sigmoid(a):
        z = 1 / (1 + np.exp(-a))
        return z
    
    def softmax(input_0):
        # input_0 = input_0 - np.max(input_0)
        exp_input_0 = np.exp(input_0)
        sum_total_0 = np.sum(exp_input_0)
        output_0 = exp_input_0 / sum_total_0
        return output_0
    
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
                mm[i, j] += b[i, j]
        return mm
    
    # 1 warm up iteration
    print("Running 1 warm up iteration ...")
    W0,W1,W2, W3 = network['weights'][0], network['weights'][1], network['weights'][2], network['weights'][3]
    b0, b1, b2, b3 = network['bias'][0], network['bias'][1], network['bias'][2], network['bias'][3]

    # layer 0
    a1 = dot_add(a0, W0, b0)
    a1 = a1 / 500
    # print(f'a1 = {a1}')
    save_bin(a1, f'{output_path}/a1.bin', np.float64, transpose)
    z1 = sigmoid(a1)
    # print(f'z1 = {z1}')
    save_bin(z1, f'{output_path}/z1.bin', np.float64, transpose)

    # layer 1
    a2 = dot_add(z1, W1, b1)
    a2 = a2 / 1500
    # print(f'a2 = {a2}')
    save_bin(a2, f'{output_path}/a2.bin', np.float64, transpose)
    z2 = sigmoid(a2)
    # print(f'z2 = {z2}')
    save_bin(z2, f'{output_path}/z2.bin', np.float64, transpose)

    # layer 2
    a3 = dot_add(z2, W2, b2)
    a3 = a3 / 1500
    # print(f'a3 = {a3}')
    save_bin(a3, f'{output_path}/a3.bin', np.float64, transpose)
    z3 = sigmoid(a3)
    # print(f'z3 = {z3}')
    save_bin(z3, f'{output_path}/z3.bin', np.float64, transpose)

    # layer 3
    a4 = dot_add(z3, W3, b3)
    a4 = a4 / 1500
    # print(f'a4 = {a4}')
    save_bin(a4, f'{output_path}/a4.bin', np.float64, transpose)
    z4 = softmax(a4)
    z4 = z4 * 1000000
    # print(f'z4 = {z4}')
    save_bin(z4, f'{output_path}/z4.bin', np.float64, transpose)
    print("Done")

    # multi iterations
    print(f"Running {iterations} iterations ...")
    start_whole_time = time.time()

    layer_0_time = 0
    layer_1_time = 0
    layer_2_time = 0
    layer_3_time = 0

    for i in range(iterations):
        start_iteration_time = time.time()
        a1 = dot_add(a0, W0, b0)
        a1 = a1 / 500
        z1 = sigmoid(a1)
        layer_0_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        a2 = dot_add(z1, W1, b1)
        a2 = a2 / 1500
        z2 = sigmoid(a2)
        layer_1_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        a3 = dot_add(z2, W2, b2)
        a3 = a3 / 1500
        z3 = sigmoid(a3)
        layer_2_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        a4 = dot_add(z3, W3, b3)
        a4 = a4 / 1500
        z4 = softmax(a4)
        z4 = z4 * 1000000
        layer_3_time += time.time() - start_iteration_time

    print("Done")

    run_whole_time = time.time() - start_whole_time
    print("1 warm up iteration and", iterations, "iterations")
    print("Single iteration time:", (run_whole_time / iterations) * 1000, "ms")
    print("Layer 0 time:", (layer_0_time / iterations) * 1000, "ms")
    print("Layer 1 time:", (layer_1_time / iterations) * 1000, "ms")
    print("Layer 2 time:", (layer_2_time / iterations) * 1000, "ms")
    print("Layer 3 time:", (layer_3_time / iterations) * 1000, "ms")

    del a1, a2, a3, a4, z1, z2, z3, z4

if __name__ == "__main__":
    network = init_network(input_path, layer_num, layer_info, transpose)
    a0 = np.random.random((layer_info[0][0], layer_info[0][1]))#, dtype=np.float64)
    # a0 = np.ones((layer_info[0][0], layer_info[0][1]))
    save_bin(a0, f'{input_path}/a0.bin', np.float64, transpose)
    # print(f'a0 = {a0}')

    '''network = read_network(data_set_path, layer_num, transpose, layer_info)
    x = read_inoutput(f'{data_set_path}/input0.bin', transpose, layer_info[0])'''
    mlp_forward(network, a0)
