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
import math
from numba import jit

from tool import *
from util import *

if len(sys.argv) == 7:
    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    output_path = sys.argv[2]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    iterations = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    N = int(sys.argv[5])
    d = int(sys.argv[6])
    query = np.random.random((batch_size, N, d)) * 256
    key = np.random.random((batch_size, N, d)) * 256
    value = np.random.random((batch_size, N, d)) * 256
    zoom = False
    mask = None
    dropout = None
else:
    print("Did not provide command line arguments correctly. Using default values from util.py")
    from util import *

def one_head_attention(query, key, value, mask=None, dropout=None, zoom=False):
    print("=======================================")
    print("Running one-head attention benchmark Python")
    print("=======================================")

    @jit(nopython=True,cache=True)
    def transpose(x, dim0=-2, dim1=-1):
        # implement the x.swapaxes(-2, -1) manually
        shape = (x.shape[0], x.shape[2], x.shape[1])
        transpose_x = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[dim0]):
                for k in range(shape[dim1]):
                    transpose_x[i, j, k] = x[i, k, j]

        return transpose_x

    @jit(nopython=True,cache=True)
    def get_max(x, axis=-1, keepdims=True):
        if axis == -1 and keepdims == True:
            # init the max_x with np.-inf with the size of shape(x) except the last dimension (axis = -1)
            max_x = np.full(x.shape[:-1], -np.inf)
            # iterate over the last dimension of x
            for i in range(x.shape[-1]):
                max_x = np.maximum(max_x, x[..., i])
            # add the last dimension to max_x
            max_x = np.expand_dims(max_x, axis=-1)
        else:
            raise NotImplementedError("Not implemented yet  for axis != -1 or keepdims != True")
        return max_x
    
    @jit(nopython=True,cache=True)
    def get_sum(x, axis=-1, keepdims=True):
        if axis == -1 and keepdims == True:
            sum_x = np.zeros(x.shape[:-1])
            for i in range(x.shape[-1]):
                sum_x += x[..., i]
            sum_x = np.expand_dims(sum_x, axis=-1)
        else:
            raise NotImplementedError("Not implemented yet  for axis != -1 or keepdims != True")
        return sum_x
    
    @jit(nopython=True,cache=True)
    def get_exp(x):
        exp_x = np.zeros(x.shape)
        for i in range(x.shape[-1]):
            exp_x[..., i] = np.exp(x[..., i])
        return exp_x
    
    @jit(nopython=True,cache=True)
    def softmax(x, axis=-1):
        m = get_max(x, axis=axis, keepdims=True)
        exp_result = get_exp(x - m)
        l = get_sum(exp_result, axis=axis, keepdims=True)
        s = exp_result / l
        return s

    @jit(nopython=True,cache=True)
    def matmul(x, y):
        if len(x.shape) == 3 and len(y.shape) == 3:
            output = np.zeros((x.shape[0], x.shape[1], y.shape[2]))
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    for k in range(y.shape[2]):
                        for l in range(x.shape[2]):
                            output[i, j, k] += x[i, j, l] * y[i, l, k]
            return output
        else:
            raise NotImplementedError("Not implemented yet for x.shape != 3 or y.shape != 3")
        
    def zomm_mask(s, mask, d_k):
        if zoom is True:
            s = s / math.sqrt(d_k)
        if mask is not None:
            s = np.where(mask == 0, -1e9, s)
        return s
    
    # 1 warm up iteration
    print("Running 1 warm up iteration ...")
    
    save_bin(query, f'{input_path}/query.bin', np.float64, transpose = False)
    save_bin(key, f'{input_path}/key.bin', np.float64, transpose = False)
    save_bin(value, f'{input_path}/value.bin', np.float64, transpose = False)
    
    d_k = query.shape[-1]
    transpose_key = transpose(key, dim0=-2, dim1=-1)
    s = matmul(query, transpose_key)
    s = zomm_mask(s, mask, d_k)
    softmax_s = softmax(s, axis=-1)
    if dropout is not None:
        softmax_s = dropout(softmax_s)
    output_result = matmul(softmax_s, value)

    save_bin(output_result, f'{output_path}/matmul_softmax_value_golden.bin', np.float64, transpose = False)
    save_bin(transpose_key, f'{output_path}/transpose_key_golden.bin', np.float64, transpose = False)
    save_bin(s, f'{output_path}/matmul_query_key_golden.bin', np.float64, transpose = False)
    save_bin(softmax_s, f'{output_path}/softmax_result_golden.bin', np.float64, transpose = False)
    print("Done")

    # multi iterations
    print(f"Running {iterations} iterations ...")
    start_whole_time = time.time()

    transpose_time = 0
    matmul_time_1 = 0
    matmul_time_2 = 0
    softmax_time = 0
    # zoom_mask_time = 0

    for i in range(iterations):
        d_k = query.shape[-1]

        start_iteration_time = time.time()
        transpose_key = transpose(key, dim0=-2, dim1=-1)
        transpose_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        s = matmul(query, transpose_key)
        matmul_time_1 += time.time() - start_iteration_time

        # start_iteration_time = time.time()
        # s = zomm_mask(s, mask, d_k)
        # zoom_mask_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        softmax_s = softmax(s, axis=-1)
        softmax_time += time.time() - start_iteration_time

        # if dropout is not None:
        #     softmax_s = dropout(softmax_s)

        start_iteration_time = time.time()
        output_result = matmul(softmax_s, value)
        matmul_time_2 += time.time() - start_iteration_time

    print("Done")

    run_whole_time = time.time() - start_whole_time
    print("1 warm up iteration and", iterations, "iterations")
    print("Single iteration time:", (run_whole_time / iterations) * 1000, "ms")
    print("transpose kernel time:", (transpose_time / iterations) * 1000, "ms")
    print("matmul_1 kernel time:", (matmul_time_1 / iterations) * 1000, "ms")
    print("softmax kernel time:", (softmax_time / iterations) * 1000, "ms")
    print("matmul_2 kernel time:", (matmul_time_2 / iterations) * 1000, "ms")

    del query, key, value, output_result

if __name__ == "__main__":

    one_head_attention(query, key, value, mask, dropout, zoom)
