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

def save_bin(data, bin_file, dtype=np.float64, transpose = False):
    if transpose == True:
        data = np.transpose(data)
    data = data.astype(np.float64)
    data.astype(dtype).tofile(bin_file)

def init_network(data_set_path, layer_num, layer_info, transpose = False):
    weights = []
    bias = []
    for i in range(layer_num):
        weights.append(np.random.random((layer_info[i][1], layer_info[i][2])))#, dtype=np.float64))
        bias.append(np.random.random((layer_info[i][0], layer_info[i][2])))#, dtype=np.float64))
        # weights.append(np.ones((layer_info[i][1], layer_info[i][2])))
        # bias.append(np.ones((layer_info[i][0], layer_info[i][2])))
    for i in range(layer_num):
        save_bin(weights[i], f'{data_set_path}/w{i}.bin', np.float64, transpose)
        save_bin(bias[i], f'{data_set_path}/b{i}.bin', np.float64, transpose)
    network = {'weights': weights, 'bias': bias}
    return network

def read_network(data_set_path, layer_num, transpose, layer_info):
    weights = []
    bias = []
    for i in range(layer_num):
        w_tmp = np.fromfile(f'{data_set_path}/w{0}.bin', dtype=np.float64)
        w_tmp.shape = (layer_info[i][0], layer_info[i][1])
        b_tmp = np.fromfile(f'{data_set_path}/b{0}.bin', dtype=np.float64)
        w_tmp.shape = (layer_info[i][0], layer_info[i][1])
        if transpose == True:
            w_tmp = np.transpose(w_tmp)
            b_tmp = np.transpose(b_tmp)
        weights.append(w_tmp)
        bias.append(w_tmp)
    network = {'weights': weights, 'bias': bias}
    return network

def read_inoutput(path, transpose, inout_info):
    input_tmp = np.fromfile(path, dtype=np.float64)
    if len(inout_info) == 2:
        input_tmp.shape = (inout_info[0], inout_info[1])
    elif len(inout_info) == 3:
        input_tmp.shape = (inout_info[0], inout_info[2])
    if transpose == True:
        input_tmp = np.transpose(input_tmp)
    return input_tmp

def check_double(input_0, input_1, shape, error_range):
    error_cnt = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if abs(input_0[i][j] - input_1[i][j]) > error_range:
                error_cnt = error_cnt + 1
    return error_cnt

if __name__ == "__main__":
    import os
    import sys
    
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
            
    network = init_network(input_path, layer_num, layer_info, transpose)
    a0 = np.random.random((layer_info[0][0], layer_info[0][1]))#, dtype=np.float64)
    # a0 = np.ones((layer_info[0][0], layer_info[0][1]))
    save_bin(a0, f'{input_path}/a0.bin', np.float64, transpose)
    # print(f'a0 = {a0}')