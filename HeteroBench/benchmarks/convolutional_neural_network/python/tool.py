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

def init_network(data_set_path, input_size, conv_kernel_size, full_connect_layer_size, transpose = False):
    W_conv = np.random.random((conv_kernel_size[0], conv_kernel_size[1]))
    # W_conv = np.ones((conv_kernel_size[0], conv_kernel_size[1]))
    print(f'W_conv.shape = {W_conv.shape}')
    W_fc = np.random.random((full_connect_layer_size[0], full_connect_layer_size[1]))
    # W_fc = np.ones((full_connect_layer_size[0], full_connect_layer_size[1]))
    print(f'W_fc.shape = {W_fc.shape}')
    b_fc = np.random.random((full_connect_layer_size[1], ))
    # b_fc = np.ones((full_connect_layer_size[1], ))
    print(f'b_fc.shape = {b_fc.shape}')
    
    save_bin(W_conv, f'{data_set_path}/W_conv.bin', np.float64, transpose)
    save_bin(W_fc, f'{data_set_path}/W_fc.bin', np.float64, transpose)
    save_bin(b_fc, f'{data_set_path}/b_fc.bin', np.float64, transpose)
    network = {'weights': [W_conv, W_fc], 'bias': [b_fc]} #, b_conv,]}
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

    network = init_network(input_path, input_size, conv_kernel_size, full_connect_layer_size, transpose)
    input_image = np.random.random((input_size))
    # input_image = np.ones((input_size))

    save_bin(input_image, f'{input_path}/input_image.bin', np.float64, transpose)
    print(f'input_image = {input_image}')