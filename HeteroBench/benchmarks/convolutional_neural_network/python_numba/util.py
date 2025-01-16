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

import os

conv2d_stride = 1
conv2d_padding = 1
conv2d_bias = 0.1
pooling_size=2
pooling_stride=2
input_size = (1024, 2048)
conv_kernel_size = (3, 3)
# full_connect_layer_size = (523953, 2048)
conv_output_height = (input_size[0] - conv_kernel_size[0]  + 2 * conv2d_padding) // conv2d_stride + 1
conv_output_width = (input_size[1] - conv_kernel_size[1] + 2 * conv2d_padding) // conv2d_stride + 1
pool_output_height = (conv_output_height - pooling_size) // pooling_stride + 1
pool_output_width = (conv_output_width - pooling_size) // pooling_stride + 1

flattened_output_size = pool_output_height * pool_output_width

full_connect_layer_size = (flattened_output_size, 2048)

transpose = False

iterations = 1

data_set_path = f'dataset'
if not os.path.exists(data_set_path):
    os.makedirs(data_set_path)