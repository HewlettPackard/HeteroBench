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

import torch
import torch.nn as nn
import numpy as np
import math
import time
import copy

def torch_softmax(x, dim=-1):
    m = torch.max(x, dim=dim, keepdim=True)[0]
    l = torch.sum(torch.exp(x - m), dim=dim, keepdim=True)
    s = torch.exp(x - m) / l
    return s

def torch_attn(query, key, value, mask=None, dropout=None, zoom=False):
    print('================= torch attention ==================')
    d_k = query.size(-1)
    s = torch.matmul(query, key.transpose(-2, -1))
    if zoom is True:
        s = s / math.sqrt(d_k)
    if mask is not None:
        s = s.masked_fill(mask == 0, -1e9)
    softmax_s = torch_softmax(s, dim=-1) # nn.functional.softmax(s, dim=-1)

    if dropout is not None:
        softmax_s = dropout(softmax_s)
    output = torch.matmul(softmax_s, value)
    print('================= torch attention end ==============\n')
    return output, softmax_s

def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    l = np.sum(np.exp(x - m), axis=axis, keepdims=True)
    s = np.exp(x - m) / l
    return s

def numpy_attn(query, key, value, mask=None, dropout=None, zoom=False):
    print('================= numpy attention ==================')
    d_k = query.shape[-1]
    s = np.matmul(query, key.swapaxes(-2, -1))
    if zoom is True:
        s = s / math.sqrt(d_k)
    if mask is not None:
        s = np.where(mask == 0, -1e9, s)
    softmax_s = softmax(s, axis=-1)

    if dropout is not None:
        softmax_s = dropout(softmax_s)
    output = np.matmul(softmax_s, value)
    print('================= numpy attention end ==============\n')
    return output, softmax_s

def manual_attn(query, key, value, mask=None, dropout=None, zoom=False):
    print('================= manual attention ==================')

    def transpose(x, dim0=-2, dim1=-1):
        # implement the x.swapaxes(-2, -1) manually
        shape = list(x.shape)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        transpose_x = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[dim0]):
                for k in range(shape[dim1]):
                    transpose_x[i, j, k] = x[i, k, j]

        return transpose_x

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
    
    def get_sum(x, axis=-1, keepdims=True):
        if axis == -1 and keepdims == True:
            sum_x = np.zeros(x.shape[:-1])
            for i in range(x.shape[-1]):
                sum_x += x[..., i]
            sum_x = np.expand_dims(sum_x, axis=-1)
        else:
            raise NotImplementedError("Not implemented yet  for axis != -1 or keepdims != True")
        return sum_x
    
    def get_exp(x):
        exp_x = np.zeros(x.shape)
        for i in range(x.shape[-1]):
            exp_x[..., i] = np.exp(x[..., i])
        return exp_x
    
    def softmax(x, axis=-1):
        m = get_max(x, axis=axis, keepdims=True)
        exp_result = get_exp(x - m)
        l = get_sum(exp_result, axis=axis, keepdims=True)
        s = exp_result / l
        return s
    
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
        
    d_k = query.shape[-1]
    transpose_key = transpose(key, dim0=-2, dim1=-1)
    s = matmul(query, transpose_key)
    s = zomm_mask(s, mask, d_k)
    softmax_s = softmax(s, axis=-1)

    if dropout is not None:
        softmax_s = dropout(softmax_s)
    output = matmul(softmax_s, value)
    print('================= manual attention end ==============\n')
    return output, softmax_s


batch_size = 8
# N = 8192
N = 256
d = 128
    
# torch_query = torch.rand(batch_size, N, d) * 256
# torch_key = torch.rand(batch_size, N, d) * 256
# torch_value = torch.rand(batch_size, N, d) * 256

# numpy_query = torch_query.numpy()
# numpy_key = torch_key.numpy()
# numpy_value = torch_value.numpy()

# # if we init date in torch and convert to numpy, the time is:
# # time of torch_attn =  2.84963321685791
# # time of numpy_attn_1 =  4.82873272895813

numpy_query = np.random.random((batch_size, N, d)) * 256
numpy_key = np.random.random((batch_size, N, d)) * 256
numpy_value = np.random.random((batch_size, N, d)) * 256

torch_query = torch.tensor(numpy_query)
torch_key = torch.tensor(numpy_key)
torch_value = torch.tensor(numpy_value)

# if we init date in numpy and convert to torch, the time is:
# time of torch_attn =  4.500532627105713
# time of numpy_attn_1 =  16.047845125198364

zoom = False
mask = None 
dropout = None

# test torch_attention
start_torch_attn = time.time()
output_torch_attn, softmax_s_torch_attn = \
    torch_attn(torch_query, torch_key, torch_value, mask, dropout, zoom)
end_torch_attn = time.time()

# test numpy_attn_1
start_numpy_attn_1 = time.time()
output_numpy_attn_1, softmax_s_numpy_attn_1 = \
    numpy_attn(numpy_query, numpy_key, numpy_value, mask, dropout, zoom)
end_numpy_attn_1 = time.time()

# test manual_attn
start_manual_attn = time.time()
output_manual_attn, softmax_s_manual_attn = \
    manual_attn(numpy_query, numpy_key, numpy_value, mask, dropout, zoom)
end_manual_attn = time.time()

print('time of torch_attn = ', end_torch_attn - start_torch_attn)
print('time of numpy_attn_1 = ', end_numpy_attn_1 - start_numpy_attn_1)
print('time of manual_attn = ', end_manual_attn - start_manual_attn)

print('================= check numpy_attn_1 ==================')
if np.allclose(softmax_s_torch_attn.numpy(), softmax_s_numpy_attn_1, atol=1e-1, rtol=1e-1):
    print('softmax_s_torch_attn == softmax_s_numpy_attn_1')
else:
    print('error: softmax_s_torch_attn != softmax_s_numpy_attn_1')
    print('softmax_s_torch_attn = ', softmax_s_torch_attn)
    print('softmax_s_numpy_attn_1 = ', softmax_s_numpy_attn_1)

if np.allclose(output_torch_attn.numpy(), output_numpy_attn_1, atol=1e-1, rtol=1e-1):
    print('output_torch_attn == output_numpy_attn_1')
else:
    print('error: output_torch_attn != output_numpy_attn_1')
    print('output_torch_attn = ', output_torch_attn)
    print('output_numpy_attn_1 = ', output_numpy_attn_1)
print('================= check numpy_attn_1 end ==============\n')

print('================= check manual_attn ==================')
if np.allclose(softmax_s_torch_attn.numpy(), softmax_s_manual_attn, atol=1e-1, rtol=1e-1):
    print('softmax_s_torch_attn == softmax_s_manual_attn')
else:
    print('error: softmax_s_torch_attn != softmax_s_manual_attn')
    print('softmax_s_torch_attn = ', softmax_s_torch_attn)
    print('softmax_s_manual_attn = ', softmax_s_manual_attn)

if np.allclose(output_torch_attn.numpy(), output_manual_attn, atol=1e-1, rtol=1e-1):
    print('output_torch_attn == output_manual_attn')
else:
    print('error: output_torch_attn != output_manual_attn')
    print('output_torch_attn = ', output_torch_attn)
    print('output_manual_attn = ', output_manual_attn)
print('================= check manual_attn end ==============\n')
