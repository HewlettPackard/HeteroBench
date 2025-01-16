/*
 * (C) Copyright [2024] Hewlett Packard Enterprise Development LP
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the Software),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
 
#include "acc_impl.h"
#include <iostream>
#include <math.h>

using namespace std;

/* This is the Cpp implementation of the following Python code */
/* Here the input and output are 3D arraies */
/*
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
*/

void get_max(double *softmax_x, double *softmax_m, 
                int batch_size, int input_h, int input_w, int axis, bool keepdims) {
    // double *softmax_m = new double[batch_size * input_h];
    #pragma acc data copyin(softmax_x[0:batch_size * input_h * input_w]) \
                     create(softmax_m[0:batch_size * input_h]) \
                     copyout(softmax_m[0:batch_size * input_h])
    {                
        #pragma acc parallel loop
        for (int i = 0; i < batch_size * input_h; i++) {
            softmax_m[i] = -INFINITY;
        }

        #pragma acc parallel loop collapse(2)
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < input_h; j++) {
                for (int k = 0; k < input_w; k++) {
                    softmax_m[i * input_h + j] = max(softmax_m[i * input_h + j], softmax_x[i * input_h * input_w + j * input_w + k]);
                }
            }
        }
    }
}

/* This is the Cpp implementation of the following Python code */
/* Here the input and output are 3D arraies */
/*
    def get_sum(x, axis=-1, keepdims=True):
        if axis == -1 and keepdims == True:
            sum_x = np.zeros(x.shape[:-1])
            for i in range(x.shape[-1]):
                sum_x += x[..., i]
            sum_x = np.expand_dims(sum_x, axis=-1)
        else:
            raise NotImplementedError("Not implemented yet  for axis != -1 or keepdims != True")
        return sum_x
*/

void get_sum(double *softmax_exp_result, double *softmax_l, 
                int batch_size, int input_h, int input_w, int axis, bool keepdims) {
    // double *softmax_l = new double[batch_size * input_h];
    #pragma acc data copyin(softmax_exp_result[0:batch_size * input_h * input_w]) \
                     create(softmax_l[0:batch_size * input_h]) \
                     copyout(softmax_l[0:batch_size * input_h])
    {
        #pragma acc parallel loop
        for (int i = 0; i < batch_size * input_h; i++) {
            softmax_l[i] = 0;
        }
        #pragma acc parallel loop collapse(2)
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < input_h; j++) {
                for (int k = 0; k < input_w; k++) {
                    softmax_l[i * input_h + j] += softmax_exp_result[i * input_h * input_w + j * input_w + k];
                }
            }
        }
    }
}

/* This is the Cpp implementation of the following Python code */
/* Here the input and output are 3D arraies */
/*
    def get_exp(x):
        exp_x = np.zeros(x.shape)
        for i in range(x.shape[-1]):
            exp_x[..., i] = np.exp(x[..., i])
        return exp_x
*/
// get_exp(x_minus_m, softmax_exp_result, batch_size, input_h, input_w);
void get_exp(double *x_minus_m, double *softmax_exp_result, int batch_size, int input_h, int input_w) {
    #pragma acc data copyin(x_minus_m[0:batch_size * input_h * input_w]) \
                     create(softmax_exp_result[0:batch_size * input_h * input_w]) \
                     copyout(softmax_exp_result[0:batch_size * input_h * input_w])
    {
        #pragma acc parallel loop collapse(2)
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < input_h; j++) {
                for (int k = 0; k < input_w; k++) {
                    softmax_exp_result[i * input_h * input_w + j * input_w + k] = exp(x_minus_m[i * input_h * input_w + j * input_w + k]);
                }
            }
        }
    }
}

/* This is the Cpp implementation of the following Python code */
/* Here the input and output are 3D arraies */
/*
    def softmax(x, axis=-1):
        m = get_max(x, axis=axis, keepdims=True)
        exp_result = get_exp(x - m)
        l = get_sum(exp_result, axis=axis, keepdims=True)
        s = exp_result / l
        return s
*/

void softmax(double *softmax_x, double *softmax_output, 
             int batch_size, int input_h, int input_w, int axis) {
    double *softmax_m = new double[batch_size * input_h];
    double *x_minus_m = new double[batch_size * input_h * input_w];
    double *softmax_exp_result = new double[batch_size * input_h * input_w];
    double *softmax_l = new double[batch_size * input_h];

    get_max(softmax_x, softmax_m, batch_size, input_h, input_w, axis, true);

    #pragma acc data copyin(softmax_x[0:batch_size * input_h * input_w]) \
                        copyin(softmax_m[0:batch_size * input_h]) \
                        create(x_minus_m[0:batch_size * input_h * input_w]) \
                        copyout(x_minus_m[0:batch_size * input_h * input_w])
    {
        #pragma acc parallel loop collapse(2)
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < input_h; j++) {
                for (int k = 0; k < input_w; k++) {
                    x_minus_m[i * input_h * input_w + j * input_w + k] = 
                        softmax_x[i * input_h * input_w + j * input_w + k] - softmax_m[i * input_h + j];
                }
            }
        }
    }

    get_exp(x_minus_m, softmax_exp_result, batch_size, input_h, input_w);
    get_sum(softmax_exp_result, softmax_l, batch_size, input_h, input_w, axis, true);

    #pragma acc data copyin(softmax_l[0:batch_size * input_h]) \
                        copyin(softmax_exp_result[0:batch_size * input_h * input_w]) \
                        create(softmax_output[0:batch_size * input_h * input_w]) \
                        copyout(softmax_output[0:batch_size * input_h * input_w])
    {
        #pragma acc parallel loop collapse(2)
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < input_h; j++) {
                for (int k = 0; k < input_w; k++) {
                    softmax_output[i * input_h * input_w + j * input_w + k] = 
                        softmax_exp_result[i * input_h * input_w + j * input_w + k] / softmax_l[i * input_h + j];
                }
            }
        }
    }
}
