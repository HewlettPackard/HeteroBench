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
 
#include "gpu_impl.h"
#include <iostream>
#include <math.h>

using namespace std;

/* This is the Cpp implementation of the following Python code */
/* Here the input and output are 3D arraies */
/*
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
*/

void matmul(double *matmul_x, double *matmul_y, double *matmul_output, 
                int batch_size, int input_h, int input_w, int output_w) {

    #pragma omp target enter data map(to: matmul_x[0:batch_size * input_h * input_w], matmul_y[0:batch_size * input_w * output_w]) map(alloc: matmul_output[0:batch_size * input_h * output_w])
    // init matmul_output to 0
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < batch_size * input_h * output_w; i++) {
        matmul_output[i] = 0;
    }
    #pragma omp target teams distribute parallel for collapse(3)
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < input_h; j++) {
            for (int k = 0; k < output_w; k++) {
                for (int l = 0; l < input_w; l++) {
                    matmul_output[i * input_h * output_w + j * output_w + k] += 
                    matmul_x[i * input_h * input_w + j * input_w + l] * 
                    matmul_y[i * input_w * output_w + l * output_w + k];
                }
            }
        }
    }
    #pragma omp target exit data map(from: matmul_output[0:batch_size * input_h * output_w]) map(release: matmul_x[0:batch_size * input_h * input_w], matmul_y[0:batch_size * input_w * output_w])
}
