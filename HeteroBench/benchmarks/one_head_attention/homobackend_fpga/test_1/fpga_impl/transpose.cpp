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
 
#include "fpga_impl.h"
#include <iostream>
#include <math.h>

using namespace std;

/* This is the Cpp implementation of the following Python code */
/* Here the input and output are 3D arraies */
/*
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
*/

void transpose(double transpose_x[BATCH_SIZE * INPUT_H * INPUT_W], double transpose_output[BATCH_SIZE * INPUT_H * INPUT_W], 
              int dim0, int dim1) {
#pragma HLS interface m_axi offset=slave bundle=transpose_inout port=transpose_x max_read_burst_length=16 num_read_outstanding=64


#pragma HLS interface m_axi offset=slave bundle=transpose_inout port=transpose_output max_read_burst_length=16 num_read_outstanding=64


    if (dim0 == -2 && dim1 == -1) {
        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < INPUT_H; j++) {
                for (int k = 0; k < INPUT_W; k++) {
                    #pragma HLS PIPELINE II=1
                    transpose_output[i * INPUT_H * INPUT_W + k * INPUT_H + j] = 
                        transpose_x[i * INPUT_H * INPUT_W + j * INPUT_W + k];
                }
            }
        }
    }

}