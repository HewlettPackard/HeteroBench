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

void matmul1(double matmul_x[BATCH_SIZE * INPUT_H * OUTPUT_W], double matmul_y[BATCH_SIZE * OUTPUT_W * INPUT_W], double matmul_output[BATCH_SIZE * INPUT_H * INPUT_W])
{
 
#pragma HLS interface m_axi offset=slave bundle=mat_x_inout port=matmul_x max_read_burst_length=16 num_read_outstanding=64
#pragma HLS interface m_axi offset=slave bundle=mat_y_in port=matmul_y max_read_burst_length=16 num_read_outstanding=64
#pragma HLS interface m_axi offset=slave bundle=mat_x_inout  port=matmul_output max_read_burst_length=16 num_read_outstanding=64


// RESET TO ZERO NO NEED
    // for (int i = 0; i < BATCH_SIZE * INPUT_H * INPUT_W; i++) {
    //     matmul_output[i] = 0;
    // }

    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < INPUT_H; j++) {
            for (int k = 0; k < INPUT_W; k++) {
                for (int l = 0; l < OUTPUT_W; l++) {
#pragma HLS PIPELINE II = 1
                    matmul_output[i * INPUT_H * INPUT_W + j * INPUT_W + k] += 
                    matmul_x[i * INPUT_H * OUTPUT_W + j * OUTPUT_W + l] * 
                    matmul_y[i * OUTPUT_W * INPUT_W + l * INPUT_W + k];
                }
            }
        }
    }
}
