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


void softmax(double softmax_x[BATCH_SIZE * INPUT_H * INPUT_H], 
             double softmax_output[BATCH_SIZE * INPUT_H * INPUT_H], 
             int axis) {

    #pragma HLS interface m_axi offset=slave bundle=softmax_inout port=softmax_x max_read_burst_length=16 num_read_outstanding=64
    #pragma HLS interface m_axi offset=slave bundle=softmax_inout port=softmax_output max_read_burst_length=16 num_read_outstanding=64

    double softmax_m_line;  // Store the maximum value of each row
    double softmax_l_line;  // Store the sum of each row
    double x_minus_m_line[INPUT_H];  // Store the result of each row after subtracting the maximum
    double softmax_exp_result_line[INPUT_H];  // Store the exponentiation result of each row

    // Iterate through each batch and row
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < INPUT_H; j++) {
            int line = i * INPUT_H + j;  // Calculate the current row index

            // Step 1: Calculate the maximum value of this row
            softmax_m_line = -INFINITY;
            for (int k = 0; k < INPUT_H; k++) {
                #pragma HLS PIPELINE II=1
                softmax_m_line = max(softmax_m_line, softmax_x[line * INPUT_H + k]);
            }

            // Step 2: Calculate x - max(x) for this row
            for (int k = 0; k < INPUT_H; k++) {
                #pragma HLS PIPELINE II=1
                x_minus_m_line[k] = softmax_x[line * INPUT_H + k] - softmax_m_line;
            }

            // Step 3: Calculate the exponentiation of x_minus_m for this row
            for (int k = 0; k < INPUT_H; k++) {
                #pragma HLS PIPELINE II=1
                softmax_exp_result_line[k] = exp(x_minus_m_line[k]);
            }

            // Step 4: Calculate the sum of the exponentiation results for this row
            softmax_l_line = 0;
            for (int k = 0; k < INPUT_H; k++) {
                #pragma HLS PIPELINE II=1
                softmax_l_line += softmax_exp_result_line[k];
            }

            // Step 5: Calculate the softmax output for this row
            for (int k = 0; k < INPUT_H; k++) {
                #pragma HLS PIPELINE II=1
                softmax_output[line * INPUT_H + k] = softmax_exp_result_line[k] / softmax_l_line;
            }
        }
    }
}