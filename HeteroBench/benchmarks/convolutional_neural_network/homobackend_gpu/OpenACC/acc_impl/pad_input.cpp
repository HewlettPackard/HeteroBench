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
/* Here the input and output are 2D arraies */
/*
def pad_input(input, padding):
  if padding == 0:
    return input
  padded_input = np.zeros((input.shape[0] + 2*padding, input.shape[1] + 2*padding))
  for i in range(input.shape[0]):
    for j in range(input.shape[1]):
      padded_input[i + padding][j + padding] = input[i][j]
  return padded_input
*/

void pad_input(double *pad_input_input, double *pad_input_output, int input_h, int input_w, int padding) 
{
  #pragma acc data copyin(pad_input_input[0:input_h*input_w], input_w, padding) \
                   create(pad_input_output[0:(input_h + 2 * padding) * (input_w + 2 * padding)]) \
                   copyout(pad_input_output[0:(input_h + 2 * padding) * (input_w + 2 * padding)])
  {
    #pragma acc parallel loop collapse(2)
    for (int i = 0; i < (input_h + 2 * padding); i++) {
      for (int j = 0; j < (input_w + 2 * padding); j++) {
        if(i < padding || i >= input_h+padding || j < padding || j >= input_w+padding) {
          pad_input_output[i * (input_w + 2 * padding) + j] = 0.0;
        } else {
          pad_input_output[i * (input_w + 2 * padding) + j] = pad_input_input[(i - padding) * input_w + (j - padding)];
        }
      }
    }
  }
}