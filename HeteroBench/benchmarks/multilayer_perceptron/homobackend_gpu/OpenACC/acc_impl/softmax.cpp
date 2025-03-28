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
def softmax(input_0):
  exp_input_0 = np.exp(input_0)
  sum_total_0 = np.sum(exp_input_0)
  output_0 = exp_input_0 / sum_total_0
  return output_0
*/

void softmax(double *softmax_input, double *exp_results, double *softmax_output, int size) {
  double sum_total_0 = 0;

  #pragma acc data copyin(softmax_input[:size]) \
                   create(exp_results[:size]) \
                   create(softmax_output[:size]) \
                   copyout(softmax_output[:size])
  {
    #pragma acc parallel loop reduction(+:sum_total_0)
    for (int i = 0; i < size; i++) {
      exp_results[i] = exp(softmax_input[i]);
      sum_total_0 += exp_results[i];
    }

    #pragma acc parallel loop
    for (int i = 0; i < size; i++) {
      softmax_output[i] = exp_results[i] / sum_total_0;
    }
  }

}