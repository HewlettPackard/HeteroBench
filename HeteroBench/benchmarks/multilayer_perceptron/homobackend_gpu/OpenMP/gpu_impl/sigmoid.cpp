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
/* Here the input and output are 2D arraies */
/*
def sigmoid(a):
  z = 1 / (1 + np.exp(-a))
  return z
*/

void sigmoid(double *sigmoid_input, double *sigmoid_output, int size) {
  #pragma omp target enter data map(to: size)
  #pragma omp target enter data map(to: sigmoid_input[:size])
  #pragma omp target enter data map(alloc: sigmoid_output[:size])

  #pragma omp target teams distribute parallel for
  for (int i = 0; i < size; i++) {
    sigmoid_output[i] = 1 / (1 + exp(-sigmoid_input[i]));
  }
  #pragma omp target exit data map(from: sigmoid_output[:size])
}