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
def relu(x):
  return np.maximum(0, x)
*/

void relu(double *relu_input, double *relu_output, int size) {
  #pragma acc data copyin(relu_input[0:size]) \
                         create(relu_output[0:size]) \
                         copyout(relu_output[0:size])
  
  {
    #pragma acc parallel loop
    for (int i = 0; i < size; i++) {
      relu_output[i] = max(0.0, relu_input[i]);
    }
  }
  // #pragma acc exit data map(from: relu_output[0:size])
  // #pragma acc exit data map(release: relu_input[0:size])
  // #pragma acc exit data map(release: size)
  
}