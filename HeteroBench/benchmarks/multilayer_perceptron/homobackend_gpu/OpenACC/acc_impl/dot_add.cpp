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
  def dot_add(x, W, b):
    mm = np.dot(x, W) + b
    return mm
*/

void dot_add(double *dot_add_input_x, double *dot_add_input_W, double *dot_add_input_b, double *dot_add_output, int x_h, int x_w, int W_h, int W_w) 
{
  #pragma acc data copyin(x_h) \
                   copyin(x_w) \
                   copyin(W_h) \
                   copyin(W_w) \
                   copyin(dot_add_input_x[:x_h*x_w]) \
                   copyin(dot_add_input_W[:x_w*W_w]) \
                   copyin(dot_add_input_b[:W_w]) \
                   create(dot_add_output[:x_h*W_w]) \
                   copyout(dot_add_output[:x_h*W_w])
  {
    #pragma acc parallel loop collapse(2)
    for (int i = 0; i < x_h; i++) {
      for (int j = 0; j < W_w; j++) {
        double tmp = 0;
        for (int k = 0; k < x_w; k++) {
          tmp += dot_add_input_x[i * x_w + k] * dot_add_input_W[k * W_w + j];
        }
        dot_add_output[i * W_w + j] = tmp + dot_add_input_b[j];
      }
    }
  }
}