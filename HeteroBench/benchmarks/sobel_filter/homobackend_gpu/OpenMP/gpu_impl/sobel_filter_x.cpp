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
#include "omp.h"
#include <cstring>
#include <iostream>
#include <math.h>

using namespace std;

void sobel_filter_x(const uint8_t *input_image, int height, int width, double *sobel_x) {
  const int kernel_x[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
  };

  #pragma omp target enter data \
    map(to: kernel_x[0:3][0:3]) \
    map(to: input_image[0:height*width]) \
    map(alloc: sobel_x[0:height*width])
  #pragma omp target teams distribute parallel for collapse(2)
  for (int row = 1; row < height - 1; ++row) {
    for (int col = 1; col < width - 1; ++col) {
      double gx = 0;
      for (int krow = -1; krow <= 1; ++krow) {
        for (int kcol = -1; kcol <= 1; ++kcol) {
          int pixel_val = input_image[(row + krow) * width + (col + kcol)];
          gx += pixel_val * kernel_x[krow + 1][kcol + 1];
        }
      }
      sobel_x[row * width + col] = gx;
    }
  }
  #pragma omp target exit data \
    map(from: sobel_x[0:height*width])
  #pragma omp target exit data \
    map(release: kernel_x[0:3][0:3]) \
    map(release: input_image[0:height*width]) \
    map(release: sobel_x[0:height*width])
}
