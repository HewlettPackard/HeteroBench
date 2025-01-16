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
 
#include "cpu_impl.h"
#include "omp.h"
#include <cstring>
#include <iostream>
#include <math.h>

using namespace std;

void gaussian_filter(const uint8_t *inImage, int height, int width,
                   uint8_t *outImage) {
  const double kernel[9] = {0.0625, 0.125, 0.0625, 0.1250, 0.250, 0.1250, 0.0625, 0.125, 0.0625};
  memset(outImage, 0, height*width);

  for (int col = OFFSET; col < width - OFFSET; col++) {
    for (int row = OFFSET; row < height - OFFSET; row++) {
      double outIntensity = 0;
      int kIndex = 0;
      int pxIndex = col + (row * width);
      for (int krow = -OFFSET; krow <= OFFSET; krow++) {
        for (int kcol = -OFFSET; kcol <= OFFSET; kcol++) {
          outIntensity +=
              inImage[pxIndex + (kcol + (krow * width))] * kernel[kIndex];
          kIndex++;
        }
      }
      outImage[pxIndex] = (uint8_t)(outIntensity);
    }
  }
}