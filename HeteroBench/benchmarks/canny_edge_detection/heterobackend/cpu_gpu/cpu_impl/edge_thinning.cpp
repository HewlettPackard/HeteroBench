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

void edge_thinning(double *intensity,
                         uint8_t *direction, int height, int width,
                         double *outImage) {
  memcpy(outImage, intensity, width * height * sizeof(double));
  #pragma omp parallel for collapse(2)
  for (int col = OFFSET; col < width - OFFSET; col++) {
    for (int row = OFFSET; row < height - OFFSET; row++) {
      int pxIndex = col + (row * width);

      // unconditionally suppress border pixels
      if (row == OFFSET || col == OFFSET || col == width - OFFSET - 1 ||
          row == height - OFFSET - 1) {
        outImage[pxIndex] = 0;
        continue;
      }

      switch (direction[pxIndex]) {
      case 1:
        if (intensity[pxIndex - 1] >= intensity[pxIndex] ||
            intensity[pxIndex + 1] > intensity[pxIndex])
          outImage[pxIndex] = 0;
        break;
      case 2:
        if (intensity[pxIndex - (width - 1)] >= intensity[pxIndex] ||
            intensity[pxIndex + (width - 1)] > intensity[pxIndex])
          outImage[pxIndex] = 0;
        break;
      case 3:
        if (intensity[pxIndex - (width)] >= intensity[pxIndex] ||
            intensity[pxIndex + (width)] > intensity[pxIndex])
          outImage[pxIndex] = 0;
        break;
      case 4:
        if (intensity[pxIndex - (width + 1)] >= intensity[pxIndex] ||
            intensity[pxIndex + (width + 1)] > intensity[pxIndex])
          outImage[pxIndex] = 0;
        break;
      default:
        outImage[pxIndex] = 0;
        break;
      }
    }
  }
}