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
#include "omp.h"
#include <cstring>
#include <iostream>
#include <math.h>

using namespace std;

void gradient_intensity_direction(const uint8_t *inImage, int height,
                                  int width, double *intensity,
                                  uint8_t *direction) {
  const int8_t Gx[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  const int8_t Gy[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
  memset(intensity, 0.0, height*width);
  memset(direction, 0, height*width);

  #pragma acc data copyin(inImage[0:height*width], Gx[0:9], Gy[0:9]) \
        copy(intensity[0:height*width], direction[0:height*width])
  {
    #pragma acc parallel loop collapse(2)
    for (int col = OFFSET; col < width - OFFSET; col++) {
      for (int row = OFFSET; row < height - OFFSET; row++) {
        double Gx_sum = 0.0;
        double Gy_sum = 0.0;
        int kIndex = 0;
        int pxIndex = col + (row * width);

        for (int krow = -OFFSET; krow <= OFFSET; krow++) {
          for (int kcol = -OFFSET; kcol <= OFFSET; kcol++) {
            Gx_sum += inImage[pxIndex + (kcol + (krow * width))] * Gx[kIndex];
            Gy_sum += inImage[pxIndex + (kcol + (krow * width))] * Gy[kIndex];
            kIndex++;
          }
        }

        if (Gx_sum == 0.0 || Gy_sum == 0.0) {
          intensity[pxIndex] = 0.0;
          direction[pxIndex] = 0;
        } else {
          intensity[pxIndex] = std::sqrt((Gx_sum * Gx_sum) + (Gy_sum * Gy_sum));
          double theta = std::atan2(Gy_sum, Gx_sum);
          theta = theta * (360.0 / (2.0 * M_PI));

          if ((theta <= 22.5 && theta >= -22.5) || (theta <= -157.5) || (theta >= 157.5))
            direction[pxIndex] = 1;  // horizontal -
          else if ((theta > 22.5 && theta <= 67.5) ||
                  (theta > -157.5 && theta <= -112.5))
            direction[pxIndex] = 2;  // north-east -> south-west /
          else if ((theta > 67.5 && theta <= 112.5) ||
                  (theta >= -112.5 && theta < -67.5))
            direction[pxIndex] = 3;  // vertical |
          else if ((theta >= -67.5 && theta < -22.5) ||
                  (theta > 112.5 && theta < 157.5))
            direction[pxIndex] = 4;  // north-west -> south-east \'
        }
      }
    }
  }
}