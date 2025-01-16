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

void hysteresis(uint8_t *input_image, int height, int width,
                uint8_t *output_image) {
  memcpy(output_image, input_image, width * height * sizeof(uint8_t));
  #pragma omp parallel for collapse(2)
  for (int col = OFFSET; col < width - OFFSET; col++) {
    for (int row = OFFSET; row < height - OFFSET; row++) {
      int pixel_index = col + (row * width);
      if (output_image[pixel_index] == 100) {
        if (output_image[pixel_index - 1] == 255 ||
            output_image[pixel_index + 1] == 255 ||
            output_image[pixel_index - width] == 255 ||
            output_image[pixel_index + width] == 255 ||
            output_image[pixel_index - width - 1] == 255 ||
            output_image[pixel_index - width + 1] == 255 ||
            output_image[pixel_index + width - 1] == 255 ||
            output_image[pixel_index + width + 1] == 255)
          output_image[pixel_index] = 255;
        else
          output_image[pixel_index] = 0;
      }
    }
  }
}
