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

void compute_gradient_magnitude(const double *sobel_x, const double *sobel_y, int height, int width, double *gradient_magnitude) {

    #pragma omp target enter data \
        map(to: sobel_x[0:height*width]) \
        map(to: sobel_y[0:height*width]) \
        map(alloc: gradient_magnitude[0:height*width])
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < height * width; ++i) {
        gradient_magnitude[i] = sqrt(sobel_x[i] * sobel_x[i] + sobel_y[i] * sobel_y[i]);
    }
    #pragma omp target exit data \
        map(from: gradient_magnitude[0:height*width])
    #pragma omp target exit data \
        map(release: sobel_x[0:height*width]) \
        map(release: sobel_y[0:height*width])
}
