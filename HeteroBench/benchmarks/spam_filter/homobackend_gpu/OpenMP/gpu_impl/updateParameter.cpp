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
#include "math.h"

// Update the parameter vector
void updateParameter(
    FeatureType param[NUM_FEATURES],
    FeatureType grad[NUM_FEATURES],
    FeatureType step_size)
{
  #pragma omp target enter data \
  map(to: grad[0:NUM_FEATURES]) \
  map(to: param[0:NUM_FEATURES])
  #pragma omp target teams distribute parallel for
  for (int i = 0; i < NUM_FEATURES; i++) {
    param[i] += step_size * grad[i];
  }
  #pragma omp target exit data \
  map(from: param[0:NUM_FEATURES])
  #pragma omp target exit data \
  map(release: grad[0:NUM_FEATURES]) \
  map(release: param[0:NUM_FEATURES])
}
