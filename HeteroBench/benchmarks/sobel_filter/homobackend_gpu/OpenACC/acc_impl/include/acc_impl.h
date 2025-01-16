#ifndef __ACC_IMPL_H__
#define __ACC_IMPL_H__
#include <iostream>

void sobel_filter_x(const uint8_t *input_image, int height, int width, double *sobel_x);
void sobel_filter_y(const uint8_t *input_image, int height, int width, double *sobel_y);
void compute_gradient_magnitude(const double *sobel_x, const double *sobel_y, int height, int width, double *gradient_magnitude);

#endif // __ACC_IMPL_H__