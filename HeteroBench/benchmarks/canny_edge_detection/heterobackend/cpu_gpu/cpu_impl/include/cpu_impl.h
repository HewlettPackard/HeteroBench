#ifndef __CPU_IMPL_H__
#define __CPU_IMPL_H__
#include <iostream>
#define KERNEL_SIZE 1
#define OFFSET 1

void gradient_intensity_direction(const uint8_t *inImage, int height,
                                  int width, double *intensity,
                                  uint8_t *direction);
                                  
void edge_thinning(double *intensity,
                         uint8_t *direction, int height, int width,
                         double *outImage);

void double_thresholding(double *suppressed_image, int height, int width,
                  int high_threshold, int low_threshold,
                  uint8_t *outImage);
#endif