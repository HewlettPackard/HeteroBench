#ifndef __GPU_IMPL_H__
#define __GPU_IMPL_H__
#include <iostream>
#define KERNEL_SIZE 1
#define OFFSET 1

void gaussian_filter(const uint8_t *inImage, int height, int width,
                   uint8_t *outImage);

void hysteresis(uint8_t *inImage, int height, int width,
                uint8_t *outImage);
#endif