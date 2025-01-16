#ifndef __FPGA_IMPL_H__
#define __FPGA_IMPL_H__
#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#define KERNEL_SIZE 1
#define OFFSET 1

void gaussian_filter(const uint8_t *inImage, int height, int width, uint8_t *outImage);
void gradient_intensity_direction(const uint8_t* inImage, int height, int width, double* intensity, uint8_t* direction);
void edge_thinning(double* intensity, uint8_t* direction, int height, int width, double* outImage);
void double_thresholding(double* suppressed_image, int height, int width, int high_threshold, int low_threshold, uint8_t* outImage);
void hysteresis(uint8_t* inImage, int height, int width, uint8_t* outImage);

#endif