#ifndef __FPGA_IMPL_H__
#define __FPGA_IMPL_H__
#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#define KERNEL_SIZE 1
#define OFFSET 1

void edge_thinning(double* intensity, uint8_t* direction, int height, int width, double* outImage);
void double_thresholding(double* suppressed_image, int height, int width, int high_threshold, int low_threshold, uint8_t* outImage);

#endif