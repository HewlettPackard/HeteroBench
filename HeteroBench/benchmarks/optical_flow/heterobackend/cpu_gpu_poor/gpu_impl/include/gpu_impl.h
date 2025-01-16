#ifndef __GPU_IMPL_H__
#define __GPU_IMPL_H__

#include "cpu_impl.h"
// typedef float pixel_t;
// typedef float outer_pixel_t;
// typedef double calc_pixel_t;
// typedef float vel_pixel_t;

// // convolution filters
// const int GRAD_WEIGHTS[] =  {1,-8,0,8,-1};
// const pixel_t GRAD_FILTER[] = {0.0755, 0.133, 0.1869, 0.2903, 0.1869, 0.133, 0.0755};
// const pixel_t TENSOR_FILTER[] = {0.3243, 0.3513, 0.3243};

// typedef struct{
// 	pixel_t x;
// 	pixel_t y;
// 	pixel_t z;
// }gradient_t;

// typedef struct{
//     outer_pixel_t val[6];
// }outer_t; 

// typedef struct{
//     outer_pixel_t val[6];
// }tensor_t;

// typedef struct{
//     vel_pixel_t x;
//     vel_pixel_t y;
// }velocity_t;

void gradient_weight_y(pixel_t *gradient_x, pixel_t *gradient_y, pixel_t *gradient_z,
                       gradient_t *y_filtered);

void tensor_weight_y(outer_t *out_product, tensor_t *tensor_y);

void tensor_weight_x(tensor_t *tensor_y, tensor_t *tensor);

void gradient_z_calc(pixel_t *frame0, pixel_t *frame1, pixel_t *frame2,
                     pixel_t *frame3, pixel_t *frame4, pixel_t *gradient_z);

void gradient_weight_x(gradient_t *y_filtered, gradient_t *filtered_gradient);

void outer_product(gradient_t *filtered_gradient, outer_t *out_product);

void flow_calc(tensor_t *tensor, velocity_t *outputs);

#endif
