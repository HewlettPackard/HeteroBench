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

void gradient_xy_calc(pixel_t *frame2, pixel_t *gradient_x, pixel_t *gradient_y);

#endif
