#include <iostream>
//#include <ap_int.h>
#include <hls_stream.h>
//#include <ap_axi_sdata.h>


typedef float pixel_t;
typedef float outer_pixel_t;
typedef double calc_pixel_t;
typedef float vel_pixel_t;
using namespace hls;
#define GRAD_WEIGHTS_SIZE 5
#define GRAD_FILTER_SIZE 7
#define TENSOR_FILTER_SIZE 3
#define COMPONENT_SIZE 6

// convolution filters
const int GRAD_WEIGHTS[] = { 1,-8,0,8,-1 };
const pixel_t GRAD_FILTER[] = { 0.0755, 0.133, 0.1869, 0.2903, 0.1869, 0.133, 0.0755 };
const pixel_t TENSOR_FILTER[] = { 0.3243, 0.3513, 0.3243 };

typedef struct {
    pixel_t x;
    pixel_t y;
    pixel_t z;
}gradient_t;

typedef struct {
    outer_pixel_t val[6];
}outer_t;
typedef struct {
    pixel_t frames[5];
} input_t;
typedef struct {
    outer_pixel_t val[6];
}tensor_t;

typedef struct {
    vel_pixel_t x;
    vel_pixel_t y;
}velocity_t;

void gradient_xy_calc(pixel_t frame[MAX_HEIGHT][MAX_WIDTH],
    pixel_t gradient_x[MAX_HEIGHT][MAX_WIDTH],
    pixel_t gradient_y[MAX_HEIGHT][MAX_WIDTH]);

void gradient_xy_calc_st(
    pixel_t frame[MAX_HEIGHT][MAX_WIDTH],
    stream<pixel_t>& gradient_x,
    stream<pixel_t>& gradient_y
);

void gradient_z_calc(pixel_t frame0[MAX_HEIGHT][MAX_WIDTH],
    pixel_t frame1[MAX_HEIGHT][MAX_WIDTH],
    pixel_t frame2[MAX_HEIGHT][MAX_WIDTH],
    pixel_t frame3[MAX_HEIGHT][MAX_WIDTH],
    pixel_t frame4[MAX_HEIGHT][MAX_WIDTH],
    pixel_t gradient_z[MAX_HEIGHT][MAX_WIDTH]);

void gradient_z_calc_st(
    pixel_t frame0[MAX_HEIGHT][MAX_WIDTH],
    pixel_t frame1[MAX_HEIGHT][MAX_WIDTH],
    pixel_t frame2[MAX_HEIGHT][MAX_WIDTH],
    pixel_t frame3[MAX_HEIGHT][MAX_WIDTH],
    pixel_t frame4[MAX_HEIGHT][MAX_WIDTH],
    stream<pixel_t>& gradient_z);

void gradient_weight_y(pixel_t gradient_x[MAX_HEIGHT][MAX_WIDTH],
    pixel_t gradient_y[MAX_HEIGHT][MAX_WIDTH],
    pixel_t gradient_z[MAX_HEIGHT][MAX_WIDTH],
    gradient_t filt_grad[MAX_HEIGHT][MAX_WIDTH]);
void gradient_weight_y_st(
    stream<pixel_t>& gradient_x_stream,
    stream<pixel_t>& gradient_y_stream,
    stream<pixel_t>& gradient_z_stream,
    stream<gradient_t>& filt_grad_stream);

void gradient_weight_x(gradient_t y_filt[MAX_HEIGHT][MAX_WIDTH],
    gradient_t filt_grad[MAX_HEIGHT][MAX_WIDTH]);

void gradient_weight_x_st(stream<gradient_t>& y_filt,
    stream<gradient_t>& filt_grad);

void outer_product(gradient_t gradient[MAX_HEIGHT][MAX_WIDTH],
    outer_t outer_product[MAX_HEIGHT][MAX_WIDTH]);

void outer_product_st(stream<gradient_t>& gradient,
    stream<outer_t>& outer_product);

void tensor_weight_y(outer_t outer[MAX_HEIGHT][MAX_WIDTH],
    tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH]);

void tensor_weight_y_st(stream<outer_t>& outer_st,
    stream<tensor_t>& tensor_y);

void tensor_weight_x_st(stream<tensor_t>& tensor_y_stream, stream<tensor_t>& tensor_stream);

void tensor_weight_x(tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH],
    tensor_t tensor[MAX_HEIGHT][MAX_WIDTH]);

void flow_calc(tensor_t tensors[MAX_HEIGHT][MAX_WIDTH],  //Input Struct Array
    velocity_t output[MAX_HEIGHT][MAX_WIDTH]);//Output Struct Arrays


void flow_calc_st(stream<tensor_t>& tensors,
    stream<velocity_t>& outputs_st);

void optical_flow_hw(pixel_t frame0[MAX_HEIGHT][MAX_WIDTH],
    pixel_t frame1[MAX_HEIGHT][MAX_WIDTH],
    pixel_t frame2[MAX_HEIGHT][MAX_WIDTH],
    pixel_t frame3[MAX_HEIGHT][MAX_WIDTH],
    pixel_t frame4[MAX_HEIGHT][MAX_WIDTH],
    pixel_t outputs[MAX_HEIGHT][2 * MAX_WIDTH]);