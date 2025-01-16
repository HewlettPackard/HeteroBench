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
 
#include "fpga_impl.h"
#include <cstdio>

void buffer_frame2(pixel_t source[MAX_HEIGHT][MAX_WIDTH], pixel_t destination1[MAX_HEIGHT][MAX_WIDTH], pixel_t destination2[MAX_HEIGHT][MAX_WIDTH]) {
    for (int i = 0; i < MAX_HEIGHT; i++) {

        for (int j = 0; j < MAX_WIDTH; j++) {
#pragma HLS PIPELINE II = 1
            destination1[i][j] = source[i][j];
            destination2[i][j] = source[i][j];
        }
    }
}

void out2host(stream<velocity_t>& outputs_st, pixel_t outputs[MAX_HEIGHT][2 * MAX_WIDTH]) {
    //#pragma HLS INTERFACE axis port=outputs_st
    //#pragma HLS INTERFACE axis port=outputs
    for (int i = 0; i < MAX_HEIGHT; ++i) {
        for (int j = 0; j < MAX_WIDTH; ++j) {
#pragma HLS PIPELINE II=1
            velocity_t temp;
            temp = outputs_st.read();
            outputs[i][2 * j] = temp.x;
            outputs[i][2 * j + 1] = temp.y;
        }
    }
}

void optical_flow_hw(pixel_t frame0[MAX_HEIGHT][MAX_WIDTH],
    pixel_t frame1[MAX_HEIGHT][MAX_WIDTH],
    pixel_t frame2[MAX_HEIGHT][MAX_WIDTH],
    pixel_t frame3[MAX_HEIGHT][MAX_WIDTH],
    pixel_t frame4[MAX_HEIGHT][MAX_WIDTH],
    pixel_t outputs[MAX_HEIGHT][2 * MAX_WIDTH])
{
#pragma HLS interface m_axi offset=slave bundle=pixel_t_in0 port=frame0 max_read_burst_length=16 num_read_outstanding=64
#pragma HLS interface s_axilite bundle=control port=frame0

#pragma HLS interface m_axi offset=slave bundle=pixel_t_in1 port=frame1 max_read_burst_length=16 num_read_outstanding=64
#pragma HLS interface s_axilite bundle=control port=frame1

#pragma HLS interface m_axi offset=slave bundle=pixel_t_in2 port=frame2 max_read_burst_length=16 num_read_outstanding=64
#pragma HLS interface s_axilite bundle=control port=frame2

#pragma HLS interface m_axi offset=slave bundle=pixel_t_in3 port=frame3 max_read_burst_length=16 num_read_outstanding=64
#pragma HLS interface s_axilite bundle=control port=frame3

#pragma HLS interface m_axi offset=slave bundle=pixel_t_in4 port=frame4 max_read_burst_length=16 num_read_outstanding=64
#pragma HLS interface s_axilite bundle=control port=frame4

#pragma HLS interface m_axi offset=slave bundle=pixel_t_out port=outputs max_write_burst_length=16 num_write_outstanding=64
#pragma HLS interface s_axilite bundle=control port=outputs
    //pragma HLS interface mode= ap_ctrl_chain bundle=control port=return
    // #pragma HLS data_pack variable=outputs

#pragma HLS interface s_axilite bundle=control port=return

pixel_t buffer2_0[MAX_HEIGHT][MAX_WIDTH];
pixel_t buffer2_1[MAX_HEIGHT][MAX_WIDTH];

#pragma HLS STREAM variable=buffer2_0 depth=8*MAX_WIDTH
#pragma HLS STREAM variable=buffer2_1 depth=8*MAX_WIDTH


    stream<pixel_t> gradient_x_st;
    stream<pixel_t> gradient_y_st;
    stream<pixel_t> gradient_z_st;
    stream<gradient_t> y_filtered_st;
    stream<gradient_t> filtered_gradient_st;
    stream<outer_t> out_product_st;
    stream<tensor_t> tensor_y_st;
    stream<tensor_t> tensor_st;
    stream<velocity_t> out_st;

#pragma HLS STREAM variable=gradient_x_st depth=8*MAX_WIDTH
#pragma HLS STREAM variable=gradient_y_st depth=8*MAX_WIDTH
#pragma HLS STREAM variable=gradient_z_st depth=8*MAX_WIDTH
#pragma HLS STREAM variable=y_filtered_st depth=4*MAX_WIDTH
#pragma HLS STREAM variable=filtered_gradient_st depth=8*MAX_WIDTH
#pragma HLS STREAM variable=out_product_st depth=4*MAX_WIDTH
#pragma HLS STREAM variable=tensor_y_st depth=4*MAX_WIDTH
#pragma HLS STREAM variable=tensor_st depth=4*MAX_WIDTH

#pragma HLS DATAFLOW
    //buffer_frames(frame0, frame1, frame2, frame3, frame4, buffer0, buffer1, buffer2_0, buffer2_1, buffer3, buffer4);
    buffer_frame2(frame2, buffer2_0, buffer2_1);
    gradient_xy_calc_st(buffer2_1, gradient_x_st, gradient_y_st);
    gradient_z_calc_st(frame0, frame1, buffer2_0, frame3, frame4, gradient_z_st);
    gradient_weight_y_st(gradient_x_st, gradient_y_st, gradient_z_st, y_filtered_st);
    gradient_weight_x_st(y_filtered_st, filtered_gradient_st);
    outer_product_st(filtered_gradient_st, out_product_st);
    tensor_weight_y_st(out_product_st, tensor_y_st);
    tensor_weight_x_st(tensor_y_st, tensor_st);
    flow_calc_st(tensor_st, out_st);
    out2host(out_st, outputs);

}
