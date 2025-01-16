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
 
#include "cpu_impl.h"
#include "gpu_impl.h"
// #include "fpga_impl.h"
#include "omp.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <vector>
#include <string>
#include <iostream>

#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <sys/stat.h>
#include <string>
// #include <ap_int.h>
#include <ctime>
#include <stdlib.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <experimental/xrt_xclbin.h>
#include <xrt/xrt_kernel.h>
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_uuid.h"

using namespace std;

#define DEVICE_ID 0

#define edge_thinning_ptr_gradient_intensity 0
#define edge_thinning_ptr_gradient_direction 1
#define edge_thinning_ptr_height 2
#define edge_thinning_ptr_width 3
#define edge_thinning_ptr_output_image 4

#define double_thresholding_ptr_suppressed_image 0
#define double_thresholding_ptr_height 1
#define double_thresholding_ptr_width 2
#define double_thresholding_ptr_high_threshold 3
#define double_thresholding_ptr_low_threshold 4
#define double_thresholding_ptr_output_image 5
struct Image {
    int width;
    int height;
    int channels;
    std::vector<unsigned char> data;

    Image(int w, int h, int c) : width(w), height(h), channels(c), data(w * h * c) {}
};

Image imread(const std::string& filename) {
    int width, height, channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    if (!data) {
        std::cerr << "Error: could not load image " << filename << std::endl;
        exit(1);
    }
    Image img(width, height, channels);
    std::copy(data, data + width * height * channels, img.data.begin());
    stbi_image_free(data);
    return img;
}

Image cvtColor(const Image& src) {
    if (src.channels != 3) {
        std::cerr << "Error: only 3-channel BGR images are supported" << std::endl;
        exit(1);
    }
    Image gray(src.width, src.height, 1);
    for (int i = 0; i < src.height; ++i) {
        for (int j = 0; j < src.width; ++j) {
            int index = i * src.width + j;
            unsigned char b = src.data[3 * index];
            unsigned char g = src.data[3 * index + 1];
            unsigned char r = src.data[3 * index + 2];
            gray.data[index] = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
        }
    }
    return gray;
}

void imwrite(const std::string& filename, const Image& img) {
    if (stbi_write_png(filename.c_str(), img.width, img.height, img.channels, img.data.data(), img.width * img.channels) == 0) {
        std::cerr << "Error: could not write image " << filename << std::endl;
        exit(1);
    }
}

void canny_edge_detect(const uint8_t *input_image, int height, int width,
                       int high_threshold, int low_threshold,
                       uint8_t *output_image) {

  uint64_t image_size = height * width;

  uint8_t *gaussian_filter_output = new uint8_t[image_size];
  double *gradient_intensity = new double[image_size];
  uint8_t *gradient_direction = new uint8_t[image_size];
  double *suppressed_output = new double[image_size];
  uint8_t *double_thresh_output = new uint8_t[image_size];
  
  // Load xclbin
  std::string xclbin_file = "overlay_hw.xclbin";
  std::cout << "Loading: " << xclbin_file << std::endl;
  xrt::device device = xrt::device(DEVICE_ID);
  xrt::uuid xclbin_uuid = device.load_xclbin(xclbin_file);
  std::cout << "Loaded xclbin: " << xclbin_file << std::endl;
  
  // create kernel object
  xrt::kernel edge_thinning_kernel = xrt::kernel(device, xclbin_uuid, "edge_thinning");
  xrt::kernel double_thresholding_kernel = xrt::kernel(device, xclbin_uuid, "double_thresholding");

  // create memory groups
  xrtMemoryGroup bank_grp_nms_gradient_intensity = edge_thinning_kernel.group_id(edge_thinning_ptr_gradient_intensity);
  xrtMemoryGroup bank_grp_nms_gradient_direction = edge_thinning_kernel.group_id(edge_thinning_ptr_gradient_direction);
  xrtMemoryGroup bank_grp_suppressed_output_image = edge_thinning_kernel.group_id(edge_thinning_ptr_output_image);

  xrtMemoryGroup bank_grp_thr_suppressed_image = double_thresholding_kernel.group_id(double_thresholding_ptr_suppressed_image);
  xrtMemoryGroup bank_grp_thr_output_image = double_thresholding_kernel.group_id(double_thresholding_ptr_output_image);

  // create buffer objects
  xrt::bo data_buffer_nms_gradient_intensity = xrt::bo(device, image_size * sizeof(double), xrt::bo::flags::normal, bank_grp_nms_gradient_intensity);
  xrt::bo data_buffer_nms_gradient_direction = xrt::bo(device, image_size * sizeof(uint8_t), xrt::bo::flags::normal, bank_grp_nms_gradient_direction);
  xrt::bo data_buffer_suppressed_output_image = xrt::bo(device, image_size * sizeof(double), xrt::bo::flags::normal, bank_grp_suppressed_output_image);

  xrt::bo data_buffer_thr_suppressed_image = xrt::bo(device, image_size * sizeof(double), xrt::bo::flags::normal, bank_grp_thr_suppressed_image);
  xrt::bo data_buffer_thr_output_image = xrt::bo(device, image_size * sizeof(uint8_t), xrt::bo::flags::normal, bank_grp_thr_output_image);

  // create kernel runner
  xrt::run run_edge_thinning(edge_thinning_kernel);
  xrt::run run_double_thresholding(double_thresholding_kernel);

  // 1 warm up iteration
  std::cout << "Running 1 warm up iteration ..." << std::endl;
  gaussian_filter(input_image, height, width, gaussian_filter_output);
  gradient_intensity_direction(gaussian_filter_output, height, width,
                               gradient_intensity, gradient_direction);

  // write data to buffer objects
  data_buffer_nms_gradient_intensity.write(gradient_intensity);
  data_buffer_nms_gradient_intensity.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_nms_gradient_direction.write(gradient_direction);
  data_buffer_nms_gradient_direction.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // set arguments of edge_thinning
  run_edge_thinning.set_arg(edge_thinning_ptr_gradient_intensity, data_buffer_nms_gradient_intensity);
  run_edge_thinning.set_arg(edge_thinning_ptr_gradient_direction, data_buffer_nms_gradient_direction);
  run_edge_thinning.set_arg(edge_thinning_ptr_height, height);
  run_edge_thinning.set_arg(edge_thinning_ptr_width, width);
  run_edge_thinning.set_arg(edge_thinning_ptr_output_image, data_buffer_suppressed_output_image);

  // run kernel
  run_edge_thinning.start();
  run_edge_thinning.wait();

  // Read back the results
  data_buffer_suppressed_output_image.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_suppressed_output_image.read(suppressed_output);

  // write data to buffer objects
  data_buffer_thr_suppressed_image.write(suppressed_output);
  data_buffer_thr_suppressed_image.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  
  // set arguments of double_thresholding
  run_double_thresholding.set_arg(double_thresholding_ptr_suppressed_image, data_buffer_thr_suppressed_image);
  run_double_thresholding.set_arg(double_thresholding_ptr_height, height);
  run_double_thresholding.set_arg(double_thresholding_ptr_width, width);
  run_double_thresholding.set_arg(double_thresholding_ptr_high_threshold, high_threshold);
  run_double_thresholding.set_arg(double_thresholding_ptr_low_threshold, low_threshold);
  run_double_thresholding.set_arg(double_thresholding_ptr_output_image, data_buffer_thr_output_image);

  // run kernel
  run_double_thresholding.start();
  run_double_thresholding.wait();

  // Read back the results
  data_buffer_thr_output_image.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_thr_output_image.read(double_thresh_output); 

  hysteresis(double_thresh_output, height, width, output_image);

  std::cout << "Done" << std::endl;

  // multi iterations
  int iterations = 5;
  std::cout << "Running " << iterations << " iterations ..." << std::endl;

  double start_whole_time = omp_get_wtime();

  double start_iteration_time;
  double gaussian_filter_time = 0;
  double gradient_time = 0;
  double supp_time = 0;
  double threshold_time = 0;
  double hysteresis_time = 0;
  for (int i = 0; i < iterations; i++) {
    start_iteration_time = omp_get_wtime();
    gaussian_filter(input_image, height, width, gaussian_filter_output);
    gaussian_filter_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    gradient_intensity_direction(gaussian_filter_output, height, width,
                                 gradient_intensity, gradient_direction);
    gradient_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    // write data to buffer objects
    data_buffer_nms_gradient_intensity.write(gradient_intensity);
    data_buffer_nms_gradient_intensity.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_nms_gradient_direction.write(gradient_direction);
    data_buffer_nms_gradient_direction.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set arguments of edge_thinning
    run_edge_thinning.set_arg(edge_thinning_ptr_gradient_intensity, data_buffer_nms_gradient_intensity);
    run_edge_thinning.set_arg(edge_thinning_ptr_gradient_direction, data_buffer_nms_gradient_direction);
    run_edge_thinning.set_arg(edge_thinning_ptr_height, height);
    run_edge_thinning.set_arg(edge_thinning_ptr_width, width);
    run_edge_thinning.set_arg(edge_thinning_ptr_output_image, data_buffer_suppressed_output_image);

    // run kernel
    run_edge_thinning.start();
    run_edge_thinning.wait();

    // Read back the results
    data_buffer_suppressed_output_image.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_suppressed_output_image.read(suppressed_output);
    supp_time += omp_get_wtime() - start_iteration_time;
    
    start_iteration_time = omp_get_wtime();
    // write data to buffer objects
    data_buffer_thr_suppressed_image.write(suppressed_output);
    data_buffer_thr_suppressed_image.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set arguments of double_thresholding
    run_double_thresholding.set_arg(double_thresholding_ptr_suppressed_image, data_buffer_thr_suppressed_image);
    run_double_thresholding.set_arg(double_thresholding_ptr_height, height);
    run_double_thresholding.set_arg(double_thresholding_ptr_width, width);
    run_double_thresholding.set_arg(double_thresholding_ptr_high_threshold, high_threshold);
    run_double_thresholding.set_arg(double_thresholding_ptr_low_threshold, low_threshold);
    run_double_thresholding.set_arg(double_thresholding_ptr_output_image, data_buffer_thr_output_image);

    // run kernel
    run_double_thresholding.start();
    run_double_thresholding.wait();

    // Read back the results
    data_buffer_thr_output_image.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_thr_output_image.read(double_thresh_output); 
    threshold_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    hysteresis(double_thresh_output, height, width, output_image);
    hysteresis_time += omp_get_wtime() - start_iteration_time;
  }
  std::cout << "Done" << std::endl;

  double run_whole_time = omp_get_wtime() - start_whole_time;
  cout << "1 warm up iteration and " << iterations << " iterations " << endl;
  cout << "Single iteration time: " << (run_whole_time / iterations) * 1000 << " ms" << endl;
  cout << "Gaussian Filter time: " << (gaussian_filter_time / iterations) * 1000 << " ms" << endl;
  cout << "Gradient time: " << (gradient_time / iterations) * 1000 << " ms" << endl;
  cout << "Edge Thinning time: " << (supp_time / iterations) * 1000 << " ms" << endl;
  cout << "Double Threshold time: " << (threshold_time / iterations) * 1000 << " ms" << endl;
  cout << "Hysteresis time: " << (hysteresis_time / iterations) * 1000 << " ms" << endl;

  delete[] gaussian_filter_output;
  delete[] gradient_intensity;
  delete[] gradient_direction;
  delete[] suppressed_output;
  delete[] double_thresh_output;
}

int main(int argc, char **argv) {
  std::cout << "=======================================" << std::endl;
  std::cout << "Running ced benchmark heterogeneous version, CPU+GPU+FPGA" << std::endl;
  std::cout << "=======================================" << std::endl;
  
  std::string input_image_path;
  std::string output_image_path;
  int low_threshold;
  int high_threshold;
  if (argc == 5) {
    input_image_path = argv[1];
    output_image_path = argv[2];
    low_threshold = std::stoi(argv[3]);
    high_threshold = std::stoi(argv[4]);
  } else {
    std::cerr << "Usage: ./ced <input_image> <output_image> <low_threshold> <high_threshold>" << std::endl;
    exit(-1);
  }

  Image input_image = imread(input_image_path);
  Image gray_image = cvtColor(input_image);

  int width = gray_image.width;
  int height = gray_image.height;
  Image canny_image(width, height, 1);
  canny_edge_detect(gray_image.data.data(), height, width, high_threshold, low_threshold, canny_image.data.data());
  imwrite(output_image_path, canny_image);

  return 0;
}
