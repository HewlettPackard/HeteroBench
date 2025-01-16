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
#include "omp.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <vector>
#include <string>
#include <iostream>

using namespace std;

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
  
  // 1 warm up iteration
  std::cout << "Running 1 warm up iteration ..." << std::endl;
  gaussian_filter(input_image, height, width, gaussian_filter_output);
  gradient_intensity_direction(gaussian_filter_output, height, width,
                               gradient_intensity, gradient_direction);
  edge_thinning(gradient_intensity, gradient_direction, height, width,
                      suppressed_output);
  double_thresholding(suppressed_output, height, width, high_threshold, low_threshold,
               double_thresh_output);
  hysteresis(double_thresh_output, height, width, output_image);
  std::cout << "Done" << std::endl;

  // multi iterations
  int iterations = ITERATIONS;
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
    edge_thinning(gradient_intensity, gradient_direction, height, width,
                        suppressed_output);
    supp_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    double_thresholding(suppressed_output, height, width, high_threshold, low_threshold,
                 double_thresh_output);
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
  std::cout << "Running ced benchmark C++ Serial" << std::endl;
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