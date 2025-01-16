"""
(C) Copyright [2024] Hewlett Packard Enterprise Development LP

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the Software),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import cv2
import sys
import time

TEST_IMAGE_INPUT = "../input/1920x1080.jpg"
TEST_IMAGE_OUTPUT = "../output/1920x1080_python.jpg"

if len(sys.argv) == 4:
    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]
    iterations = int(sys.argv[3])
else:
    print("Did not provide command line arguments correctly. Using default values")
    print("Usage: python main.py <input_image_path> <output_image_path> <iterations>")
    input_image_path = TEST_IMAGE_INPUT
    output_image_path = TEST_IMAGE_OUTPUT 
    iterations = 20

def sobel_filter_x(input_image, height, width):
    sobel_x = np.zeros((height *width), np.double)
    kernel_x = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]]

    for row in range(1, height - 1, 1):
        for col in range(1, width - 1, 1):
            gx = 0
            for krow in range(-1, 2, 1):
                for kcol in range(-1, 2, 1):
                    pixel_val = input_image[(row + krow) * width + (col + kcol)]
                    gx += pixel_val * kernel_x[krow + 1][kcol + 1]
            sobel_x[row * width + col] = gx
    return sobel_x

def sobel_filter_y(input_image, height, width):
    sobel_y = np.zeros((height *width), np.double)
    kernel_y = [
    [-1, -2, -1],
    [0,  0,  0],
    [1,  2,  1]]

    for row in range(1, height - 1, 1):
        for col in range(1, width - 1, 1):
            gy = 0
            for krow in range(-1, 2, 1):
                for kcol in range(-1, 2, 1):
                    pixel_val = input_image[(row + krow) * width + (col + kcol)]
                    gy += pixel_val * kernel_y[krow + 1][kcol + 1]
            sobel_y[row * width + col] = gy
    return sobel_y

def compute_gradient_magnitude(sobel_x, sobel_y, height, width):
    gradient_magnitude = np.zeros((height *width), np.double)
    for i in range(0, height * width, 1):
        gradient_magnitude[i] = np.sqrt(sobel_x[i] * sobel_x[i] + sobel_y[i] * sobel_y[i])

    return gradient_magnitude

def sobel_filter(input_image, height, width):
    output_image = np.zeros((height *width), np.uint8)

    # 1 warm up iteration
    start_warmup_time = time.time()
    sobel_x = sobel_filter_x(input_image, height, width)
    sobel_y = sobel_filter_y(input_image, height, width)
    gradient_magnitude = compute_gradient_magnitude(sobel_x, sobel_y, height, width)
    warmup_time = time.time() - start_warmup_time
    print("Warmup Time:", warmup_time * 1000, "ms")

    # multi iterations
    start_whole_time = time.time()

    start_iteration_time = 0
    sobel_filter_x_time = 0
    sobel_filter_y_time = 0
    gradient_magnitude_time = 0

    for i in range(iterations):
        start_iteration_time = time.time() 
        sobel_x = sobel_filter_x(input_image, height, width)
        sobel_filter_x_time += time.time() - start_iteration_time

        start_iteration_time = time.time() 
        sobel_y = sobel_filter_y(input_image, height, width)
        sobel_filter_y_time += time.time() - start_iteration_time

        start_iteration_time = time.time() 
        gradient_magnitude = compute_gradient_magnitude(sobel_x, sobel_y, height, width)
        gradient_magnitude_time += time.time() - start_iteration_time

    run_whole_time = time.time() - start_whole_time
    print("1 warm up iteration and", iterations, "iterations")
    print("Single iteration time:", (run_whole_time / iterations) * 1000, "ms")
    print("Sobel filter x time:", (sobel_filter_x_time / iterations) * 1000, "ms")
    print("Sobel filter y time:", (sobel_filter_y_time / iterations) * 1000, "ms")
    print("Gradient magnitude time:", (gradient_magnitude_time / iterations) * 1000, "ms")

    for i in range( height * width):
        output_image[i] = np.uint8(gradient_magnitude[i])

    return output_image

def sobel_filter_wrapper(input_image_path, output_image_path):
    input_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    cv2.waitKey(0)
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    height, width, channel = input_image.shape

    sobel_image = sobel_filter(np.array(gray_image).flatten(), height, width)

    output_image = sobel_image.reshape(height, width)
    cv2.imwrite(output_image_path, output_image)
    cv2.destroyAllWindows()
    print("Output image at ", output_image_path)

def main():
    print("=======================================")
    print("Running sobel_filter benchmark Python")
    print("=======================================")
    sobel_filter_wrapper(input_image_path, output_image_path)

if __name__ == "__main__":
    main()