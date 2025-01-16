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

if len(sys.argv) == 6:
    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]
    low_threshold = int(sys.argv[3])
    high_threshold = int(sys.argv[4])
    iterations = int(sys.argv[5])
    OFFSET = 1
else:
    print("Did not provide command line arguments correctly. Using default values from util.py")
    print("Usage: python main.py <input_image_path> <output_image_path> <low_threshold> <high_threshold> <iterations>")
    from util import *
    input_image_path = TEST_IMAGE_INPUT
    output_image_path = TEST_IMAGE_OUTPUT 
    low_threshold = 30
    high_threshold = 90
    iterations = 20

def gaussian_filter(inImage, height, width):
    kernel = [0.0625, 0.125, 0.0625, 0.1250, 0.250, 0.1250, 0.0625, 0.125, 0.0625]
    outImage = np.zeros((height *width), np.uint8)
    
    for col in range(OFFSET, width - OFFSET, 1):
        for row in range(OFFSET, height - OFFSET, 1):
            outIntensity = 0.0
            kIndex = 0
            pxIndex = col + (row * width)
            for krow in range(-OFFSET, OFFSET+1, 1):
                for kcol in range(-OFFSET, OFFSET+1, 1):
                    outIntensity += inImage[pxIndex + (kcol + (krow * width))] * kernel[kIndex]
                    kIndex += 1
            outImage[pxIndex] = np.uint8(outIntensity)
    
    return outImage

def gradient_intensity_direction(inImage, height, width):
    Gx = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
    Gy = [1, 2, 1, 0, 0, 0, -1, -2, -1]
    intensity = np.zeros((height*width), np.double)
    direction = np.zeros((height*width), np.uint8)
    
    for col in range(OFFSET, width - OFFSET, 1):
        for row in range(OFFSET, height - OFFSET, 1):
            Gx_sum = 0.0
            Gy_sum = 0.0
            kIndex = 0
            pxIndex = col + (row * width)

            for krow in range(-OFFSET, OFFSET+1, 1):
                for kcol in range(-OFFSET, OFFSET+1, 1):
                    Gx_sum += inImage[pxIndex + (kcol + (krow * width))] * Gx[kIndex]
                    Gy_sum += inImage[pxIndex + (kcol + (krow * width))] * Gy[kIndex]
                    kIndex += 1

            if (Gx_sum == 0.0) or (Gy_sum == 0.0):
                intensity[pxIndex] = 0.0
                direction[pxIndex] = np.uint8(0)
            else:
                intensity[pxIndex] = np.sqrt(Gx_sum**2 + Gy_sum**2)
                theta = np.arctan2(Gy_sum, Gx_sum)
                theta = theta * (360.0 / (2.0 * np.pi))

                if (theta <= 22.5 and theta >= -22.5) or (theta <= -157.5) or (theta >= 157.5):
                    direction[pxIndex] = np.uint8(1)    # horizontal -
                elif (theta > 22.5 and theta <= 67.5) or (theta > -157.5 and theta <= -112.5):
                    direction[pxIndex] = np.uint8(2)    # north-east -> south-west /
                elif (theta > 67.5 and theta <= 112.5) or (theta >= -112.5 and theta < -67.5):
                    direction[pxIndex] = np.uint8(3)    # vertical |
                elif (theta >= -67.5 and theta < -22.5) or (theta > 112.5 and theta < 157.5):
                    direction[pxIndex] = np.uint8(4)    # north-west -> south-east \'

    return intensity, direction

def edge_thinning(intensity, direction, height, width):
    outImage = intensity.copy()
    
    for col in range(OFFSET, width - OFFSET, 1):
        for row in range(OFFSET, height - OFFSET, 1):
            pxIndex = col + (row * width)

            # unconditionally suppress border pixels
            if ((row == OFFSET) or (col == OFFSET) or (col == width - OFFSET - 1) or (row == height - OFFSET - 1)):
                outImage[pxIndex] = 0
                continue

            match direction[pxIndex]:
                case 1:
                    if ( (intensity[pxIndex - 1] >= intensity[pxIndex]) or \
                         (intensity[pxIndex + 1] >  intensity[pxIndex]) ):
                        outImage[pxIndex] = 0

                case 2:
                    if ( (intensity[pxIndex - (width - 1)] >= intensity[pxIndex]) or \
                         (intensity[pxIndex + (width - 1)] > intensity[pxIndex]) ):
                        outImage[pxIndex] = 0

                case 3:
                    if (intensity[pxIndex - (width)] >= intensity[pxIndex]) or \
                        (intensity[pxIndex + (width)] > intensity[pxIndex]):
                        outImage[pxIndex] = 0

                case 4:
                    if (intensity[pxIndex - (width + 1)] >= intensity[pxIndex]) or \
                        (intensity[pxIndex + (width + 1)] > intensity[pxIndex]):
                        outImage[pxIndex] = 0

                case _:
                    outImage[pxIndex] = 0

    return outImage

def double_thresholding(suppressed_image, height, width, high_threshold, low_threshold):
    outImage = np.zeros((height*width), np.uint8)

    for col in range(0, width, 1):
        for row in range(0, height, 1):
            pxIndex = col + (row * width)
            if (suppressed_image[pxIndex] > high_threshold):
                outImage[pxIndex] = 255   # Strong edge
            elif (suppressed_image[pxIndex] > low_threshold):
                outImage[pxIndex] = 100;  # Weak edge
            else:
                outImage[pxIndex] = 0     # Not an edge

    return outImage

def hysteresis(inImage, height, width):
    outImage = inImage.copy()
    
    for col in range(OFFSET, width - OFFSET, 1):
        for row in range(OFFSET, height - OFFSET, 1):
            pxIndex = col + (row * width)
            if (outImage[pxIndex] == 100):
                if (outImage[pxIndex - 1] == 255) or \
                   (outImage[pxIndex + 1] == 255) or \
                   (outImage[pxIndex - width] == 255) or \
                   (outImage[pxIndex + width] == 255) or \
                   (outImage[pxIndex - width - 1] == 255) or \
                   (outImage[pxIndex - width + 1] == 255) or \
                   (outImage[pxIndex + width - 1] == 255) or \
                   (outImage[pxIndex + width + 1] == 255):
                    outImage[pxIndex] = 255
                else:
                    outImage[pxIndex] = 0

    return outImage

def canny_edge_detect(inImage, height, width, high_threshold, low_threshold):
    # 1 warm up iteration
    print("Running 1 warm up iteration ...")
    gaussian_filter_output = gaussian_filter(inImage, height, width)
    gradient_intensity, gradient_direction = gradient_intensity_direction(gaussian_filter_output, height, width)
    suppressed_output = edge_thinning(gradient_intensity, gradient_direction, height, width)
    double_thresh_output = double_thresholding(suppressed_output, height, width, high_threshold, low_threshold)
    outImage = hysteresis(double_thresh_output, height, width)
    print("Done")

    # multi iterations
    print("Running", iterations, "iteration ...")
    start_whole_time = time.time()

    gaussian_filter_time = 0
    gradient_time = 0
    supp_time = 0
    threshold_time = 0
    hysteresis_time = 0

    for i in range(iterations):
        start_iteration_time = time.time() 
        gaussian_filter_output = gaussian_filter(inImage, height, width)
        gaussian_filter_time += time.time()  - start_iteration_time

        start_iteration_time = time.time() 
        gradient_intensity, gradient_direction = gradient_intensity_direction(gaussian_filter_output, height, width)
        gradient_time += time.time()  - start_iteration_time

        start_iteration_time = time.time() 
        suppressed_output = edge_thinning(gradient_intensity, gradient_direction, height, width)
        supp_time += time.time()  - start_iteration_time

        start_iteration_time = time.time() 
        double_thresh_output = double_thresholding(suppressed_output, height, width, high_threshold, low_threshold)
        threshold_time += time.time()  - start_iteration_time

        start_iteration_time = time.time() 
        outImage = hysteresis(double_thresh_output, height, width)
        hysteresis_time += time.time()  - start_iteration_time
    
    run_whole_time = time.time() - start_whole_time
    print("Done")
    print("1 warm up iteration and", iterations, "iterations")
    print("Single iteration time:", (run_whole_time / iterations) * 1000, "ms")
    print("Gaussian Filter time:", (gaussian_filter_time / iterations) * 1000, "ms")
    print("Gradient time:", (gradient_time / iterations) * 1000, "ms")
    print("Edge Thinning time:", (supp_time / iterations) * 1000, "ms")
    print("Double Thresholding time:", (threshold_time / iterations) * 1000, "ms")
    print("Hysteresis time:", (hysteresis_time / iterations) * 1000, "ms")

    del gaussian_filter_output
    del gradient_intensity
    del gradient_direction
    del suppressed_output
    del double_thresh_output

    return outImage

def main():

    print("=======================================")
    print("Running ced benchmark Python")
    print("=======================================")
    input_image = cv2.imread(input_image_path)
    cv2.waitKey(0)
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    height, width, channel = input_image.shape
    
    canny_image = canny_edge_detect(np.array(gray_image).flatten(), height, width, high_threshold, low_threshold)
    
    output_image = canny_image.reshape(height, width)
    cv2.imwrite(output_image_path, output_image)
    cv2.destroyAllWindows()
    print("Output image at ", output_image_path)

if __name__ == "__main__":
    main()