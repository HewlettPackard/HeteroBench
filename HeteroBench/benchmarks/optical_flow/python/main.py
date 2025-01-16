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
import sys
import time

# cpu_impl.h
if len(sys.argv) == 6:
    dataPath = sys.argv[1]
    outFile = sys.argv[2]
    max_height = int(sys.argv[3])
    max_width = int(sys.argv[4])
    iterations = int(sys.argv[5])
else:
    print("Did not provide command line arguments correctly. Using default values")
    print("Usage: python main.py <dataPath> <outFile> <max_height> <max_width> <iterations>")
    dataPath = "benchmarks/optical_flow/datasets/larger"
    outFile = "benchmarks/optical_flow/datasets/output/acc_output.flo"
    max_height = 2160
    max_width = 3840
    iterations = 20

pixel_t = np.float32
outer_pixel_t = np.float32
calc_pixel_t = np.float64
vel_pixel_t = np.float32

GRAD_WEIGHTS = np.array([1, -8, 0, 8, -1], dtype=np.int32)
GRAD_FILTER = np.array([0.0755, 0.133, 0.1869, 0.2903, 0.1869, 0.133, 0.0755], dtype=pixel_t)
TENSOR_FILTER = np.array([0.3243, 0.3513, 0.3243], dtype=pixel_t)

class gradient_t:
    def __init__(self, x=pixel_t(0), y=pixel_t(0), z=pixel_t(0)):
        self.x = x
        self.y = y
        self.z = z

class outer_t:
    def __init__(self):
        self.val = np.zeros(6, dtype=outer_pixel_t)

class tensor_t:
    def __init__(self):
        self.val = np.zeros(6, dtype=outer_pixel_t)

class velocity_t:
    def __init__(self, x=vel_pixel_t(0), y=vel_pixel_t(0)):
        self.x = x
        self.y = y

# gradient_xy_calc.cpp
def gradient_xy_calc(frame, gradient_x, gradient_y):
    for r in range(max_height + 2):
        for c in range(max_width + 2):
            x_grad = pixel_t(0)
            y_grad = pixel_t(0)
            if r >= 4 and r < max_height and c >= 4 and c < max_width:
                for i in range(5):
                    x_grad += frame[r-2, c-i] * GRAD_WEIGHTS[4-i]
                    y_grad += frame[r-i, c-2] * GRAD_WEIGHTS[4-i]
                gradient_x[r-2, c-2] = x_grad / 12
                gradient_y[r-2, c-2] = y_grad / 12
            elif r >= 2 and c >= 2:
                gradient_x[r-2, c-2] = 0
                gradient_y[r-2, c-2] = 0

# gradient_z_calc.cpp
def gradient_z_calc(frame0, frame1, frame2, frame3, frame4, gradient_z):
    for r in range(max_height):
        for c in range(max_width):
            gradient_z[r, c] = 0.0
            gradient_z[r, c] += frame0[r, c] * GRAD_WEIGHTS[0]
            gradient_z[r, c] += frame1[r, c] * GRAD_WEIGHTS[1]
            gradient_z[r, c] += frame2[r, c] * GRAD_WEIGHTS[2]
            gradient_z[r, c] += frame3[r, c] * GRAD_WEIGHTS[3]
            gradient_z[r, c] += frame4[r, c] * GRAD_WEIGHTS[4]
            gradient_z[r, c] /= 12.0

# gradient_weight_y.cpp
def gradient_weight_y(gradient_x, gradient_y, gradient_z, filt_grad):
    for r in range(max_height + 3):
        for c in range(max_width):
            acc = gradient_t()
            acc.x = 0.0
            acc.y = 0.0
            acc.z = 0.0
            if r >= 6 and r < max_height:
                for i in range(7):
                    acc.x += gradient_x[r-i, c] * GRAD_FILTER[i]
                    acc.y += gradient_y[r-i, c] * GRAD_FILTER[i]
                    acc.z += gradient_z[r-i, c] * GRAD_FILTER[i]
                filt_grad[r-3, c] = acc
            elif r >= 3:
                filt_grad[r-3, c] = acc

# gradient_weight_x.cpp
def gradient_weight_x(y_filt, filt_grad):
    for r in range(max_height):
        for c in range(max_width + 3):
            acc = gradient_t()
            acc.x = 0.0
            acc.y = 0.0
            acc.z = 0.0
            if c >= 6 and c < max_width:
                for i in range(7):
                    acc.x += y_filt[r, c-i].x * GRAD_FILTER[i]
                    acc.y += y_filt[r, c-i].y * GRAD_FILTER[i]
                    acc.z += y_filt[r, c-i].z * GRAD_FILTER[i]
                filt_grad[r, c-3] = acc
            elif c >= 3:
                filt_grad[r, c-3] = acc

# outer_product.cpp
def outer_product(gradient, outer_product):
    for r in range(max_height):
        for c in range(max_width):
            grad = gradient[r, c]
            out = outer_t()
            out.val[0] = grad.x * grad.x
            out.val[1] = grad.y * grad.y
            out.val[2] = grad.z * grad.z
            out.val[3] = grad.x * grad.y
            out.val[4] = grad.x * grad.z
            out.val[5] = grad.y * grad.z
            outer_product[r, c] = out

# tensor_weight_y.cpp
def tensor_weight_y(outer, tensor_y):
    for r in range(max_height + 1):
        for c in range(max_width):
            acc = tensor_t()
            for k in range(6):
                acc.val[k] = 0.0

            if r >= 2 and r < max_height:
                for i in range(3):
                    for component in range(6):
                        acc.val[component] += outer[r-i, c].val[component] * TENSOR_FILTER[i]
            if r >= 1:
                tensor_y[r-1, c] = acc

# tensor_weight_x.cpp
def tensor_weight_x(tensor_y, tensor):
    for r in range(max_height):
        for c in range(max_width + 1):
            acc = tensor_t()
            for k in range(6):
                acc.val[k] = 0.0

            if c >= 2 and c < max_width:
                for i in range(3):
                    for component in range(6):
                        acc.val[component] += tensor_y[r, c-i].val[component] * TENSOR_FILTER[i]
            if c >= 1:
                tensor[r, c-1] = acc

# flow_calc.cpp
def flow_calc(tensors, output):
    for r in range(max_height):
        for c in range(max_width):
            if 2 <= r < max_height - 2 and 2 <= c < max_width - 2:
                denom = tensors[r, c].val[0] * tensors[r, c].val[1] - tensors[r, c].val[3] * tensors[r, c].val[3]
                # the following line is not existing in the original code
                # add to avoid Python runtime error about division by zero
                if denom != 0:
                    output[r, c].x = (tensors[r, c].val[5] * tensors[r, c].val[3] - tensors[r, c].val[4] * tensors[r, c].val[1]) / denom
                    output[r, c].y = (tensors[r, c].val[4] * tensors[r, c].val[3] - tensors[r, c].val[5] * tensors[r, c].val[0]) / denom
            else:
                output[r, c].x = 0
                output[r, c].y = 0

# main.cpp
# ignore check results for now
def optical_flow_sw(frame0, frame1, frame2, frame3, frame4, outputs, outFile):
    gradient_x = np.zeros((max_height, max_width), dtype=pixel_t)
    gradient_y = np.zeros((max_height, max_width), dtype=pixel_t)
    gradient_z = np.zeros((max_height, max_width), dtype=pixel_t)
    y_filtered = np.empty((max_height, max_width), dtype=object)
    filtered_gradient = np.empty((max_height, max_width), dtype=object)
    out_product = np.empty((max_height, max_width), dtype=object)
    tensor_y = np.empty((max_height, max_width), dtype=object)
    tensor = np.empty((max_height, max_width), dtype=object)

    for i in range(max_height):
        for j in range(max_width):
            y_filtered[i][j] = gradient_t()
            filtered_gradient[i][j] = gradient_t()
            out_product[i][j] = outer_t()
            tensor_y[i][j] = tensor_t()
            tensor[i][j] = tensor_t()

    print("Running 1 warm up iteration ...")
    gradient_xy_calc(frame2, gradient_x, gradient_y)
    gradient_z_calc(frame0, frame1, frame2, frame3, frame4, gradient_z)
    gradient_weight_y(gradient_x, gradient_y, gradient_z, y_filtered)
    gradient_weight_x(y_filtered, filtered_gradient)
    outer_product(filtered_gradient, out_product)
    tensor_weight_y(out_product, tensor_y)
    tensor_weight_x(tensor_y, tensor)
    flow_calc(tensor, outputs)
    print("Done")


    print(f"Running {iterations} iterations ...")
    start_whole_time = time.time()

    gradient_xy_calc_time = 0
    gradient_z_calc_time = 0
    gradient_weight_y_time = 0
    gradient_weight_x_time = 0
    outer_product_time = 0
    tensor_weight_y_time = 0
    tensor_weight_x_time = 0
    flow_calc_time = 0

    for iter in range(iterations):
        start_iteration_time = time.time()
        gradient_xy_calc(frame2, gradient_x, gradient_y)
        gradient_xy_calc_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        gradient_z_calc(frame0, frame1, frame2, frame3, frame4, gradient_z)
        gradient_z_calc_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        gradient_weight_y(gradient_x, gradient_y, gradient_z, y_filtered)
        gradient_weight_y_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        gradient_weight_x(y_filtered, filtered_gradient)
        gradient_weight_x_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        outer_product(filtered_gradient, out_product)
        outer_product_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        tensor_weight_y(out_product, tensor_y)
        tensor_weight_y_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        tensor_weight_x(tensor_y, tensor)
        tensor_weight_x_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        flow_calc(tensor, outputs)
        flow_calc_time += time.time() - start_iteration_time

    print("Done")

    run_whole_time = time.time() - start_whole_time
    print(f"1 warm up iteration and {iterations} iterations")
    print(f"Single iteration time: {run_whole_time / iterations * 1000:.2f} ms")
    print(f"gradient_xy_calc time: {gradient_xy_calc_time / iterations * 1000:.2f} ms")
    print(f"gradient_z_calc time: {gradient_z_calc_time / iterations * 1000:.2f} ms")
    print(f"gradient_weight_y time: {gradient_weight_y_time / iterations * 1000:.2f} ms")
    print(f"gradient_weight_x time: {gradient_weight_x_time / iterations * 1000:.2f} ms")
    print(f"outer_product time: {outer_product_time / iterations * 1000:.2f} ms")
    print(f"tensor_weight_y time: {tensor_weight_y_time / iterations * 1000:.2f} ms")
    print(f"tensor_weight_x time: {tensor_weight_x_time / iterations * 1000:.2f} ms")
    print(f"flow_calc time: {flow_calc_time / iterations * 1000:.2f} ms")

def main():
    print("=======================================")
    print("Running optical_flow benchmark Python")
    print("=======================================")

    frame_files = [f"{dataPath}/frame{i+1}.ppm" for i in range(5)]

    print("Reading input files ...")
    import cv2
    imgs = [cv2.imread(f) for f in frame_files]
    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

    frames = np.zeros((5, max_height, max_width), dtype=pixel_t)
    outputs = np.empty((max_height, max_width), dtype=velocity_t)

    for i in range(max_height):
        for j in range(max_width):
            outputs[i, j] = velocity_t()

    for f in range(5):
        frames[f] = imgs_gray[f]

    optical_flow_sw(frames[0], frames[1], frames[2], frames[3], frames[4], outputs, outFile)

if __name__ == "__main__":
    main()