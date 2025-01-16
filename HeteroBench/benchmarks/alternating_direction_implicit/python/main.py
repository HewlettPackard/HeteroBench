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

# Parameters
# TSTEPS = 50
# N = 1024
# ITERATIONS = 10

if len(sys.argv) == 4:
    ITERATIONS = int(sys.argv[1])
    TSTEPS = int(sys.argv[2])
    N = int(sys.argv[3])
else:
    print("Did not provide command line arguments correctly. Using default values")
    print("Usage: python main.py <iterations> <tsteps> <n>")
    ITERATIONS = 10
    TSTEPS = 50
    N = 1024

def init_array(n, X, A, B):
    for c1 in range(n):
        for c2 in range(n):
            X[c1, c2] = ((c1 * (c2 + 1) + 1) / n)
            A[c1, c2] = ((c1 * (c2 + 2) + 2) / n)
            B[c1, c2] = ((c1 * (c2 + 3) + 3) / n)

def kernel_adi(tsteps, n, X, A, B):
    for c0 in range(tsteps):
        for c2 in range(n):
            for c8 in range(1, n):
                B[c2, c8] = B[c2, c8] - A[c2, c8] * A[c2, c8] / B[c2, c8 - 1]
            for c8 in range(1, n):
                X[c2, c8] = X[c2, c8] - X[c2, c8 - 1] * A[c2, c8] / B[c2, c8 - 1]
            for c8 in range(n - 2):
                X[c2, n - c8 - 2] = (X[c2, n - c8 - 2] - X[c2, n - c8 - 3] * A[c2, n - c8 - 3]) / B[c2, n - c8 - 3]
        for c2 in range(n):
            X[c2, n - 1] = X[c2, n - 1] / B[c2, n - 1]
        for c2 in range(n):
            for c8 in range(1, n):
                B[c8, c2] = B[c8, c2] - A[c8, c2] * A[c8, c2] / B[c8 - 1, c2]
            for c8 in range(1, n):
                X[c8, c2] = X[c8, c2] - X[c8 - 1, c2] * A[c8, c2] / B[c8 - 1, c2]
            for c8 in range(n - 2):
                X[n - c8 - 2, c2] = (X[n - c8 - 2, c2] - X[n - c8 - 3, c2] * A[n - c8 - 3, c2]) / B[n - c8 - 2, c2]
        for c2 in range(n):
            X[n - 1, c2] = X[n - 1, c2] / B[n - 1, c2]

def main():
    print("=======================================")
    print("Running adi benchmark Python")
    print("=======================================")

    n = N
    tsteps = TSTEPS

    X = np.zeros((N, N), dtype=np.float64)
    A = np.zeros((N, N), dtype=np.float64)
    B = np.zeros((N, N), dtype=np.float64)

    print("Running 1 warm up iteration ...")
    init_array(n, X, A, B)
    kernel_adi(tsteps, n, X, A, B)
    print("Done")

    print(f"Running {ITERATIONS} iterations ...")
    start_whole_time = time.time()
    init_array_time = 0
    kernel_adi_time = 0

    for _ in range(ITERATIONS):
        start_iteration_time = time.time()
        init_array(n, X, A, B)
        init_array_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        kernel_adi(tsteps, n, X, A, B)
        kernel_adi_time += time.time() - start_iteration_time
    print("Done")

    run_whole_time = time.time() - start_whole_time
    print(f"1 warm up iteration and {ITERATIONS} iterations")
    print(f"Single iteration time: {(run_whole_time / ITERATIONS) * 1000:.2f} ms")
    print(f"Init array time: {(init_array_time / ITERATIONS) * 1000:.2f} ms")
    print(f"Kernel adi time: {(kernel_adi_time / ITERATIONS) * 1000:.2f} ms")

if __name__ == "__main__":
    main()
