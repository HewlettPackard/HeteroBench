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

if len(sys.argv) == 7:
    iterations = int(sys.argv[1])
    ni = int(sys.argv[2])
    nj = int(sys.argv[3])
    nk = int(sys.argv[4])
    nl = int(sys.argv[5])
    nm = int(sys.argv[6])
else:
    print("Did not provide command line arguments correctly. Using default values")
    print("Usage: python main.py <iterations> <ni> <nj> <nk> <nl> <nm>")
    iterations = 20
    ni = 1024
    nj = 1024
    nk = 1024
    nl = 1024
    nm = 1024

def golden_kernel_3mm(A, B, C, D):
    # use numpy for reference
    E = np.dot(A, B)
    F = np.dot(C, D)
    G = np.dot(E, F)
    return G

def kernel_3m_0(A, B, E):
    for c1 in range(0, ni):
        for c2 in range(0, nj):
            for c5 in range(0, nk):
                E[c1][c2] += A[c1][c5] * B[c5][c2]
    # E = np.dot(A, B)
    # return E

def kernel_3m_1(C, D, F):
    for c1 in range(0, nj):
        for c2 in range(0, nl):
            for c5 in range(0, nm):
                F[c1][c2] += C[c1][c5] * D[c5][c2]
    # F = np.dot(C, D)
    # return F

def kernel_3m_2(E, F, G):
    for c1 in range(0, ni):
        for c2 in range(0, nj):
            for c6 in range(0, nl):
                G[c1][c6] += E[c1][c2] * F[c2][c6]
    # G = np.dot(E, F)
    # return G

def init_array_3m(A, B, C, D):
    # random data
    A[:] = np.random.rand(ni, nk)
    B[:] = np.random.rand(nk, nj)
    C[:] = np.random.rand(nj, nm)
    D[:] = np.random.rand(nm, nl)

def main():
    print("=======================================")
    print("Running 3mm benchmark Python")
    print("=======================================")

    A = np.zeros((ni, nk), dtype=np.float64)
    B = np.zeros((nk, nj), dtype=np.float64)
    C = np.zeros((nj, nm), dtype=np.float64)
    D = np.zeros((nm, nl), dtype=np.float64)
    E = np.zeros((ni, nj), dtype=np.float64)
    F = np.zeros((nj, nl), dtype=np.float64)
    G = np.zeros((ni, nl), dtype=np.float64)

    init_array_3m(A, B, C, D)

    print("Running 1 warm up iteration ...")
    kernel_3m_0(A, B, E)
    kernel_3m_1(C, D, F)
    kernel_3m_2(E, F, G)
    print("Done")

    print("Checking results ...")
    G_ref = golden_kernel_3mm(A, B, C, D)
    error = 0
    for i in range(ni):
        for j in range(nl):
            if abs(G_ref[i][j] - G[i][j]) > 0.1:
                error += 1
                print("Mismatch at position ({}, {}): expected {}, but got {}".format(i, j, G_ref[i][j], G[i][j]))
    print("Done")

    print(f"Running {iterations} iterations ...")
    start_whole_time = time.time()
    kernel_3m_0_time = 0
    kernel_3m_1_time = 0
    kernel_3m_2_time = 0

    for i in range(iterations):
        start_iteration_time = time.time()
        kernel_3m_0(A, B, E)
        kernel_3m_0_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        kernel_3m_1(C, D, F)
        kernel_3m_1_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        kernel_3m_2(E, F, G)
        kernel_3m_2_time += time.time() - start_iteration_time

    print("Done")

    run_whole_time = time.time() - start_whole_time
    print("1 warm up iteration and", iterations, "iterations")
    print(f"Single iteration time: {run_whole_time / iterations * 1000} ms")
    print(f"kernel_3m_0 time: {kernel_3m_0_time / iterations * 1000} ms")
    print(f"kernel_3m_1 time: {kernel_3m_1_time / iterations * 1000} ms")
    print(f"kernel_3m_2 time: {kernel_3m_2_time / iterations * 1000} ms")

if __name__ == "__main__":
    main()