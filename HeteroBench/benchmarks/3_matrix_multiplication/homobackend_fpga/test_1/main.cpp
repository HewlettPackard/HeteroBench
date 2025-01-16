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
 
#include "omp.h"
#include "init_array.h"
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

#include "fpga_impl.h"

using namespace std;

#define DEVICE_ID 0

#define kernel_3mm_ptr_E 0
#define kernel_3mm_ptr_A 1
#define kernel_3mm_ptr_B 2
#define kernel_3mm_ptr_F 3
#define kernel_3mm_ptr_C 4
#define kernel_3mm_ptr_D 5
#define kernel_3mm_ptr_G 6

void* xmalloc (size_t num)
{
  void* nnew = NULL;
  int ret = posix_memalign (&nnew, 32, num);
  if (! nnew || ret)
    {
      fprintf (stderr, "[PolyBench] posix_memalign: cannot allocate memory");
      exit (1);
    }
  return nnew;
}

void* polybench_alloc_data(unsigned long long int n, int elt_size)
{
  /// FIXME: detect overflow!
  size_t val = n;
  val *= elt_size;
  void* ret = xmalloc (val);

  return ret;
}

void golden_kernel_3mm(double E[NI + 0][NJ + 0],double A[NI + 0][NK + 0],double B[NK + 0][NJ + 0],double F[NJ + 0][NL + 0],double C[NJ + 0][NM + 0],double D[NM + 0][NL + 0],double G[NI + 0][NL + 0])
{
  int c1;
  int c2;
  int c5;
  int c6;

  for (c1 = 0; c1 <= NI - 1; c1++) {
    for (c2 = 0; c2 <= NL - 1; c2++) {
      G[c1][c2] = 0;
    }
  }

  for (c1 = 0; c1 <= NJ - 1; c1++) {
    for (c2 = 0; c2 <= NL - 1; c2++) {
      F[c1][c2] = 0;
    }
  }

  for (c1 = 0; c1 <= NJ - 1; c1++) {
    for (c2 = 0; c2 <= NL - 1; c2++) {
      for (c5 = 0; c5 <= NM - 1; c5++) {
      F[c1][c2] += C[c1][c5] * D[c5][c2];
      }
    }
  }

  for (c1 = 0; c1 <= NI - 1; c1++) {
    for (c2 = 0; c2 <= NJ - 1; c2++) {
      E[c1][c2] = 0;
    }
  }

  for (c1 = 0; c1 <= NI - 1; c1++) {
    for (c2 = 0; c2 <= NJ - 1; c2++) {
      for (c5 = 0; c5 <= NK - 1; c5++) {
      E[c1][c2] += A[c1][c5] * B[c5][c2];
      }
      for (c6 = 0; c6 <= NL - 1; c6++) {
      G[c1][c6] += E[c1][c2] * F[c2][c6];
      }
    }
  }
}

int main(int argc,char **argv)
{
  std::cout << "=======================================" << std::endl;
  std::cout << "Running 3mm benchmark C++ HLS" << std::endl;
  std::cout << "=======================================" << std::endl;
  
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;
  int nm = NM;

  /* Variable declaration/allocation. */
  double (*E)[NI + 0][NJ + 0];
  E = ((double (*)[NI + 0][NJ + 0])(polybench_alloc_data(((NI + 0) * (NJ + 0)),(sizeof(double )))));
  double (*A)[NI + 0][NK + 0];
  A = ((double (*)[NI + 0][NK + 0])(polybench_alloc_data(((NI + 0) * (NK + 0)),(sizeof(double )))));
  double (*B)[NK + 0][NJ + 0];
  B = ((double (*)[NK + 0][NJ + 0])(polybench_alloc_data(((NK + 0) * (NJ + 0)),(sizeof(double )))));
  double (*F)[NJ + 0][NL + 0];
  F = ((double (*)[NJ + 0][NL + 0])(polybench_alloc_data(((NJ + 0) * (NL + 0)),(sizeof(double )))));
  double (*C)[NJ + 0][NM + 0];
  C = ((double (*)[NJ + 0][NM + 0])(polybench_alloc_data(((NJ + 0) * (NM + 0)),(sizeof(double )))));
  double (*D)[NM + 0][NL + 0];
  D = ((double (*)[NM + 0][NL + 0])(polybench_alloc_data(((NM + 0) * (NL + 0)),(sizeof(double )))));
  double (*G)[NI + 0][NL + 0];
  G = ((double (*)[NI + 0][NL + 0])(polybench_alloc_data(((NI + 0) * (NL + 0)),(sizeof(double )))));

  init_array(ni, nj, nk, nl, nm, *A, *B, *C, *D, *E, *F, *G);

  // Load xclbin
  std::string xclbin_file = "overlay_hw.xclbin";
  std::cout << "Loading: " << xclbin_file << std::endl;
  xrt::device device = xrt::device(DEVICE_ID);
  xrt::uuid xclbin_uuid = device.load_xclbin(xclbin_file);
  std::cout << "Loaded xclbin: " << xclbin_file << std::endl;
  
  // create kernel object
  xrt::kernel kernel_3mm_kernel = xrt::kernel(device, xclbin_uuid, "kernel_3mm");

  // create memory groups
  xrtMemoryGroup bank_grp_kernel_3mm_E = kernel_3mm_kernel.group_id(kernel_3mm_ptr_E);
  xrtMemoryGroup bank_grp_kernel_3mm_A = kernel_3mm_kernel.group_id(kernel_3mm_ptr_A);
  xrtMemoryGroup bank_grp_kernel_3mm_B = kernel_3mm_kernel.group_id(kernel_3mm_ptr_B);
  xrtMemoryGroup bank_grp_kernel_3mm_F = kernel_3mm_kernel.group_id(kernel_3mm_ptr_F);
  xrtMemoryGroup bank_grp_kernel_3mm_C = kernel_3mm_kernel.group_id(kernel_3mm_ptr_C);
  xrtMemoryGroup bank_grp_kernel_3mm_D = kernel_3mm_kernel.group_id(kernel_3mm_ptr_D);
  xrtMemoryGroup bank_grp_kernel_3mm_G = kernel_3mm_kernel.group_id(kernel_3mm_ptr_G);

  // create buffer objects
  xrt::bo data_buffer_kernel_3mm_E = xrt::bo(device, sizeof(double) * (NI + 0) * (NJ + 0), xrt::bo::flags::normal, bank_grp_kernel_3mm_E);
  xrt::bo data_buffer_kernel_3mm_A = xrt::bo(device, sizeof(double) * (NI + 0) * (NK + 0), xrt::bo::flags::normal, bank_grp_kernel_3mm_A);
  xrt::bo data_buffer_kernel_3mm_B = xrt::bo(device, sizeof(double) * (NK + 0) * (NJ + 0), xrt::bo::flags::normal, bank_grp_kernel_3mm_B);
  xrt::bo data_buffer_kernel_3mm_F = xrt::bo(device, sizeof(double) * (NJ + 0) * (NL + 0), xrt::bo::flags::normal, bank_grp_kernel_3mm_F);
  xrt::bo data_buffer_kernel_3mm_C = xrt::bo(device, sizeof(double) * (NJ + 0) * (NM + 0), xrt::bo::flags::normal, bank_grp_kernel_3mm_C);
  xrt::bo data_buffer_kernel_3mm_D = xrt::bo(device, sizeof(double) * (NM + 0) * (NL + 0), xrt::bo::flags::normal, bank_grp_kernel_3mm_D);
  xrt::bo data_buffer_kernel_3mm_G = xrt::bo(device, sizeof(double) * (NI + 0) * (NL + 0), xrt::bo::flags::normal, bank_grp_kernel_3mm_G);

  // create kernel runner
  xrt::run run_kernel_3mm(kernel_3mm_kernel);

  // 1 warm up iteration
  std::cout << "Running 1 warm up iteration ..." << std::endl;
  // write data to buffer objects
  data_buffer_kernel_3mm_E.write(E);
  data_buffer_kernel_3mm_E.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_3mm_A.write(A);
  data_buffer_kernel_3mm_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_3mm_B.write(B);
  data_buffer_kernel_3mm_B.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_3mm_F.write(F);
  data_buffer_kernel_3mm_F.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_3mm_C.write(C);
  data_buffer_kernel_3mm_C.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_3mm_D.write(D);
  data_buffer_kernel_3mm_D.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  data_buffer_kernel_3mm_G.write(G);
  data_buffer_kernel_3mm_G.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // set arguments of kernel_3mm
  run_kernel_3mm.set_arg(kernel_3mm_ptr_E, data_buffer_kernel_3mm_E);
  run_kernel_3mm.set_arg(kernel_3mm_ptr_A, data_buffer_kernel_3mm_A);
  run_kernel_3mm.set_arg(kernel_3mm_ptr_B, data_buffer_kernel_3mm_B);
  run_kernel_3mm.set_arg(kernel_3mm_ptr_F, data_buffer_kernel_3mm_F);
  run_kernel_3mm.set_arg(kernel_3mm_ptr_C, data_buffer_kernel_3mm_C);
  run_kernel_3mm.set_arg(kernel_3mm_ptr_D, data_buffer_kernel_3mm_D);
  run_kernel_3mm.set_arg(kernel_3mm_ptr_G, data_buffer_kernel_3mm_G);

  // run kernel_3mm
  // kernel_3mm(*E, *A, *B, *F, *C, *D, *G);
  run_kernel_3mm.start();
  run_kernel_3mm.wait();

  // Read back the results
  data_buffer_kernel_3mm_E.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_kernel_3mm_E.read(E);
  data_buffer_kernel_3mm_F.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_kernel_3mm_F.read(F);
  data_buffer_kernel_3mm_G.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  data_buffer_kernel_3mm_G.read(G);

  std::cout << "Done" << std::endl;

  double (*G_golden)[NI + 0][NL + 0];
  G_golden = ((double (*)[NI + 0][NL + 0])(polybench_alloc_data(((NI + 0) * (NL + 0)),(sizeof(double )))));
/*
  // check result
  std::cout << "Checking results ..." << std::endl;
  golden_kernel_3mm(*E, *A, *B, *F, *C, *D, *G_golden);
  golden_kernel_3mm(*E, *A, *B, *F, *C, *D, *G_golden);

  int error = 0;
  for (int i = 0; i < NI; i++) {
    for (int j = 0; j < NL; j++) {
      if (abs((*G)[i][j] - (*G_golden)[i][j]) > 0.1) {
        error += 1;
        // cout << "Mismatch at position " << i << " " << j << ": " << (*G)[i][j] << " " << (*G_golden)[i][j] << endl;
      }
    }
  }
  if (error) {
    cout << "Error count: " << error << endl;
  } else {
    // cout << "No errors detected" << endl;
  }

  std::cout << "Done" << std::endl;
*/

  // multi iterations
  int iterations = ITERATIONS;
  std::cout << "Running " << iterations << " iterations ..." << std::endl;

  double start_whole_time = omp_get_wtime();
  double start_iteration_time;
  double kernel_3mm_time = 0;

  for (int i = 0; i < iterations; i++) {
    start_iteration_time = omp_get_wtime();
    // write data to buffer objects
    data_buffer_kernel_3mm_E.write(E);
    data_buffer_kernel_3mm_E.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_3mm_A.write(A);
    data_buffer_kernel_3mm_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_3mm_B.write(B);
    data_buffer_kernel_3mm_B.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_3mm_F.write(F);
    data_buffer_kernel_3mm_F.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_3mm_C.write(C);
    data_buffer_kernel_3mm_C.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_3mm_D.write(D);
    data_buffer_kernel_3mm_D.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    data_buffer_kernel_3mm_G.write(G);
    data_buffer_kernel_3mm_G.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // set arguments of kernel_3mm
    run_kernel_3mm.set_arg(kernel_3mm_ptr_E, data_buffer_kernel_3mm_E);
    run_kernel_3mm.set_arg(kernel_3mm_ptr_A, data_buffer_kernel_3mm_A);
    run_kernel_3mm.set_arg(kernel_3mm_ptr_B, data_buffer_kernel_3mm_B);
    run_kernel_3mm.set_arg(kernel_3mm_ptr_F, data_buffer_kernel_3mm_F);
    run_kernel_3mm.set_arg(kernel_3mm_ptr_C, data_buffer_kernel_3mm_C);
    run_kernel_3mm.set_arg(kernel_3mm_ptr_D, data_buffer_kernel_3mm_D);
    run_kernel_3mm.set_arg(kernel_3mm_ptr_G, data_buffer_kernel_3mm_G);

    // run kernel_3mm
    // kernel_3mm(*E, *A, *B, *F, *C, *D, *G);
    run_kernel_3mm.start();
    run_kernel_3mm.wait();

    // Read back the results
    data_buffer_kernel_3mm_E.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_kernel_3mm_E.read(E);
    data_buffer_kernel_3mm_F.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_kernel_3mm_F.read(F);
    data_buffer_kernel_3mm_G.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    data_buffer_kernel_3mm_G.read(G);
    kernel_3mm_time += omp_get_wtime() - start_iteration_time;
  }
  std::cout << "Done" << std::endl;

  double run_whole_time = omp_get_wtime() - start_whole_time;
  cout << "1 warm up iteration and " << iterations << " iterations " << endl;
  cout << "Single iteration time: " << (run_whole_time / iterations) * 1000 << " ms" << endl;
  cout << "Kernel 3mm time: " << (kernel_3mm_time / iterations) * 1000 << " ms" << endl;

  free(((void *)E));
  free(((void *)A));
  free(((void *)B));
  free(((void *)F));
  free(((void *)C));
  free(((void *)D));
  free(((void *)G));
  free(((void *)G_golden));

  return 0;
}