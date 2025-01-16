# HeteroBench: Multi-kernel Benchmarks for Heterogeneous Systems

## Overview
HeteroBench is a comprehensive benchmark suite designed to evaluate heterogeneous systems with various accelerators, including CPUs, GPUs (NVIDIA, AMD, Intel), and FPGAs (Xilinx). It features multi-kernel applications spanning domains like image processing, machine learning, numerical computation, and physical simulation. HeteroBench aims to assist users in assessing performance, optimizing hardware usage, and facilitating decision-making in HPC environments.

## Key Features
- **Broad Hardware Compatibility**: Supports CPUs, GPUs from multiple brands, and FPGAs.
- **Multi-language Support**: Benchmarks available in Python, Numba-accelerated Python, and various C++ versions (serial, OpenMP, OpenACC, CUDA, and Vitis HLS).
- **Customizable Kernel Placement**: Enables kernel-level optimization for different hardware backends by users.
- **Fair Comparisons**: Standardized algorithms across platforms for unbiased benchmarking.
- **User-friendly Design**: Simplified setup with configuration files and a top-level Python script for project management.

## Included Benchmarks
The suite currently includes benchmarks across four domains:

<table>
  <thead>
    <tr>
      <th><strong>Benchmarks (Abbreviation)</strong></th>
      <th style="text-align: center;"><strong># of Compute Kernels</strong></th>
      <th><strong>Application Domain</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Canny Edge Detection (ced)</td>
      <td style="text-align: center;">5</td>
      <td rowspan="3">Image Processing</td>
    </tr>
    <tr>
      <td>Sobel Filter (sbf)</td>
      <td style="text-align: center;">3</td>
    </tr>
    <tr>
      <td>Optical Flow (opf)</td>
      <td style="text-align: center;">8</td>
    </tr>
    <tr>
      <td>Convolutional Neural Network (cnn)</td>
      <td style="text-align: center;">5</td>
      <td rowspan="4">Machine Learning</td>
    </tr>
    <tr>
      <td>Multilayer Perceptron (mlp)</td>
      <td style="text-align: center;">3</td>
    </tr>
    <tr>
      <td>Digit Recognition (dgr)</td>
      <td style="text-align: center;">2</td>
    </tr>
    <tr>
      <td>Spam Filter (spf)</td>
      <td style="text-align: center;">4</td>
    </tr>
    <tr>
      <td>3 Matrix Multiplication (3mm)</td>
      <td style="text-align: center;">2</td>
      <td rowspan="2">Numerical Computation</td>
    </tr>
    <tr>
      <td>Alternating Direction Implicit (adi)</td>
      <td style="text-align: center;">2</td>
    </tr>
    <tr>
      <td>Parallelize Particle (ppc)</td>
      <td style="text-align: center;">2</td>
      <td>Physical Simulation</td>
    </tr>
  </tbody>
</table>


> **Note**: Abbreviations will be used throughout the documentation. For detailed descriptions of each benchmark, refer to the `readme.md` under each subdirectory and the corresponding part in the paper.


## Get Started
### Environment Setup
Make some neccessary changes according to your system directory. 

* Set the path to clang and openmp:

  ```sh
  export LLVM_INSTALL_PATH=/path/to/llvm-project/llvm-install
  export PATH=$LLVM_INSTALL_PATH/llvm/bin:$PATH
  export LD_LIBRARY_PATH=$LLVM_INSTALL_PATH/llvm/lib:$LD_LIBRARY_PATH
  ```

* If using NVIDIA GPU:

  ```sh
  export NVHPC=/opt/nvidia/hpc_sdk
  export PATH=$NVHPC/Linux_x86_64/24.5/compilers/bin:$PATH
  export MANPATH=$NVHPC/Linux_x86_64/24.5/compilers/man:$MANPATH
  export LD_LIBRARY_PATH=$NVHPC/Linux_x86_64/24.5/compilers/lib:$LD_LIBRARY_PATH
  ```

* If using AMD GPU:

  ```sh
  export ROCM_PATH=/opt/rocm-6.1.1
  export PATH=$ROCM_PATH/llvm/bin:$PATH
  export LD_LIBRARY_PATH=$ROCM_PATH/llvm/lib:$LD_LIBRARY_PATH
  ```

* If using Intel GPU:

  ```sh
  source /opt/intel/oneapi/setvars.sh
  ```

* Change the environment settings under `./config_json/env_config.json`. Usually, the defualt parameters work for most of the systems. The main changes could be:
  * Set the path to the Xilinx libraries by changing `xilinx_xrt` and `xilinx_hls`.
  * If you'd like to use the profiler, set the `profiling_tool`, for example:
    * For Nvidia GPU: 
      * `ncu --set full`.
      * `ncu  --metrics gpu__time_duration.sum --target-processes all`.
      * `nsys profile -t cuda,nvtx  --stats=true --force-overwrite true -o profiling_report`.
    * For AMD GPU: `rocprof --timestamp on`.
    * For Intel GPU: `vtune  -collect gpu-offload -knob collect-programming-api=true --`.

### Benchmark Setup

You can check and change the benchmark setting under `./config_json/proj_config.json`. The main changes could be:
  
* Set the input and output paths to run customized inputs.
* Set the iteration times of each kernel.
* Set the brand of GPU by changing `gpu_brand`. Choose from `nvidia`, `amd`, and `intel`. HeteroBench will use different commands to build the project accordingly.
* Set the path to the board files of your target PCIe FPGA board by changing `fpga_board`.
> Note: each benchmark will run 1 warm-up test intuitively and then run 1 iteration by default. Users can increase the iteration times, then the benchmark will run multiple times of tests, and then get the average time consumption.

### Instructions

Usage:

```sh
python heterobench.py <benchmark> <action> <backend> <options>
```

`benchmark`: the name of benchmarks, you can use both the full name and the abbreviation as shown in the previous table.

  * For example, for Canny Edge Detection, you can use `canny_edge_detection` or `ced`.

`action`: the actions to be applied on the benchmark

  * `build`: to build the project.
  * `run`: to run or execute the project.
  * `clean`: to clean the compiled output and execution files of CPU and GPU.
  * `clean_fpga`: to clean all the FPGA related compiled output files and directories.
  * `clean_all`: to clean everything.

`backend`: the backend to be targeted

  * `python`: to run the general Python version of the benchmark.
  * `numba`: to run the Python version accelerated by Numba of the benchmark.
  * `cpu`: to run the C++ code on CPU.
  * `gpu_omp`: to run the OpenMP C++ code on GPU.
  * `gpu_acc`: to run the OpenACC C++ code on Nvidia GPU.
  * `gpu_cuda`: to run the CUDA code on Nvidia GPU.
  * `fpga`: to run the C++ code on FPGA.
  * `hetero`: to run the heterogeneous version on both GPU and FPGA. (only for the `ced` benchmark yet)

`options`: the optional choices to be applied on the benchmark

  * `parallel`: the **defult** option, add parallel pragmas to the code to use OpenMP, OpenACC, CUDA or HLS.
  * `serial`: when running all the kernels on CPU, you can choose this option to only run the benchmark via general Cpp implementation in serial without using OpenMP or HLS.
  * `fpga_compile`: to run the FPGA synthesis of kernels.
  * `fpga_link`: to run the FPGA implementation of kernels.
  * `fpga_all`: to run `fpga_compile` and `fpga_link`.

Some example commands:

`python heterobench.py ced run python`

`python heterobench.py sobel_filter build cpu`

`python heterobench.py cnn build fpga fpga_all`

When turn on `serial`:

`python heterobench.py 3mm build cpu serial`

`python heterobench.py 3mm run cpu serial`

`python heterobench.py 3mm clean cpu serial`

### Profiling results

The time informations of kernels will be printed in the ternmial, as well as a profiling `.log` file in the `logs/` directory.