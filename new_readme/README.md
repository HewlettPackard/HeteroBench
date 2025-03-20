# HeteroBench: Multi-kernel Benchmarks for Heterogeneous Systems

## Overview
HeteroBench is a comprehensive benchmark suite designed to evaluate heterogeneous systems with various accelerators, including CPUs, GPUs (NVIDIA, AMD, Intel), and FPGAs (AMD). It features multi-kernel applications spanning domains like image processing, machine learning, numerical computation, and physical simulation. HeteroBench aims to assist users in assessing performance, optimizing hardware usage, and facilitating decision-making in HPC environments.

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

> **Note**: Abbreviations will be used throughout the documentation. For detailed descriptions of each benchmark, refer to the corresponding part in the [paper](https://doi.org/10.1145/3676151.3719366).

## Directory Structure

At the top level, you will find the following files and directories:

- **LICENSE.md**: Open source license information for the repository.
- **README.md**: This file, providing an overview and instructions.
- **[Docker_Setup.md](Docker_Setup.md)**: Information on setting up Docker for running applications on NVIDIA GPUs.
- **[Source_Compilation.md](Source_Compilation.md)**: Instructions for compiling and running HeteroBench from source code.
- **Dockerfile**: Docker file used to create the container.
- **run.sh**: A single-point run file for building and running all scripts.
- **HeteroBench/**: The main directory containing all benchmark codes:
  - **heterobench.py**: The main Python script for managing and executing all benchmarks.
  - **benchmarks/**: Contains 11 subdirectories, one for each benchmark. Each includes Python and C++ implementations along with Makefiles.
  - **config_json/**: Contains configuration files specifying system environment settings and benchmark parameters.
  - **logs/**: Generated after running benchmarks, contains log files and terminal outputs showing time consumption profiling results.

## Quick Start Guide

### Option 1: Using Docker (Recommended)

Docker provides the simplest way to run HeteroBench without manual environment setup. This is especially useful for reproducibility and testing.

```bash
# Clone the repository
git clone git@github.com:HewlettPackard/HeteroBench.git
cd HeteroBench

# Run the provided script (builds and runs benchmarks in Docker)
./run.sh
```

By default, the script builds and runs benchmarks for CPUs and NVIDIA GPUs using OpenMP, OpenACC, and CUDA. For detailed Docker setup instructions, see [Docker_Setup.md](Docker_Setup.md).

### Option 2: Compile from Source

For users who prefer to compile from source or need to modify the benchmarks, follow the instructions in [Source_Compilation.md](Source_Compilation.md).

## Execution Time Reference

The table below shows the average build and execution time for each scenario on an *AMD EPYC 7713 64-Core processor* and an *NVIDIA A100 GPU*:

| **Scenarios**        | **Compile Time (s)** | **Run Time (s)**   |
|----------------------|----------------------|--------------------|
| Python               | N/A                  | Over 10 hrs        |
| Numba                | N/A                  | Over 3 hrs         |
| CPU                  | 100                  | 74                 |
| GPU OpenMP           | 114                  | 57                 |
| GPU OpenACC          | 118                  | 44                 |
| GPU CUDA             | 125                  | 36                 |
| FPGA                 | Over 24 hrs          | Over 3 hrs         |

Note: Python and Numba scenarios are commented out by default in the `run.sh` file due to their long execution times. To include them, uncomment lines `#37` and `#38` in the run.sh file, or just run:

```bash
sed -i '37,38 s/^# //' run.sh
```

## Additional Notes
### Adding New GPU Architectures

HeteroBench supports adding new GPU architectures. Here are two methods to add a new architecture:

#### Method 1: Update proj_config.json File

Edit the file `config_json/proj_config.json` to update the GPU architecture (`gpu_arch`) for each benchmark:

```bash
sed -i 's/"gpu_arch": ""/"gpu_arch": "YOUR_ARCHITECTURE"/g' HeteroBench/config_json/proj_config.json
```

Replace `YOUR_ARCHITECTURE` with your GPU architecture identifier (e.g., `sm_80` for NVIDIA RTX A16).

#### Method 2: Update heterobench.py

The `heterobench.py` file contains a mapping dictionary `gpu_arch_map` of GPU models to their architecture codes. You can add your GPU model to it to ensure automatic architecture detection.

### Running Without a GPU

HeteroBench can be run on systems without a GPU by using only CPU implementations:

1. To build and run C++ benchmarks on a CPU only, modify the Dockerfile:
   ```bash
   sed -i '69,71 {s/^/# /}' Dockerfile
   echo "" >> Dockerfile
   echo "RUN python3 heterobench.py all build cpu serial" >> Dockerfile
   ```
   This will comment out the GPU related build, and add a CPU serial build in addition.

2. Update run.sh to run CPU benchmarks:
   ```bash
   sed -i '40,42 {s/^/# /}' run.sh
   echo "" >> run.sh
   echo "run cpu serial" >> run.sh
   ```

3. For Docker execution on systems without a GPU, remove the `--gpus all` parameter:
   ```bash
   sed -i '31 {s/--gpus all -it/-it/g}' run.sh
   ```

### Running on an FPGA

Users can follow the steps in [Source_Compilation.md](Source_Compilation.md) to run benchmarks on an FPGA. We use AMD Alveo U280 by default, specifically the `xilinx_u280_gen3x16_xdma_1_202211_1` platform. If you need to use a different FPGA, please update the `"fpga_board"` field of each benchmark in `config_json/proj_config.json` to your specific platform version. You can do this by running:

```bash
sed -i 's/"fpga_board": "xilinx_u280_gen3x16_xdma_1_202211_1"/"fpga_board": "YOUR_FPGA_PLATFORM"/g' HeteroBench/config_json/proj_config.json
```

**Additionally**, it's important to note that if your FPGA board doesn't have **HBM** (High-bandwidth memory), you'll need to comment out the HBM-related lines in the `xclbin_overlay.cfg` file located in each benchmark's `fpga_impl` directory. You can do this by running the following command:

```bash
find HeteroBench/benchmarks -name "xclbin_overlay.cfg" -exec sed -i 's/.*HBM.*/#&/' {} \;
```

This command will search for all `xclbin_overlay.cfg` files in the benchmarks directory and remove any lines containing "HBM".

## Profiling Results

After running benchmarks, time information for each kernel will be printed in the terminal and saved to log files in the `logs/` directory. These logs provide detailed performance metrics for analysis.

## Contributing

We welcome contributions to HeteroBench! If you'd like to add new benchmarks, support for new hardware, or improve existing code, please submit a pull request or open an issue.

## Support and Contact

If you have any questions or need assistance with HeteroBench, feel free to contact:

- Alok Mishra: alok.mishra@hpe.com
- Hongzheng Tian: hongzhet@uci.edu, or hongzheng.tian@hpe.com
