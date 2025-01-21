## Directory Structure

At the top level, you will find the following files and directories:

- **README.md**: Instructions on building and running the artifact.
- **Docker Setup.md**: Information on setting up Docker for running applications on NVIDIA GPUs. (Other GPUs will need different setup).
- **Dockerfile**: Docker file used to create the container.
- **LICENSE.md**: Open Source License information for the repository.
- **run.sh**: A single-point run file for building and running all scripts.
- **HeteroBench**: This is the the directory for the repository which has all the benchmark codes. It contains the `heterobench.py` file, which is the main Python script for managing and executing all benchmarks. The repository is organized as follows:
  - **Benchmark Directory**: Contains 11 subdirectories for benchmarks and 1 GPU test directory. Each benchmark application is stored in its subdirectory, which includes Python and C++ implementations along with Makefiles. Each subdirectory contains different versions of implementations.
  - **Configuration Files Directory**: Contains two configuration files specifying system environment settings and benchmark parameters.
  - **Logs**: After running a benchmark, a `log` directory will be generated. This directory contains log files and terminal outputs showing time consumption profiling results.

---

## HeteroBench Docker Run

To execute the artifact, ensure that the Docker environment is properly set up. Follow the instructions provided in the `Docker Setup.md` file. Once the environment is configured, run the `run.sh` file. This script builds the Dockerfile to set up the environment within the Docker container.

On an NVIDIA GPU, *HeteroBench* supports six scenarios: Python, Numba, CPU, OpenMP, OpenACC, and CUDA. The table below shows the average build and execution time for each scenario on an *AMD EPYC 7713 64-Core processor* and an *NVIDIA A100 GPU*. Note that running the Python and Numba scenarios on a CPU can take over 10 hours and 3 hours, respectively. Therefore, these scenarios are commented out by default in the `run.sh` file. By default, the script runs benchmarks for CPUs and NVIDIA GPUs using OpenMP, OpenACC, and CUDA.

| **Scenarios**        | **Compile Time (s)** | **Run Time (s)**   |
|----------------------|----------------------|--------------------|
| Python               | N/A                  | Over 10 hrs        |
| Numba                | N/A                  | Over 3 hrs         |
| CPU                  | 100                  | 74                 |
| GPU OpenMP           | 114                  | 57                 |
| GPU OpenACC          | 118                  | 44                 |
| GPU CUDA             | 125                  | 36                 |
| FPGA                 | Over 24 hrs          | Over 3 hrs         |

The script generates output logs in the `logs` directory upon completion. To include the Python and Numba scenarios, uncomment lines `#37` and `#38` in the `run.sh` file.

### Additional Notes

- To benchmark other GPUs, such as AMD or Intel, a different Docker setup is required.
- For FPGA benchmarking, the system must be configured specifically for the target FPGA. However, the benchmark execution process remains consistent.
- For detailed instructions on running the benchmarks manually, refer to the `README.md` file in the `heterobench` directory.

If you have any questions, feel free to contact us.

Point of Contact:
- Alok Mishra : alok.mishra@hpe.com
- Hongzheng Tian : hongzheng.tian@hpe.com