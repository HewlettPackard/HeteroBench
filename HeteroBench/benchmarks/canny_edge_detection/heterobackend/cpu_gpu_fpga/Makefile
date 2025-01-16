CPU_CXX := clang++
CPU_CXXFLAGS := -Wall -Wextra -pedantic -std=c++17 -fopenmp -Wno-deprecated-declarations
CPU_LDFLAGS := -fopenmp

GPU_CXX := nvc++
GPU_CXXFLAGS := -Wall -Wextra
GPU_LDFLAGS := 

OPENMP_OFFLOAD_LIBS := -mp=gpu

# Xilinx XRT and HLS paths
XILINX_XRT := ${xilinx_xrt}
XILINX_HLS := ${xilinx_hls}
CPU_CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_HLS)/include
CPU_LDFLAGS += -L$(XILINX_XRT)/lib -lxrt_coreutil -pthread
GPU_LDFLAGS += -L$(XILINX_XRT)/lib -lxrt_coreutil -pthread

# Add FPGA specific includes and libraries
FPGA_INCLUDE := $(XILINX_HLS)/include
CPU_CXXFLAGS += -I$(FPGA_INCLUDE)

# input path
INPUT_PATH := ${input}

# output path
OUTPUT_PATH := ${output}

# low threshold
LOW_THRESHOLD := ${low_threshold}

# high threshold
HIGH_THRESHOLD := ${high_threshold}

# iterations
ITERATIONS := ${iteration_times}

# Project name
TARGET_EXEC := ${target}_sw

# Source files
SOURCES := \
	main.cpp \
	cpu_impl/hysteresis.cpp

# MACRO DEFINITION
MACRO := -DITERATIONS=$(ITERATIONS)

# GPU Source files
GPU_SOURCES := \
	gpu_impl/gaussian_filter.cpp \
	gpu_impl/gradient_indensity_direction.cpp 

# FPGA Source files
FPGA_SOURCES := \
	fpga_impl/edge_thinning.cpp \
	fpga_impl/double_thresholding.cpp

# Object files
OBJECTS := $(SOURCES:.cpp=.o)
GPU_OBJECTS := $(GPU_SOURCES:.cpp=.o)

# Include directories
INCLUDE_DIRS := cpu_impl/include gpu_impl/include fpga_impl/include .
INCLUDE_PARAMS := $(addprefix -I, $(INCLUDE_DIRS))

# Link OpenMP target offload library
OPENMP_OFFLOAD_LIB := ${lomptarget}

# Default target
all: $(TARGET_EXEC)

$(TARGET_EXEC): $(OBJECTS) $(GPU_OBJECTS)
	$(GPU_CXX) $(GPU_LDFLAGS) $(OPENMP_OFFLOAD_LIBS) -o $@ $^ $(OPENMP_OFFLOAD_LIB) 

main.o: main.cpp
	$(CPU_CXX) $(CPU_CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

# To compile other CPU source files
cpu_impl/%.o: cpu_impl/%.cpp
	$(CPU_CXX) $(CPU_CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

# To compile GPU source file
gpu_impl/%.o: gpu_impl/%.cpp
	$(GPU_CXX) $(GPU_CXXFLAGS) $(MACRO) $(OPENMP_OFFLOAD_LIBS) $(INCLUDE_PARAMS)  -c $< -o $@

# Run
run:
# rocprof --timestamp on ./$(TARGET_EXEC) $(INPUT_PATH) $(OUTPUT_PATH) $(LOW_THRESHOLD) $(HIGH_THRESHOLD)
	${profiling_tool} ./$(TARGET_EXEC) $(INPUT_PATH) $(OUTPUT_PATH) $(LOW_THRESHOLD) $(HIGH_THRESHOLD)

# FPGA Compilation

TARGET := hw
XOCCFLAGS := --platform ${fpga_board} -t $(TARGET) -s -g

FPGA_OBJECTS := $(FPGA_SOURCES:fpga_impl/%.cpp=fpga_impl/%_$(TARGET).xo)

fpga_compile: $(FPGA_OBJECTS)

fpga_impl/%_$(TARGET).xo: fpga_impl/%.cpp
	v++ $(XOCCFLAGS) $(INCLUDE_PARAMS) $(MACRO) --kernel $* -c -o $@ $<

# FPGA Linking
XOCCLFLAGS := --kernel_frequency $(kernel_frequency)

XOS := $(FPGA_OBJECTS)

XSA_OBJ := overlay_$(TARGET).xsa
XCLBIN  := overlay_$(TARGET).xclbin

fpga_link: $(XCLBIN)

ifeq ($(PLATFORM), xilinx_vck5000_gen4x8_xdma_2_202210_1)

$(XCLBIN): $(XSA_OBJ)
	v++ -p -t $(TARGET) -f $(PLATFORM) $^ -o $@ --package.boot_mode=ospi

$(XSA_OBJ): $(XOS)
	v++ -l $(XOCCFLAGS) $(XOCCLFLAGS) --config ./fpga_impl/xclbin_overlay.cfg -o $@ $^

else ifeq ($(PLATFORM), xilinx_u280_gen3x16_xdma_1_202211_1)

$(XCLBIN): $(XOS)
	v++ -l $(XOCCFLAGS) $(XOCCLFLAGS) --config ./fpga_impl/xclbin_overlay.cfg -o $@ $^

else ifeq ($(PLATFORM), xilinx_u250_gen3x16_xdma_4_1_202210_1)

$(XCLBIN): $(XOS)
	v++ -l $(XOCCFLAGS) $(XOCCLFLAGS) --config ./fpga_impl/xclbin_overlay.cfg -o $@ $^

endif

# FPGA all
fpga_all: fpga_compile fpga_link

# Clean up
clean_all:
	rm -f $(TARGET_EXEC) $(OBJECTS) $(GPU_OBJECTS) $(FPGA_SOURCES:.cpp=.o) $(XOS) $(XSA_OBJ) $(XCLBIN)
	$(RM) -rf *.xo *.xclbin *.ltx *.xclbin.package_summary *.xsa *.xsa.link_summary *.backup.log *.jou *.log
	$(RM) -rf ./fpga_impl/*.xo ./fpga_impl/*.xo.compile_summary
	$(RM) -r .Xil _x .ipcache

clean_fpga:
	rm -f $(XOS) $(XSA_OBJ) $(XCLBIN)
	$(RM) -rf *.xo *.xclbin *.ltx *.xclbin.package_summary *.xsa *.xsa.link_summary *.backup.log *.jou *.log
	$(RM) -r *.Xil _x .ipcache

clean_summaries:
	$(RM) -rf *.xclbin.package_summary *.xsa.link_summary
	$(RM) -rf ./fpga_impl/*.xo.compile_summary

clean:
	rm -f $(TARGET_EXEC) $(OBJECTS) $(GPU_OBJECTS)

.PHONY: all clean run help fpga_compile fpga_link fpga_all clean_all clean_fpga clean_summaries

help:
	@echo "Usage: make [all|clean|run|help]"
	@echo "  all:        Build the project"
	@echo "  clean:      Clean up the project"
	@echo "  run:        Run the project"
	@echo "  help:       Show this help message"
	@echo "  fpga_compile: Compile FPGA source files"
	@echo "  fpga_link: Link FPGA hardware binaries"
	@echo "  fpga_all: Compile and link FPGA source files"
	@echo "  clean_all: Clean up all files, including FPGA files, which contains FPGA binaries and they need long time to compile and link"
	@echo "  clean_fpga: Clean up FPGA files, which contains FPGA binaries and they need long time to compile and link"
	@echo "  clean_summaries: Clean up summaries"
