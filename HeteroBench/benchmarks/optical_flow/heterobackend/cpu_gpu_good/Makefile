CPU_CXX := clang++
CPU_CXXFLAGS := -Wall -Wextra -pedantic -std=c++11 -fopenmp -w
CPU_LDFLAGS := -fopenmp

GPU_CXX := nvc++
GPU_CXXFLAGS := -Wall -Wextra
GPU_LDFLAGS := 

OPENMP_OFFLOAD_LIBS := -mp=gpu

# Include imageLib directory
IMAGELIB_DIR := ../../imageLib
CXXFLAGS += -I$(IMAGELIB_DIR)

# input path
INPUT_PATH := /home/hongzhet/myHome/cova_benchmarks/HeteroBench/benchmarks/optical_flow/datasets/larger

# output path
OUTPUT_PATH := /home/hongzhet/myHome/cova_benchmarks/HeteroBench/benchmarks/optical_flow/datasets/output/acc_output.flo

# iterations
ITERATIONS := 20

# max height
MAX_HEIGHT := 2160

# max width
MAX_WIDTH := 3840

# Project name
TARGET_EXEC := opf_sw

# Source files
SOURCES := main.cpp \
    $(IMAGELIB_DIR)/Convert.cpp \
	$(IMAGELIB_DIR)/Convolve.cpp \
	$(IMAGELIB_DIR)/flowIO.cpp \
    $(IMAGELIB_DIR)/Image.cpp \
	$(IMAGELIB_DIR)/ImageIO.cpp \
	$(IMAGELIB_DIR)/RefCntMem.cpp \
	cpu_impl/gradient_z_calc.cpp \
	cpu_impl/gradient_weight_x.cpp \
	cpu_impl/outer_product.cpp \
	cpu_impl/flow_calc.cpp \
	cpu_impl/gradient_weight_y.cpp \
	cpu_impl/tensor_weight_y.cpp \
	cpu_impl/tensor_weight_x.cpp

# MACRO DEFINITION
MACRO := -DITERATIONS=$(ITERATIONS) -DMAX_HEIGHT=$(MAX_HEIGHT) -DMAX_WIDTH=$(MAX_WIDTH)

# GPU Source files
GPU_SOURCES := 	gpu_impl/gradient_xy_calc.cpp 

# Object files
OBJECTS := $(SOURCES:.cpp=.o)

GPU_OBJECTS := $(GPU_SOURCES:.cpp=.o)

# Include directories
INCLUDE_DIRS := cpu_impl/include gpu_impl/include $(IMAGELIB_DIR)
INCLUDE_PARAMS := $(addprefix -I, $(INCLUDE_DIRS))

# Link OpenMP target offload library
OPENMP_OFFLOAD_LIB := 

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
	./$(TARGET_EXEC) -p $(INPUT_PATH) -o $(OUTPUT_PATH)

# Clean up
clean:
	rm -f $(TARGET_EXEC) $(OBJECTS) $(GPU_OBJECTS)

.PHONY: all clean run help

help:
	@echo "Usage: make [all|clean|run|help]"
	@echo "  all:        Build the project"
	@echo "  clean:      Clean up the project"
	@echo "  run:        Run the project"
	@echo "  help:       Show this help message"