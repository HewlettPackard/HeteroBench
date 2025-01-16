CXX := nvc++
NVCC := nvcc
CXXFLAGS := -Wall -Wextra -w
LDFLAGS := -cuda

# input path
INPUT_PATH := /home/mishraal/git/cova_benchmarks/new/HeteroBench/benchmarks/sobel_filter/input/1439x1919_stanford.jpg

# output path
OUTPUT_PATH := /home/mishraal/git/cova_benchmarks/new/HeteroBench/benchmarks/sobel_filter/output/1439x1919_stanford.jpg

# iterations
ITERATIONS := 1

# Project name
TARGET_EXEC := sbf_sw

# Source files
SOURCES := main.cpp

# MACRO DEFINITION
MACRO := -DITERATIONS=$(ITERATIONS)

# GPU Source files
GPU_SOURCES := cuda_impl/compute_gradient_magnitude.cu cuda_impl/sobel_filter_x.cu cuda_impl/sobel_filter_y.cu

# Object files
OBJECTS := $(SOURCES:.cpp=.o) 

GPU_OBJECTS := $(GPU_SOURCES:.cu=.o)

# Include directories
INCLUDE_DIRS := cuda_impl/include .
INCLUDE_PARAMS := $(addprefix -I, $(INCLUDE_DIRS))

# Default target
all: $(TARGET_EXEC)

$(TARGET_EXEC): $(OBJECTS) $(GPU_OBJECTS)
	$(CXX)  $(LDFLAGS) -o $@ $^

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

# To compile GPU source file
cuda_impl/%.o: cuda_impl/%.cu
	$(NVCC) -lineinfo  $(INCLUDE_PARAMS) -c $< -o $@

# Run
run:
	./$(TARGET_EXEC) $(INPUT_PATH) $(OUTPUT_PATH)

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