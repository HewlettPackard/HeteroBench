CXX := nvc++
CXXFLAGS := -Wall -Wextra -w
LDFLAGS := 
OPENMP_OFFLOAD_LIBS := -mp=gpu

# input path
INPUT_PATH := /home/mishraal/git/cova_benchmarks/new/HeteroBench/benchmarks/canny_edge_detection/input/1439x1919_stanford.jpg

# output path
OUTPUT_PATH := /home/mishraal/git/cova_benchmarks/new/HeteroBench/benchmarks/canny_edge_detection/output/1439x1919_stanford.jpg

# low threshold
LOW_THRESHOLD := 30

# high threshold
HIGH_THRESHOLD := 90

# iterations
ITERATIONS := 1

# Project name
TARGET_EXEC := ced_sw

# Source files
SOURCES := main.cpp

# MACRO DEFINITION
MACRO := -DITERATIONS=$(ITERATIONS)

# GPU Source files
GPU_SOURCES := gpu_impl/gaussian_filter.cpp gpu_impl/gradient_intensity_direction.cpp gpu_impl/edge_thinning.cpp gpu_impl/double_thresholding.cpp gpu_impl/hysteresis.cpp

# Object files
OBJECTS := $(SOURCES:.cpp=.o)

GPU_OBJECTS := $(GPU_SOURCES:.cpp=.o)

# Include directories
INCLUDE_DIRS := gpu_impl/include .
INCLUDE_PARAMS := $(addprefix -I, $(INCLUDE_DIRS))

# Link OpenMP target offload library
OPENMP_OFFLOAD_LIB := 

# Default target
all: $(TARGET_EXEC)

$(TARGET_EXEC): $(OBJECTS) $(GPU_OBJECTS)
	$(CXX) $(OPENMP_OFFLOAD_LIBS) $(LDFLAGS) -o $@ $^ $(OPENMP_OFFLOAD_LIB) 

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

# To compile GPU source file
gpu_impl/%.o: gpu_impl/%.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(OPENMP_OFFLOAD_LIBS) $(INCLUDE_PARAMS)  -c $< -o $@

# Run
PROFILING_TOOL := 
run:
	$(PROFILING_TOOL) ./$(TARGET_EXEC) $(INPUT_PATH) $(OUTPUT_PATH) $(LOW_THRESHOLD) $(HIGH_THRESHOLD)

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