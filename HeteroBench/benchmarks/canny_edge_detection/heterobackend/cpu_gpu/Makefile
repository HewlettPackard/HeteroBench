CPU_CXX := clang++
CPU_CXXFLAGS := -Wall -Wextra -pedantic -std=c++11 -fopenmp
CPU_LDFLAGS := -fopenmp

GPU_CXX := nvc++
GPU_CXXFLAGS := -Wall -Wextra
GPU_LDFLAGS := 

OPENMP_OFFLOAD_LIBS := -mp=gpu

# input path
INPUT_PATH := ../../input/1439x1919_stanford.jpg 

# output path
OUTPUT_PATH := ../../output/1439x1919_stanford.jpg 

# low threshold
LOW_THRESHOLD := 30

# high threshold
HIGH_THRESHOLD := 90

# iterations
ITERATIONS := 20

# Project name
TARGET_EXEC := ced_sw

# Source files
SOURCES := \
	main.cpp \
	cpu_impl/edge_thinning.cpp \
	cpu_impl/double_thresholding.cpp \
	cpu_impl/gradient_intensity_direction.cpp

# GPU Source files
GPU_SOURCES := \
	gpu_impl/gaussian_filter.cpp \
	gpu_impl/hysteresis.cpp

# MACRO DEFINITION
MACRO := -DITERATIONS=$(ITERATIONS)

# Object files
OBJECTS := $(SOURCES:.cpp=.o)
GPU_OBJECTS := $(GPU_SOURCES:.cpp=.o)

# Include directories
INCLUDE_DIRS := cpu_impl/include gpu_impl/include .
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
# rocprof --timestamp on ./$(TARGET_EXEC) $(INPUT_PATH) $(OUTPUT_PATH) $(LOW_THRESHOLD) $(HIGH_THRESHOLD)
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