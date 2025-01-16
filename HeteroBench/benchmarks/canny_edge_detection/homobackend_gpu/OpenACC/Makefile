CXX := nvc++
CXXFLAGS := -Wall -Wextra -w
LDFLAGS := -acc

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
GPU_SOURCES := acc_impl/gaussian_filter.cpp acc_impl/gradient_intensity_direction.cpp acc_impl/edge_thinning.cpp acc_impl/double_thresholding.cpp acc_impl/hysteresis.cpp

# Object files
OBJECTS := $(SOURCES:.cpp=.o)

GPU_OBJECTS := $(GPU_SOURCES:.cpp=.o)

# Include directories
INCLUDE_DIRS := acc_impl/include .
INCLUDE_PARAMS := $(addprefix -I, $(INCLUDE_DIRS))

# Default target
all: $(TARGET_EXEC)

$(TARGET_EXEC): $(OBJECTS) $(GPU_OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

# To compile GPU source file
acc_impl/%.o: acc_impl/%.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDE_PARAMS)  -c $< -o $@

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