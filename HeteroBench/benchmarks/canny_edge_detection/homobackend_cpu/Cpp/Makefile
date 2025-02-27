CXX := clang++
CXXFLAGS := -Wall -Wextra -pedantic -std=c++11 -fopenmp -w
LDFLAGS := -fopenmp
OPENMP_LIBS := -lgomp

# input path
INPUT_PATH := 

# output path
OUTPUT_PATH := 

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

# CPU Source files
CPU_SOURCES := cpu_impl/gaussian_filter.cpp cpu_impl/gradient_intensity_direction.cpp cpu_impl/edge_thinning.cpp cpu_impl/double_thresholding.cpp cpu_impl/hysteresis.cpp

# Object files
OBJECTS := $(SOURCES:.cpp=.o) 

CPU_OBJECTS := $(CPU_SOURCES:.cpp=.o)

# Include directories
INCLUDE_DIRS := cpu_impl/include .
INCLUDE_PARAMS := $(addprefix -I, $(INCLUDE_DIRS))

# Default target
all: $(TARGET_EXEC)

$(TARGET_EXEC): $(OBJECTS) $(CPU_OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(OPENMP_LIBS)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

# To compile other CPU source files
cpu_impl/%.o: cpu_impl/%.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

# Run
run:
	./$(TARGET_EXEC) $(INPUT_PATH) $(OUTPUT_PATH) $(LOW_THRESHOLD) $(HIGH_THRESHOLD)

# Clean up
clean:
	rm -f $(TARGET_EXEC) $(OBJECTS) $(CPU_OBJECTS)

.PHONY: all clean run help

help:
	@echo "Usage: make [all|clean|run|help]"
	@echo "  all:        Build the project"
	@echo "  clean:      Clean up the project"
	@echo "  run:        Run the project"
	@echo "  help:       Show this help message"