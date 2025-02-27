CXX := clang++
CXXFLAGS := -Wall -Wextra -pedantic -std=c++11 -fopenmp -w
LDFLAGS := -fopenmp
OPENMP_LIBS := -lgomp

# input path
INPUT_PATH := 

# output path
OUTPUT_PATH := 

# iterations
ITERATIONS := 1

# Project name
TARGET_EXEC := sbf_sw

# Source files
SOURCES := main.cpp

# MACRO DEFINITION
MACRO := -DITERATIONS=$(ITERATIONS)

# CPU Source files
CPU_SOURCES := cpu_impl/compute_gradient_magnitude.cpp cpu_impl/sobel_filter_x.cpp cpu_impl/sobel_filter_y.cpp

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
	./$(TARGET_EXEC) $(INPUT_PATH) $(OUTPUT_PATH)

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