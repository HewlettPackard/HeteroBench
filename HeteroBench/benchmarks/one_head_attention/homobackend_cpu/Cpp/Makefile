CXX := clang++
CXXFLAGS := -Wall -Wextra -pedantic -std=c++11 -fopenmp -w
LDFLAGS := -fopenmp
OPENMP_LIBS := -lgomp

# input path
INPUT_PATH := /home/hongzhet/myProjects/cova_benchmarks/HeteroBench/benchmarks/one_head_attention/input

# output path
OUTPUT_PATH := /home/hongzhet/myProjects/cova_benchmarks/HeteroBench/benchmarks/one_head_attention/output

# iteration_times
ITERATIONS := 1
# batch_size
BATCH_SIZE := 8
# N
N := 1024
# d
D := 128

# Project name
TARGET_EXEC := oha_sw

# Source files
SOURCES := main.cpp

# MACRO DEFINITION
MACRO := -DITERATIONS=$(ITERATIONS) \
	-DBATCH_SIZE=$(BATCH_SIZE) \
	-DN=$(N) \
	-DD=$(D)

# CPU Source files
CPU_SOURCES := cpu_impl/transpose.cpp cpu_impl/softmax.cpp cpu_impl/matmul.cpp

# Object files
OBJECTS := $(SOURCES:.cpp=.o)

CPU_OBJECTS := $(CPU_SOURCES:.cpp=.o)

INCLUDE_DIRS := cpu_impl/include
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