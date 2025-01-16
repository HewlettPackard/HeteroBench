CXX := clang++
CXXFLAGS := -Wall -Wextra -pedantic -std=c++11 -fopenmp -w
LDFLAGS := -fopenmp
OPENMP_LIBS := -lgomp

# Include imageLib directory
IMAGELIB_DIR := ../../imageLib
CXXFLAGS += -I$(IMAGELIB_DIR)

# input path
INPUT_PATH := /home/hongzhet/myProjects/cova_benchmarks/HeteroBench/benchmarks/optical_flow/datasets/larger

# output path
OUTPUT_PATH := /home/hongzhet/myProjects/cova_benchmarks/HeteroBench/benchmarks/optical_flow/datasets/output/acc_output.flo

# iterations
ITERATIONS := 1

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
	$(IMAGELIB_DIR)/RefCntMem.cpp

# MACRO DEFINITION
MACRO := -DITERATIONS=$(ITERATIONS) -DMAX_HEIGHT=$(MAX_HEIGHT) -DMAX_WIDTH=$(MAX_WIDTH)

# CPU Source files
CPU_SOURCES := cpu_impl/gradient_xy_calc.cpp cpu_impl/gradient_z_calc.cpp cpu_impl/gradient_weight_y.cpp cpu_impl/gradient_weight_x.cpp cpu_impl/outer_product.cpp cpu_impl/tensor_weight_y.cpp cpu_impl/tensor_weight_x.cpp cpu_impl/flow_calc.cpp

# Object files
OBJECTS := $(SOURCES:.cpp=.o) 

CPU_OBJECTS := $(CPU_SOURCES:.cpp=.o)

# Include directories
INCLUDE_DIRS := cpu_impl/include $(IMAGELIB_DIR)
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
	./$(TARGET_EXEC) -p $(INPUT_PATH) -o $(OUTPUT_PATH)

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