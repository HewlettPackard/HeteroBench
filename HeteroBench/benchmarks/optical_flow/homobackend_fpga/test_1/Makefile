CXX := clang++
CXXFLAGS := -Wall -Wextra -pedantic -std=c++17 -fopenmp -Wno-deprecated-declarations -w
LDFLAGS := -fopenmp

# Xilinx XRT and HLS paths
XILINX_XRT := /opt/xilinx/xrt
XILINX_HLS := /tools/Xilinx/Vitis_HLS/2023.1/
CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_HLS)/include
LDFLAGS += -L$(XILINX_XRT)/lib -lxrt_coreutil -pthread

# Add FPGA specific includes and libraries
FPGA_INCLUDE := $(XILINX_HLS)/include
CXXFLAGS += -I$(FPGA_INCLUDE)

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

# FPGA Source files
FPGA_SOURCES := fpga_impl/optical_flow_hw.cpp fpga_impl/gradient_xy_calc.cpp fpga_impl/gradient_z_calc.cpp fpga_impl/gradient_weight_y.cpp fpga_impl/gradient_weight_x.cpp fpga_impl/outer_product.cpp fpga_impl/tensor_weight_y.cpp fpga_impl/tensor_weight_x.cpp fpga_impl/flow_calc.cpp

# Object files
OBJECTS := $(SOURCES:.cpp=.o)

# Include directories
INCLUDE_DIRS := fpga_impl/include $(IMAGELIB_DIR)
INCLUDE_PARAMS := $(addprefix -I, $(INCLUDE_DIRS))

# Link OpenMP target offload library
OPENMP_OFFLOAD_LIB := -lomptarget

# Default target
all: $(TARGET_EXEC)

$(TARGET_EXEC): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(OPENMP_OFFLOAD_LIB) 

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

# Run
run:
	./$(TARGET_EXEC) -p $(INPUT_PATH) -o $(OUTPUT_PATH)

# FPGA Compilation

TARGET := hw
FPGA_TOP := optical_flow_hw
XOCCFLAGS := --platform xilinx_u280_gen3x16_xdma_1_202211_1 -t $(TARGET) -s -g

# Since this benchmark has a top function, we don't need to compile all the other kernels separately.
FPGA_OBJECTS := fpga_impl/$(FPGA_TOP)_$(TARGET).xo

fpga_compile: $(FPGA_OBJECTS)

fpga_impl/$(FPGA_TOP)_$(TARGET).xo: $(FPGA_SOURCES)
	v++ $(XOCCFLAGS) $(INCLUDE_PARAMS)  $(MACRO) --kernel $(FPGA_TOP) -c -o $@ $^

# FPGA Linking
XOCCLFLAGS := --kernel_frequency $(kernel_frequency)

XOS := $(FPGA_OBJECTS)

XSA_OBJ := overlay_$(TARGET).xsa
XCLBIN  := overlay_$(TARGET).xclbin

fpga_link: $(XCLBIN)

ifeq (xilinx_u280_gen3x16_xdma_1_202211_1, xilinx_vck5000_gen4x8_xdma_2_202210_1)
$(XCLBIN): $(XSA_OBJ)
	v++ -p -t $(TARGET) -f xilinx_u280_gen3x16_xdma_1_202211_1 $^ -o $@ --package.boot_mode=ospi
$(XSA_OBJ): $(XOS)
	v++ -l $(XOCCFLAGS) $(XOCCLFLAGS) --config ./fpga_impl/xclbin_overlay.cfg -o $@ $^
else ifeq (xilinx_u280_gen3x16_xdma_1_202211_1, xilinx_u280_gen3x16_xdma_1_202211_1)
$(XCLBIN): $(XOS)
	v++ -l $(XOCCFLAGS) $(XOCCLFLAGS) --config ./fpga_impl/xclbin_overlay.cfg -o $@ $^
else ifeq (xilinx_u280_gen3x16_xdma_1_202211_1,xilinx_u250_gen3x16_xdma_4_1_202210_1)
$(XCLBIN): $(XOS)
	v++ -l $(XOCCFLAGS) $(XOCCLFLAGS) --config ./fpga_impl/xclbin_overlay.cfg -o $@ $^
endif

# FPGA all
fpga_all: fpga_compile fpga_link

# Clean up
clean_all:
	rm -f $(TARGET_EXEC) $(OBJECTS) $(FPGA_SOURCES:.cpp=.o) $(XOS) $(XSA_OBJ) $(XCLBIN)
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
	rm -f $(TARGET_EXEC) $(OBJECTS)

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