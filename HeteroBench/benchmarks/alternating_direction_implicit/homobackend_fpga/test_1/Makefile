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

# iterations
ITERATIONS := 1

# TSTEPS, N
TSTEPS := 50
N := 1024

# Project name
TARGET_EXEC := krnl_adi_sw

# Source files
SOURCES := main.cpp

# MACRO DEFINITION
MACRO := -DITERATIONS=$(ITERATIONS) -DTSTEPS=$(TSTEPS) -DN=$(N)

# FPGA Source files
FPGA_SOURCES := fpga_impl/init_array.cpp fpga_impl/kernel_adi.cpp

# Object files
OBJECTS := $(SOURCES:.cpp=.o)

# Include directories
INCLUDE_DIRS := fpga_impl/include
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
	./$(TARGET_EXEC) $(INPUT_PATH) $(OUTPUT_PATH)

# FPGA Compilation

TARGET := hw
XOCCFLAGS := --platform xilinx_u280_gen3x16_xdma_1_202211_1 -t $(TARGET) -s -g

FPGA_OBJECTS := $(FPGA_SOURCES:fpga_impl/%.cpp=fpga_impl/%_$(TARGET).xo)

fpga_compile: $(FPGA_OBJECTS)

fpga_impl/%_$(TARGET).xo: fpga_impl/%.cpp
	v++ $(XOCCFLAGS) $(INCLUDE_PARAMS) $(MACRO) --kernel $* -c -o $@ $<

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