CXX := clang++
CXXFLAGS := -Wall -Wextra -pedantic -std=c++11 -fopenmp -w
LDFLAGS := -fopenmp
OPENMP_LIBS := -lgomp

# iterations
ITERATIONS := 1

# NI, NJ, NK, NL, NM
NI := 1024
NJ := 1024
NK := 1024
NL := 1024
NM := 1024

# Project name
TARGET_EXEC := krnl_3mm_sw

# Source files
SOURCES := main.cpp init_array.cpp

# MACRO DEFINITION
MACRO := -DITERATIONS=$(ITERATIONS) -DNI=$(NI) -DNJ=$(NJ) -DNK=$(NK) -DNL=$(NL) -DNM=$(NM)

# CPU Source files
CPU_SOURCES := cpu_impl/kernel_3mm_0.cpp cpu_impl/kernel_3mm_1.cpp cpu_impl/kernel_3mm_2.cpp

# Object files
OBJECTS := $(SOURCES:.cpp=.o)

CPU_OBJECTS := $(CPU_SOURCES:.cpp=.o)

INCLUDE_DIRS := cpu_impl/include .
INCLUDE_PARAMS := $(addprefix -I, $(INCLUDE_DIRS))

# Default target
all: $(TARGET_EXEC)

$(TARGET_EXEC): $(OBJECTS) $(CPU_OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(OPENMP_LIBS)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

init_array.o: init_array.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

# To compile other CPU source files
cpu_impl/%.o: cpu_impl/%.cpp
	$(CXX) $(CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

# Run
run:
	./$(TARGET_EXEC)

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