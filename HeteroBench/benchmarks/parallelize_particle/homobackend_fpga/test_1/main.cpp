/*
 * (C) Copyright [2024] Hewlett Packard Enterprise Development LP
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the Software),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
 
// standard C/C++ headers
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <cmath>
#include <time.h>
#include <fstream>
#include <sys/time.h>
#include "omp.h"
// fpga related headers
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <sys/stat.h>
#include <string>
// #include <ap_int.h>
#include <ctime>
#include <stdlib.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <experimental/xrt_xclbin.h>
#include <xrt/xrt_kernel.h>
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_uuid.h"

// other headers
#include "fpga_impl.h"
#define Top_kernel_hw_ptr_particles 0
#define Top_kernel_hw_ptr_grid_static 1
#define DEVICE_ID 0
#define DEBUG 0
using namespace std;


particle_t particles_dyn[NPARTICLES];
particle_t particles[NPARTICLES];
bool compare_particles(const particle_t& p1, const particle_t& p2) {
	const double epsilon = 1e-10; // Tolerance for floating-point comparison
	return std::fabs(p1.x - p2.x) < epsilon &&
		std::fabs(p1.y - p2.y) < epsilon &&
		std::fabs(p1.vx - p2.vx) < epsilon &&
		std::fabs(p1.vy - p2.vy) < epsilon &&
		std::fabs(p1.ax - p2.ax) < epsilon &&
		std::fabs(p1.ay - p2.ay) < epsilon;
}

bool compare_particle_arrays(particle_t* arr1, particle_t* arr2, int size) {
	for (int i = 0; i < size; ++i) {
		if (!compare_particles(arr1[i], arr2[i])) {
			return false;
		}
	}
	return true;
}

void parallelize_particle_simulation(int n, int nsteps, int savefreq, FILE* fsave)
{

	double size = SIZE;

	// Create a grid for optimizing the interactions
	int gridSize = GRID_SIZE; // TODO: Rounding errors?

	grid_t grid;
	grid_init(grid, gridSize);


	std::cout << "Running " << nsteps << " iterations ..." << std::endl;
	// Simulate a number of time steps
	double start_whole_time = omp_get_wtime();

	double start_iteration_time;
	double start_compute_forces_time = 0;
	double start_move_particles_time = 0;

	for (int step = 0; step < nsteps; step++)
	{
		// Compute forces
		start_iteration_time = omp_get_wtime();
		compute_forces(particles_dyn, n, grid);
		start_compute_forces_time += omp_get_wtime() - start_iteration_time;

		// Move particles
		start_iteration_time = omp_get_wtime();
		move_particles(particles_dyn, n, grid);
		start_move_particles_time += omp_get_wtime() - start_iteration_time;
		// for (int i = 0; i < grid.size * grid.size; i++) {
		// 	if (grid.node_counts[i] > 1)
		// 		printf("Grid coordinate %d: %d nodes\n", i, grid.node_counts[i]);
		// }
		// Save if necessary
		if (fsave && (step % savefreq) == 0)
			save(fsave, n, particles);
	}
	std::cout << "Done" << std::endl;

	// double end_whole_time = omp_get_wtime() - start_whole_time;
	// std::cout << "Total " << nsteps << " number of steps" << endl;
	// std::cout << "Single iteration time: " << (end_whole_time / nsteps) * 1000 << " ms" << endl;
	// std::cout << "Compute forces time: " << (start_compute_forces_time / nsteps) * 1000 << " ms" << endl;
	// std::cout << "Move particles time: " << (start_move_particles_time / nsteps) * 1000 << " ms" << endl;

	grid_clear(grid);


	if (fsave)
	{
		fclose(fsave);
	}
}

void parallelize_particle_simulation_static(int n, int nsteps, int savefreq, FILE* fsave, linkedlist_static grid_static[gridsize2])
{


	// Load xclbin
	std::string xclbin_file = "overlay_hw.xclbin";
	std::cout << "Loading: " << xclbin_file << std::endl;
	xrt::device device = xrt::device(DEVICE_ID);
	xrt::uuid xclbin_uuid = device.load_xclbin(xclbin_file);
	std::cout << "Loaded xclbin: " << xclbin_file << std::endl;

	// create kernel object
	xrt::kernel Top_kernel_hw_kernel = xrt::kernel(device, xclbin_uuid, "top_kernel");
	// create memory groups
	xrtMemoryGroup bank_tp_particles = Top_kernel_hw_kernel.group_id(0);
	xrtMemoryGroup  bank_tp_grid_static = Top_kernel_hw_kernel.group_id(1);

	std::cout << "Running " << NSTEPS << " iteration ..." << std::endl;
	// create buffer objects
	xrt::bo data_buffer_Top_kernel_hw_particles = xrt::bo(device, sizeof(particles), xrt::bo::flags::normal, bank_tp_particles);
	xrt::bo data_buffer_Top_kernel_hw_grid_static = xrt::bo(device, gridsize2 * sizeof(linkedlist_static), xrt::bo::flags::normal, bank_tp_grid_static);
	cout << sizeof(grid_static) << "VS" << gridsize2 * sizeof(linkedlist_static) << endl;
	cout << sizeof(particles) << "VS" << n * sizeof(particle_t) << endl;
	// create kernel runner
	xrt::run run_Top_kernel_hw(Top_kernel_hw_kernel);
	double start_whole_time = omp_get_wtime();

	data_buffer_Top_kernel_hw_particles.write(particles);
	data_buffer_Top_kernel_hw_particles.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	data_buffer_Top_kernel_hw_grid_static.write(grid_static);
	data_buffer_Top_kernel_hw_grid_static.sync(XCL_BO_SYNC_BO_TO_DEVICE);

	// set arguments of kernel
	run_Top_kernel_hw.set_arg(Top_kernel_hw_ptr_particles, data_buffer_Top_kernel_hw_particles);
	run_Top_kernel_hw.set_arg(Top_kernel_hw_ptr_grid_static, data_buffer_Top_kernel_hw_grid_static);

	run_Top_kernel_hw.start();
	run_Top_kernel_hw.wait();

	data_buffer_Top_kernel_hw_particles.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
	data_buffer_Top_kernel_hw_particles.read(particles);

	data_buffer_Top_kernel_hw_grid_static.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
	data_buffer_Top_kernel_hw_grid_static.read(grid_static);

	std::cout << "Done" << std::endl;
	double run_whole_time = omp_get_wtime() - start_whole_time;
	cout << "Total " << NSTEPS << " iterations " << endl;
	cout << "Total " << NPARTICLES << " particles" << endl;
	cout << "Single iteration time: " << (run_whole_time / NSTEPS) * 1000 << " ms" << endl;


	if (fsave)
	{
		fclose(fsave);
	}
}

//
//  benchmarking program
//
int main(int argc, char** argv)
{

	std::cout << "=======================================" << std::endl;
	std::cout << "Running parallelize_particle benchmark C++ HLS" << std::endl;
	std::cout << "=======================================" << std::endl;
	linkedlist_static grid_static[gridsize2];

	int n = NPARTICLES;
	int nsteps = NSTEPS;
	int savefreq = SAVEFREQ;
	char* savename = read_string(argc, argv, "-o", NULL);

	FILE* fsave = savename ? fopen(savename, "w") : NULL;

	cout << "Number of particles: " << n << endl;

	double size = set_size(n);
	init_particles(n, particles_dyn, particles);
	parallelize_particle_simulation(n, nsteps, savefreq, fsave);
	parallelize_particle_simulation_static(n, nsteps, savefreq, fsave, grid_static);

	bool arrays_equal = compare_particle_arrays(particles_dyn, particles, NPARTICLES);

	if (arrays_equal) {
		std::cout << "The particle arrays are equal." << std::endl;
	}
	else {
		std::cout << "The particle arrays are not equal." << std::endl;
	}
	return 0;
}
