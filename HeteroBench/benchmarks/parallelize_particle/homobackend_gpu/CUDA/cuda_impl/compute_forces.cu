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
 
#include "cuda_impl.h"
#include <cuda_runtime.h>

using namespace std;

__device__ int d_Min(int a, int b) { return a < b ? a : b; }
__device__ int d_Max(int a, int b) { return a > b ? a : b; }

__device__ int grid_coord_cp1(double c)
{
    return (int)floor(c / cutoff);
}

//
//  interact two particles
//
__device__ void apply_force_static(particle_t& d_particle, particle_t& d_neighbor)
{
	double dx = d_neighbor.x - d_particle.x;
	double dy = d_neighbor.y - d_particle.y;
	double r2 = dx * dx + dy * dy;
	if (r2 > cutoff * cutoff)
		return;
	r2 = fmax(r2, min_r * min_r);
	double r = sqrt(r2);

	//
	//  very simple short-range repulsive force
	//
	double coef = (1 - cutoff / r) / r2 / mass;
	d_particle.ax += coef * dx;
	d_particle.ay += coef * dy;
}


__global__ void compute_forces_kernel(particle_t* d_particles, int n, linkedlist_static *d_grid_static) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < n) {
		// Reset acceleration
		d_particles[i].ax = d_particles[i].ay = 0;

		// Use the grid to traverse neighbours
		int gx = grid_coord_cp1(d_particles[i].x);
		int gy = grid_coord_cp1(d_particles[i].y);

		for (int x = d_Max(gx - 1, 0); x <= d_Min(gx + 1, gridsize - 1); x++)
		{
			for (int y = d_Max(gy - 1, 0); y <= d_Min(gy + 1, gridsize - 1); y++)
			{
				linkedlist_static* curr = &d_grid_static[x * gridsize + y];
				int t = curr->index_now;
				while (t != 0)
				{
					apply_force_static(d_particles[i], (curr->value[t]));
					t--;
				}
			}
		}
	}
}

void compute_forces_static(particle_t* particles, int n, linkedlist_static grid_static[gridsize2])
{
	particle_t* d_particles;
	linkedlist_static *d_grid_static;

	cudaMalloc(&d_particles, n*sizeof(particle_t));
	cudaMalloc(&d_grid_static, gridsize2*sizeof(linkedlist_static));

	cudaMemcpy(d_particles, particles, n*sizeof(particle_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_grid_static, grid_static, gridsize2*sizeof(particle_t), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

	compute_forces_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_particles, n, d_grid_static);

	cudaMemcpy(particles, d_particles, n*sizeof(particle_t), cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();
	// check for error
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess) {
		// print the CUDA error message and exit
		printf("CUDA error compute_forces_kernel: %s\n", cudaGetErrorString(error));
		exit(-1);
	}

	cudaFree(d_particles);
	cudaFree(d_grid_static);
}