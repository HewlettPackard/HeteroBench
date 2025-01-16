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

using namespace std;

__device__ int grid_coord_cp2(double c)
{
    return (int)floor(c / cutoff);
}

__device__ int grid_coord_flat_cp2(int size, double x, double y)
{
    return grid_coord_cp2(x) * size + grid_coord_cp2(y);
}

__device__ void grid_add_static_cp2(linkedlist_static grad_static[gridsize2], particle_t* p)
{
    int gridCoord = grid_coord_flat_cp2(gridsize, p->x, p->y);

    grad_static[gridCoord].value[grad_static[gridCoord].index_now] = *p;
    grad_static[gridCoord].index_now++;

}

__device__ void grid_move_cp2(particle_t& p, double size)
{
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    //
    //  bounce from walls
    //
    while (p.x < 0 || p.x > size)
    {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }
    while (p.y < 0 || p.y > size)
    {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

__global__ void move_particles_kernel(particle_t* particles, int n, linkedlist_static *grid)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < n) {
		int gc = grid_coord_flat_cp2(gridsize, particles[i].x, particles[i].y);

		grid_move_cp2(particles[i], gridsize);

		// Re-add the particle if it has changed grid position
		if (gc != grid_coord_flat_cp2(gridsize, particles[i].x, particles[i].y))
		{
			grid_add_static_cp2(grid, &particles[i]);
		}
    }
}

void move_particles_static(particle_t* particles, int n, linkedlist_static grid[gridsize2])
{
    particle_t *d_particles;
    linkedlist_static *d_grid;

	cudaMalloc(&d_particles, n*sizeof(particle_t));
	cudaMalloc(&d_grid, gridsize2*sizeof(linkedlist_static));

	cudaMemcpy(d_particles, particles, n*sizeof(particle_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_grid, grid, gridsize2*sizeof(particle_t), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

	move_particles_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_particles, n, d_grid);

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
	cudaFree(d_grid);
}