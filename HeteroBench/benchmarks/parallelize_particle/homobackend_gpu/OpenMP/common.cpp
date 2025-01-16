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
 
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <sys/time.h>

#include "gpu_impl.h"

double size;

//
//  timer
//
double read_timer( )
{
	static bool initialized = false;
	static struct timeval start;
	struct timeval end;
	if( !initialized )
	{
		gettimeofday( &start, NULL );
		initialized = true;
	}
	gettimeofday( &end, NULL );
	return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

//
//  keep density constant
//
double set_size( int n )
{
	size = sqrt( density * n );
	return size;
}

//
//  Initialize the particle positions and velocities
//
void init_particles( int n, particle_t *p, particle_t *p_sta )
{
	long seed = (long)time(NULL);
	srand48(seed);
	int sx = (int)ceil(sqrt((double)n));
	int sy = (n+sx-1)/sx;

	int *shuffle = (int*)malloc( n * sizeof(int) );
	for( int i = 0; i < n; i++ )
		shuffle[i] = i;

	for( int i = 0; i < n; i++ ) 
	{
		//
		//  make sure particles are not spatially sorted
		//
		long seed = (long)time(NULL);
		srand48(seed);
		int j = lrand48()%(n-i);
		int k = shuffle[j];
		shuffle[j] = shuffle[n-i-1];

		//
		//  distribute particles evenly to ensure proper spacing
		//
		p[i].x = size*(1.+(k%sx))/(1+sx);
		p[i].y = size*(1.+(k/sx))/(1+sy);
		p_sta[i].x = p[i].x;
		p_sta[i].y = p[i].y ;
		//
		//  assign random velocities within a bound
		//
        p[i].vx = drand48()*2-1;
		p[i].vy = drand48()*2-1;
		p_sta[i].vx =  p[i].vx;
		p_sta[i].vy =p[i].vy;
	}
	free( shuffle );
}

//
//  I/O routines
//
void save( FILE *f, int n, particle_t *p )
{
	static bool first = true;
	if( first )
	{
		fprintf( f, "%d %g\n", n, size );
		first = false;
	}
	for( int i = 0; i < n; i++ ){
		fprintf( f, "%g %g\n", p[i].x, p[i].y );
		std::cout<< "x " <<p[i].x << "y " << p[i].y<<std::endl;
	}
		
}

//
//  command line option processing
//
int find_option( int argc, char **argv, const char *option )
{
	for( int i = 1; i < argc; i++ )
		if( strcmp( argv[i], option ) == 0 )
			return i;
	return -1;
}

int read_int( int argc, char **argv, const char *option, int default_value )
{
	int iplace = find_option( argc, argv, option );
	if( iplace >= 0 && iplace < argc-1 )
		return atoi( argv[iplace+1] );
	return default_value;
}

char *read_string( int argc, char **argv, const char *option, char *default_value )
{
	int iplace = find_option( argc, argv, option );
	if( iplace >= 0 && iplace < argc-1 )
		return argv[iplace+1];
	return default_value;
}

inline int Min(int a, int b) { return a < b ? a : b; }
inline int Max(int a, int b) { return a > b ? a : b; }

void move_particles(particle_t* particles, int n, grid_t& grid)
{
	for (int i = 0; i < n; i++)
	{
		int gc = grid_coord_flat(grid.size, particles[i].x, particles[i].y);

		grid_move(particles[i], grid.size);

		// Re-add the particle if it has changed grid position
		if (gc != grid_coord_flat(grid.size, particles[i].x, particles[i].y))
		{
			if (!grid_remove(grid, &particles[i], gc))
			{
				fprintf(stdout, "Error: Failed to remove particle '%p'. Code must be faulty. Blame source writer.\n", &particles[i]);
				exit(3);
			}
			grid_add(grid, &particles[i]);
		}
	}
}

void apply_force(particle_t& particle, particle_t& neighbor)
{
	double dx = neighbor.x - particle.x;
	double dy = neighbor.y - particle.y;
	double r2 = dx * dx + dy * dy;
	if (r2 > cutoff * cutoff)
		return;
	r2 = fmax(r2, min_r * min_r);
	double r = sqrt(r2);

	//
	//  very simple short-range repulsive force
	//
	double coef = (1 - cutoff / r) / r2 / mass;
	particle.ax += coef * dx;
	particle.ay += coef * dy;
}

void compute_forces(particle_t* particles, int n, grid_t& grid)
{
	for (int i = 0; i < n; i++)
	{
		// Reset acceleration
		particles[i].ax = particles[i].ay = 0;

		// Use the grid to traverse neighbours
		int gx = grid_coord(particles[i].x);
		int gy = grid_coord(particles[i].y);

		for (int x = Max(gx - 1, 0); x <= Min(gx + 1, grid.size - 1); x++)
		{
			for (int y = Max(gy - 1, 0); y <= Min(gy + 1, grid.size - 1); y++)
			{
				linkedlist_t* curr = grid.grid[x * grid.size + y];
				while (curr != 0)
				{
					apply_force(particles[i], *(curr->value));
					curr = curr->next;
				}
			}
		}
	}
}