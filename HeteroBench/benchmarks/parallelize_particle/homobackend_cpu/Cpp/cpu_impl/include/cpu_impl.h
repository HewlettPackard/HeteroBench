#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <vector>

//
//  tuned constants
//
constexpr double density = 0.0005;
constexpr double mass = 0.01;
constexpr double cutoff = 0.01;
constexpr double min_r = (cutoff/100);
constexpr double dt = 0.0005;

//
//  saving parameters
//
// const int NSTEPS = 20;
// const int NPARTICLES = 1000000;
// const int SAVEFREQ = 10;

//
// particle data structure
//

template<typename T>
constexpr T sqrt_helper(T x, T curr, T prev) {
    return curr == prev ? curr : sqrt_helper(x, 0.5 * (curr + x / curr), curr);
}

template<typename T>
constexpr T constexpr_sqrt(T x) {
    return sqrt_helper(x, x, static_cast<T>(0));
}

constexpr double SIZE = constexpr_sqrt(density * NPARTICLES);
constexpr int GRID_SIZE = static_cast<int>((SIZE / cutoff) + 1);
constexpr int gridsize = GRID_SIZE;
constexpr int gridsize2 = GRID_SIZE * GRID_SIZE;
constexpr int max_linkedlist_size = 5;

//
// particle data structure
//
typedef struct 
{
  double x;
  double y;
  double vx;
  double vy;
  double ax;
  double ay;
} particle_t;

struct linkedlist
{
	linkedlist * next;
	particle_t * value;
};

typedef struct linkedlist linkedlist_t;

struct grid
{
	int size;
	linkedlist_t ** grid;
  int * node_counts;
};

typedef struct grid grid_t;
//static version
struct linkedlist_static
{
  particle_t value[max_linkedlist_size];
  int index_now = 0;
};



//
// grid routines
//

int grid_coord(double c);
int grid_coord_flat(int size, double x, double y);
void grid_init(grid_t & grid, int size);
void grid_add(grid_t & grid, particle_t * particle);
bool grid_remove(grid_t & grid, particle_t * p, int gridCoord = -1);
void grid_move( particle_t &p , double size);
void grid_clear(grid_t & grid);
int  grid_size(grid_t & grid);
void grid_add_static(linkedlist_static grad_static[gridsize2], particle_t * p);
void compute_forces_static(particle_t * particles, int n, linkedlist_static grid_static[gridsize2] );
void move_particles_static(particle_t * particles, int n, linkedlist_static grid[gridsize2]);