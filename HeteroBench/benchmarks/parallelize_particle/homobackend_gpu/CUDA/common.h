// #include "gpu_impl.h"

//
//  timing routines
//
double read_timer( );

//
//  simulation routines
//
double set_size( int n );
void init_particles( int n, particle_t *p , particle_t *p_sta);
void move_particles(particle_t* particles, int n, grid_t& grid);
void compute_forces(particle_t* particles, int n, grid_t& grid);

//
//  I/O routines
//
FILE *open_save( char *filename, int n );
void save( FILE *f, int n, particle_t *p );

//
//  argument processing routines
//
int find_option( int argc, char **argv, const char *option );
int read_int( int argc, char **argv, const char *option, int default_value );
char *read_string( int argc, char **argv, const char *option, char *default_value );