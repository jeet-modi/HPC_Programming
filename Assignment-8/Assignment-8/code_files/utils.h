#ifndef UTILS_H
#define UTILS_H

typedef struct {
    double x;
    double y;
    int is_void;
} Points;

extern int GRID_X, GRID_Y, NUM_Points, Maxiter;
extern double dx, dy;

Points* load_input(const char *filename);
void write_output(double *mesh);

void interpolate_points(Points *local_points, int local_n,
                        double *local_mesh,
                        double **thread_mesh,
                        int num_threads,
                        int grid_size,
                        int stride);

void normalize_mesh(double *mesh, int grid_size,
                    double *min_val, double *max_val);

void denormalize_mesh(double *mesh, int grid_size,
                      double min_val, double max_val);

void move_particles(Points *local_points, int local_n,
                    double *mesh, int stride);

#endif
