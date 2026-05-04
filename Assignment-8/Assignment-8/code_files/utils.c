#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

int GRID_X, GRID_Y, NUM_Points, Maxiter;
double dx, dy;

Points* load_input(const char *filename)
{
    FILE *fp = fopen(filename, "rb");

    fread(&GRID_X, sizeof(int), 1, fp);
    fread(&GRID_Y, sizeof(int), 1, fp);
    fread(&NUM_Points, sizeof(int), 1, fp);
    fread(&Maxiter, sizeof(int), 1, fp);

    Points *points = malloc(NUM_Points * sizeof(Points));

    for (int i = 0; i < NUM_Points; i++) {
        fread(&points[i].x, sizeof(double), 1, fp);
        fread(&points[i].y, sizeof(double), 1, fp);
        points[i].is_void = 0;
    }

    fclose(fp);
    return points;
}

void write_output(double *mesh)
{
    FILE *fp = fopen("output.txt", "w");

    int stride = GRID_X + 1;

    for (int j = 0; j <= GRID_Y; j++) {
        for (int i = 0; i <= GRID_X; i++) {
            fprintf(fp, "%.6lf ", mesh[j * stride + i]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

void interpolate_points(Points *local_points, int local_n,
                        double *local_mesh,
                        double **thread_mesh,
                        int num_threads,
                        int grid_size,
                        int stride)
{
    memset(local_mesh, 0, grid_size * sizeof(double));

    for (int t = 0; t < num_threads; t++)
        memset(thread_mesh[t], 0, grid_size * sizeof(double));

    double inv_dx = GRID_X;
    double inv_dy = GRID_Y;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *pmesh = thread_mesh[tid];

        #pragma omp for
        for (int p = 0; p < local_n; p++) {

            if (local_points[p].is_void) continue;

            double x = local_points[p].x;
            double y = local_points[p].y;

            int i = (int)(x * inv_dx);
            int j = (int)(y * inv_dy);

            if (i < 0) i = 0;
            if (j < 0) j = 0;
            if (i >= GRID_X) i = GRID_X - 1;
            if (j >= GRID_Y) j = GRID_Y - 1;

            double lx = x - i * dx;
            double ly = y - j * dy;

            double w00 = (dx - lx) * (dy - ly);
            double w10 = lx * (dy - ly);
            double w01 = (dx - lx) * ly;
            double w11 = lx * ly;

            int idx = j * stride + i;

            pmesh[idx] += w00;
            pmesh[idx + 1] += w10;
            pmesh[idx + stride] += w01;
            pmesh[idx + stride + 1] += w11;
        }
    }

    for (int t = 0; t < num_threads; t++)
        for (int i = 0; i < grid_size; i++)
            local_mesh[i] += thread_mesh[t][i];
}

void normalize_mesh(double *mesh, int grid_size,
                    double *min_val, double *max_val)
{
    double local_min = 1e30;
    double local_max = -1e30;

    #pragma omp parallel for reduction(min:local_min) reduction(max:local_max)
    for (int i = 0; i < grid_size; i++) {
        if (mesh[i] < local_min) local_min = mesh[i];
        if (mesh[i] > local_max) local_max = mesh[i];
    }

    *min_val = local_min;
    *max_val = local_max;

    double range = (*max_val - *min_val);

    if (range > 1e-15) {
        #pragma omp parallel for
        for (int i = 0; i < grid_size; i++)
            mesh[i] = 2.0 * (mesh[i] - *min_val) / range - 1.0;
    }
}

void denormalize_mesh(double *mesh, int grid_size,
                      double min_val, double max_val)
{
    double range = max_val - min_val;

    if (range > 1e-15) {
        #pragma omp parallel for
        for (int i = 0; i < grid_size; i++)
            mesh[i] = ((mesh[i] + 1.0) * 0.5) * range + min_val;
    }
}

void move_particles(Points *local_points, int local_n,
                    double *mesh, int stride)
{
    double inv_dx = GRID_X;
    double inv_dy = GRID_Y;

    #pragma omp parallel for
    for (int p = 0; p < local_n; p++) {

        if (local_points[p].is_void) continue;

        double x = local_points[p].x;
        double y = local_points[p].y;

        int i = (int)(x * inv_dx);
        int j = (int)(y * inv_dy);

        if (i < 0) i = 0;
        if (j < 0) j = 0;
        if (i >= GRID_X) i = GRID_X - 1;
        if (j >= GRID_Y) j = GRID_Y - 1;

        double lx = x - i * dx;
        double ly = y - j * dy;

        double w00 = (dx - lx) * (dy - ly);
        double w10 = lx * (dy - ly);
        double w01 = (dx - lx) * ly;
        double w11 = lx * ly;

        int idx = j * stride + i;

        double Fi =
            w00 * mesh[idx] +
            w10 * mesh[idx + 1] +
            w01 * mesh[idx + stride] +
            w11 * mesh[idx + stride + 1];

        local_points[p].x += Fi * dx;
        local_points[p].y += Fi * dy;

        if (local_points[p].x < 0 || local_points[p].x > 1 ||
            local_points[p].y < 0 || local_points[p].y > 1)
            local_points[p].is_void = 1;
    }
}
