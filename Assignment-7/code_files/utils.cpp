#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

double min_val, max_val;

void interpolation(double *mesh_value, Points *points) {
    int total_cells = GRID_X * GRID_Y;
    memset(mesh_value, 0, total_cells * sizeof(double));

    double scale_x = (double)NX;
    double scale_y = (double)NY;
    double area_cell = dx * dy;

    int thread_count = omp_get_max_threads();
    
    double *all_local = (double*)calloc(thread_count * total_cells, sizeof(double));

    #pragma omp parallel 
    {
        int thread_id = omp_get_thread_num();
        double *local_grid = &all_local[thread_id * total_cells];

        #pragma omp for schedule(static)
        for (int idx = 0; idx < NUM_Points; idx++) {
            if (points->is_void[idx]) continue; 

            double pos_x = points->x[idx];
            double pos_y = points->y[idx];

            int grid_i = (int)(pos_x * scale_x);
            int grid_j = (int)(pos_y * scale_y);

            grid_i = (grid_i >= NX) ? NX - 1 : ((grid_i < 0) ? 0 : grid_i);
            grid_j = (grid_j >= NY) ? NY - 1 : ((grid_j < 0) ? 0 : grid_j);

            double frac_x = (pos_x * scale_x) - grid_i;
            double frac_y = (pos_y * scale_y) - grid_j;

            double one_minus_x = 1.0 - frac_x;
            double one_minus_y = 1.0 - frac_y;

            double weight_00 = one_minus_x * one_minus_y * area_cell;
            double weight_10 = frac_x * one_minus_y * area_cell;
            double weight_01 = one_minus_x * frac_y * area_cell;
            double weight_11 = frac_x * frac_y * area_cell;

            int index_base = grid_j * GRID_X + grid_i;

            local_grid[index_base] += weight_00;
            local_grid[index_base + 1] += weight_10;
            local_grid[index_base + GRID_X] += weight_01;
            local_grid[index_base + GRID_X + 1] += weight_11;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int cell = 0; cell < total_cells; cell++) {
        double accum = 0.0;
        for (int t = 0; t < thread_count; t++) {
            accum += all_local[t * total_cells + cell];
        }
        mesh_value[cell] += accum;
    }

    free(all_local);
}

void normalization(double *mesh_value) {
    int total_cells = GRID_X * GRID_Y;
    min_val = mesh_value[0];
    max_val = mesh_value[0];
    
    #pragma omp parallel
    {
        double local_min_val = mesh_value[0];
        double local_max_val = mesh_value[0];
        
        #pragma omp for schedule(static)
        for (int idx = 0; idx < total_cells; idx++) {
            if (mesh_value[idx] < local_min_val) local_min_val = mesh_value[idx];
            if (mesh_value[idx] > local_max_val) local_max_val = mesh_value[idx];
        }
        
        #pragma omp critical
        {
            if (local_min_val < min_val) min_val = local_min_val;
            if (local_max_val > max_val) max_val = local_max_val;
        }
    }

    double diff = max_val - min_val;
    if (diff == 0.0) diff = 1.0; 

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < total_cells; idx++) {
        mesh_value[idx] = 2.0 * (mesh_value[idx] - min_val) / diff - 1.0;
    }
}

void mover(double *mesh_value, Points *points) {
    double scale_x = (double)NX;
    double scale_y = (double)NY;
    double area_cell = dx * dy;

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < NUM_Points; idx++) {
        if (points->is_void[idx]) continue;

        double pos_x = points->x[idx];
        double pos_y = points->y[idx];

        int grid_i = (int)(pos_x * scale_x);
        int grid_j = (int)(pos_y * scale_y);

        grid_i = (grid_i >= NX) ? NX - 1 : ((grid_i < 0) ? 0 : grid_i);
        grid_j = (grid_j >= NY) ? NY - 1 : ((grid_j < 0) ? 0 : grid_j);

        double frac_x = (pos_x * scale_x) - grid_i;
        double frac_y = (pos_y * scale_y) - grid_j;

        double one_minus_x = 1.0 - frac_x;
        double one_minus_y = 1.0 - frac_y;

        double weight_00 = one_minus_x * one_minus_y * area_cell;
        double weight_10 = frac_x * one_minus_y * area_cell;
        double weight_01 = one_minus_x * frac_y * area_cell;
        double weight_11 = frac_x * frac_y * area_cell;

        int index_base = grid_j * GRID_X + grid_i;

        double force_val = weight_00 * mesh_value[index_base] +
                           weight_10 * mesh_value[index_base + 1] +
                           weight_01 * mesh_value[index_base + GRID_X] +
                           weight_11 * mesh_value[index_base + GRID_X + 1];

        points->x[idx] += force_val * dx;
        points->y[idx] += force_val * dy;

        if (points->x[idx] < 0.0 || points->x[idx] > 1.0 || 
            points->y[idx] < 0.0 || points->y[idx] > 1.0) {
            points->is_void[idx] = true;
        }
    }
}

void denormalization(double *mesh_value) {
    int total_cells = GRID_X * GRID_Y;
    double diff = max_val - min_val;
    if (diff == 0.0) diff = 1.0;

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < total_cells; idx++) {
        mesh_value[idx] = (mesh_value[idx] + 1.0) * diff / 2.0 + min_val;
    }
}

long long int void_count(Points *points) {
    long long int count_voids = 0;
    #pragma omp parallel for reduction(+:count_voids)
    for (int idx = 0; idx < NUM_Points; idx++) {
        count_voids += (int)points->is_void[idx];
    }
    return count_voids;
}

void save_mesh(double *mesh_value) {
    FILE *file_ptr = fopen("Mesh.out", "w");
    if (!file_ptr) {
        printf("Error creating Mesh.out\n");
        exit(1);
    }
    for (int row = 0; row < GRID_Y; row++) {
        for (int col = 0; col < GRID_X; col++) {
            fprintf(file_ptr, "%lf ", mesh_value[row * GRID_X + col]);
        }
        fprintf(file_ptr, "\n");
    }
    fclose(file_ptr);
}