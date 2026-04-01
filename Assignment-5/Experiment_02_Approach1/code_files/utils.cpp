#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

// Interpolation (Serial Code)
void interpolation(double *grid_arr, Points *pt_list) {
    double scale_x = (double)NX; 
    double scale_y = (double)NY;

    for (int idx = 0; idx < NUM_Points; idx++) {
        double pos_x = pt_list[idx].x;
        double pos_y = pt_list[idx].y;

        int cell_i = (int)(pos_x * scale_x);
        int cell_j = (int)(pos_y * scale_y);

        if (cell_i >= NX - 1) cell_i = NX - 2;
        if (cell_j >= NY - 1) cell_j = NY - 2;

        double local_x = pos_x - (cell_i * dx);
        double local_y = pos_y - (cell_j * dy);

        double weight_xm = dx - local_x;
        double weight_ym = dy - local_y;

        int base = cell_j * GRID_X + cell_i;

        grid_arr[base]                     += weight_xm * weight_ym;
        grid_arr[base + 1]                 += local_x * weight_ym;
        grid_arr[base + GRID_X]            += weight_xm * local_y;
        grid_arr[base + GRID_X + 1]        += local_x * local_y;
    }
}

// --------------------------------------
// Mover: Deferred Insertion (Serial)
// --------------------------------------
void mover_serial_deferred(Points *pt_list, double stepX, double stepY)
{
    int write_ptr = 0;

    for (int idx = 0; idx < NUM_Points; idx++)
    {
        unsigned int rng_seed = 1234 + idx;

        double shift_x = ((double)rand_r(&rng_seed) / RAND_MAX * 2.0 - 1.0) * stepX;
        double shift_y = ((double)rand_r(&rng_seed) / RAND_MAX * 2.0 - 1.0) * stepY;

        double new_pos_x = pt_list[idx].x + shift_x;
        double new_pos_y = pt_list[idx].y + shift_y;

        if (new_pos_x >= 0.0 && new_pos_x <= 1.0 && new_pos_y >= 0.0 && new_pos_y <= 1.0)
        {
            pt_list[write_ptr].x = new_pos_x;
            pt_list[write_ptr].y = new_pos_y;
            write_ptr++;
        }
    }

    for (int idx = write_ptr; idx < NUM_Points; idx++)
    {
        unsigned int rng_seed = 5678 + idx;
        pt_list[idx].x = (double)rand_r(&rng_seed) / RAND_MAX;
        pt_list[idx].y = (double)rand_r(&rng_seed) / RAND_MAX;
    }
}

// -----------------------------
// Mover: Parallel Deferred
// -----------------------------
void mover_parallel_deferred(Points *pt_list, double stepX, double stepY)
{
    double *temp_x = (double *)malloc(NUM_Points * sizeof(double));
    double *temp_y = (double *)malloc(NUM_Points * sizeof(double));
    int *is_valid = (int *)malloc(NUM_Points * sizeof(int));
    int *scan_arr = (int *)malloc(NUM_Points * sizeof(int));

    // Phase 1: Move + mark valid
    #pragma omp parallel for
    for (int idx = 0; idx < NUM_Points; idx++)
    {
        unsigned int rng_seed = 1234 + idx;

        double shift_x = ((double)rand_r(&rng_seed) / RAND_MAX * 2.0 - 1.0) * stepX;
        double shift_y = ((double)rand_r(&rng_seed) / RAND_MAX * 2.0 - 1.0) * stepY;

        double new_pos_x = pt_list[idx].x + shift_x;
        double new_pos_y = pt_list[idx].y + shift_y;

        temp_x[idx] = new_pos_x;
        temp_y[idx] = new_pos_y;

        is_valid[idx] = (new_pos_x >= 0.0 && new_pos_x <= 1.0 && new_pos_y >= 0.0 && new_pos_y <= 1.0);
    }

    // Phase 2: Prefix sum
    scan_arr[0] = is_valid[0];
    for (int i = 1; i < NUM_Points; i++)
    {
        scan_arr[i] = scan_arr[i - 1] + is_valid[i];
    }

    int valid_total = scan_arr[NUM_Points - 1];
    int removed_total = NUM_Points - valid_total;

    // Phase 3: Scatter
    #pragma omp parallel for
    for (int idx = 0; idx < NUM_Points; idx++)
    {
        if (is_valid[idx])
        {
            int target_idx = scan_arr[idx] - 1;

            pt_list[target_idx].x = temp_x[idx];
            pt_list[target_idx].y = temp_y[idx];
        }
    }

    // Phase 4: Fill voids
    #pragma omp parallel for
    for (int idx = valid_total; idx < NUM_Points; idx++)
    {
        unsigned int rng_seed = 5678 + idx;

        pt_list[idx].x = (double)rand_r(&rng_seed) / RAND_MAX;
        pt_list[idx].y = (double)rand_r(&rng_seed) / RAND_MAX;
    }

    free(temp_x);
    free(temp_y);
    free(is_valid);
    free(scan_arr);
}

// Write mesh to file
void save_mesh(double *grid_arr) {

    FILE *file_handle = fopen("Mesh.out", "w");
    if (!file_handle) {
        printf("Error creating Mesh.out\n");
        exit(1);
    }

    for (int row = 0; row < GRID_Y; row++) {
        for (int col = 0; col < GRID_X; col++) {
            fprintf(file_handle, "%lf ", grid_arr[row * GRID_X + col]);
        }
        fprintf(file_handle, "\n");
    }

    fclose(file_handle);
}