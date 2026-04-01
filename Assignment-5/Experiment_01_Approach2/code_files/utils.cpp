#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

// ---------------------------------------------------------
// Cloud-In-Cell (CIC) Bilinear Interpolation
// ---------------------------------------------------------
void interpolation(double *grid_data, Points *pt_arr) {
    for(int idx = 0; idx < GRID_X * GRID_Y; idx++) {
        grid_data[idx] = 0.0;
    }

    for (int p_idx = 0; p_idx < NUM_Points; p_idx++) {
        double coord_x = pt_arr[p_idx].x * NX;
        double coord_y = pt_arr[p_idx].y * NY;

        int base_i = (int)coord_x;
        int base_j = (int)coord_y;

        if (base_i >= NX) base_i = NX - 1;
        if (base_j >= NY) base_j = NY - 1;

        double frac_x = coord_x - base_i;
        double frac_y = coord_y - base_j;

        double wt_00 = (1.0 - frac_x) * (1.0 - frac_y);
        double wt_10 = frac_x * (1.0 - frac_y);
        double wt_01 = (1.0 - frac_x) * frac_y;
        double wt_11 = frac_x * frac_y;

        grid_data[base_i + base_j * GRID_X] += wt_00;
        grid_data[(base_i + 1) + base_j * GRID_X] += wt_10;
        grid_data[base_i + (base_j + 1) * GRID_X] += wt_01;
        grid_data[(base_i + 1) + (base_j + 1) * GRID_X] += wt_11;
    }
}

// ---------------------------------------------------------
// APPROACH 1: Deferred Insertion (Serial)
// ---------------------------------------------------------
void mover_serial_deferred(Points *pt_arr, double step_x, double step_y) {
    int removed_cnt = 0;
    int *empty_slots = (int *)malloc(NUM_Points * sizeof(int));

    for (int idx = 0; idx < NUM_Points; idx++) {
        double rand_dx = ((double)rand() / RAND_MAX * 2.0 * step_x) - step_x;
        double rand_dy = ((double)rand() / RAND_MAX * 2.0 * step_y) - step_y;

        pt_arr[idx].x += rand_dx;
        pt_arr[idx].y += rand_dy;

        if (pt_arr[idx].x < 0.0 || pt_arr[idx].x > 1.0 || pt_arr[idx].y < 0.0 || pt_arr[idx].y > 1.0) {
            empty_slots[removed_cnt++] = idx;
        }
    }

    for (int k_idx = 0; k_idx < removed_cnt; k_idx++) {
        int replace_idx = empty_slots[k_idx];
        pt_arr[replace_idx].x = (double)rand() / RAND_MAX;
        pt_arr[replace_idx].y = (double)rand() / RAND_MAX;
    }

    free(empty_slots);
}

// ---------------------------------------------------------
// APPROACH 2: Immediate Replacement (Serial)
// ---------------------------------------------------------
void mover_serial_immediate(Points *pt_arr, double step_x, double step_y) {
    for (int idx = 0; idx < NUM_Points; idx++) {
        double rand_dx = ((double)rand() / RAND_MAX * 2.0 * step_x) - step_x;
        double rand_dy = ((double)rand() / RAND_MAX * 2.0 * step_y) - step_y;

        double new_x = pt_arr[idx].x + rand_dx;
        double new_y = pt_arr[idx].y + rand_dy;

        if (new_x < 0.0 || new_x > 1.0 || new_y < 0.0 || new_y > 1.0) {
            pt_arr[idx].x = (double)rand() / RAND_MAX;
            pt_arr[idx].y = (double)rand() / RAND_MAX;
        } else {
            pt_arr[idx].x = new_x;
            pt_arr[idx].y = new_y;
        }
    }
}

// ---------------------------------------------------------
// APPROACH 1: Deferred Insertion (OpenMP Parallel)
// ---------------------------------------------------------
void mover_parallel_deferred(Points *pt_arr, double step_x, double step_y) {
    int total_removed = 0;
    int *shared_empty = (int *)malloc(NUM_Points * sizeof(int));

    #pragma omp parallel
    {
        unsigned int rng_seed = 12345 ^ omp_get_thread_num();
        
        int *thread_empty = (int *)malloc(NUM_Points * sizeof(int));
        int thread_count = 0;

        #pragma omp for
        for (int idx = 0; idx < NUM_Points; idx++) {
            double rand_dx = ((double)rand_r(&rng_seed) / RAND_MAX * 2.0 * step_x) - step_x;
            double rand_dy = ((double)rand_r(&rng_seed) / RAND_MAX * 2.0 * step_y) - step_y;

            pt_arr[idx].x += rand_dx;
            pt_arr[idx].y += rand_dy;

            if (pt_arr[idx].x < 0.0 || pt_arr[idx].x > 1.0 || pt_arr[idx].y < 0.0 || pt_arr[idx].y > 1.0) {
                thread_empty[thread_count++] = idx;
            }
        }

        int write_start;
        #pragma omp atomic capture
        {
            write_start = total_removed;
            total_removed += thread_count;
        }

        for (int k_idx = 0; k_idx < thread_count; k_idx++) {
            shared_empty[write_start + k_idx] = thread_empty[k_idx];
        }
        free(thread_empty);

        #pragma omp barrier

        #pragma omp for
        for (int k_idx = 0; k_idx < total_removed; k_idx++) {
            int replace_idx = shared_empty[k_idx];
            pt_arr[replace_idx].x = (double)rand_r(&rng_seed) / RAND_MAX;
            pt_arr[replace_idx].y = (double)rand_r(&rng_seed) / RAND_MAX;
        }
    }

    free(shared_empty);
}

// ---------------------------------------------------------
// APPROACH 2: Immediate Replacement (OpenMP Parallel)
// ---------------------------------------------------------
void mover_parallel_immediate(Points *pt_arr, double step_x, double step_y) {
    #pragma omp parallel
    {
        unsigned int rng_seed = 12345 ^ omp_get_thread_num();

        #pragma omp for
        for (int idx = 0; idx < NUM_Points; idx++) {
            double rand_dx = ((double)rand_r(&rng_seed) / RAND_MAX * 2.0 * step_x) - step_x;
            double rand_dy = ((double)rand_r(&rng_seed) / RAND_MAX * 2.0 * step_y) - step_y;

            double new_x = pt_arr[idx].x + rand_dx;
            double new_y = pt_arr[idx].y + rand_dy;

            if (new_x < 0.0 || new_x > 1.0 || new_y < 0.0 || new_y > 1.0) {
                pt_arr[idx].x = (double)rand_r(&rng_seed) / RAND_MAX;
                pt_arr[idx].y = (double)rand_r(&rng_seed) / RAND_MAX;
            } else {
                pt_arr[idx].x = new_x;
                pt_arr[idx].y = new_y;
            }
        }
    }
}

// ---------------------------------------------------------
// Helper: Save Mesh
// ---------------------------------------------------------
void save_mesh(double *grid_data) {
    FILE *file_ptr = fopen("Mesh.out", "wb");
    if (file_ptr) {
        fwrite(grid_data, sizeof(double), GRID_X * GRID_Y, file_ptr);
        fclose(file_ptr);
    }
}