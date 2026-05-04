#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "utils.h"

int main(int argc, char **argv)
{
    int rank, size;

    Points *points = NULL;
    Points *local_points = NULL;

    int base, rem, local_n;
    int *sendcounts = NULL;
    int *displs = NULL;

    int grid_size, stride;

    double *local_mesh = NULL;
    double *global_mesh = NULL;
    double *raw_mesh = NULL;

    double **thread_mesh;
    int num_threads;

    double total_start, total_end;

    double min_val, max_val;

    /* Timing variables */
    double t_interp = 0.0;
    double t_reduce = 0.0;
    double t_bcast  = 0.0;
    double t_norm   = 0.0;
    double t_mover  = 0.0;
    double t_denorm = 0.0;

    double ts, te;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
        points = load_input(argv[1]);

    MPI_Bcast(&GRID_X, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&GRID_Y, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NUM_Points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Maxiter, 1, MPI_INT, 0, MPI_COMM_WORLD);

    dx = 1.0 / GRID_X;
    dy = 1.0 / GRID_Y;

    stride = GRID_X + 1;
    grid_size = (GRID_X + 1) * (GRID_Y + 1);

    /* Scatter */
    base = NUM_Points / size;
    rem = NUM_Points % size;
    local_n = (rank < rem) ? base + 1 : base;

    if (rank == 0) {
        sendcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));

        displs[0] = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = ((i < rem) ? base + 1 : base) * sizeof(Points);
            if (i > 0) displs[i] = displs[i - 1] + sendcounts[i - 1];
        }
    }

    local_points = malloc(local_n * sizeof(Points));

    MPI_Scatterv(points, sendcounts, displs, MPI_BYTE,
                 local_points, local_n * sizeof(Points),
                 MPI_BYTE, 0, MPI_COMM_WORLD);

    local_mesh = calloc(grid_size, sizeof(double));
    global_mesh = calloc(grid_size, sizeof(double));
    raw_mesh = calloc(grid_size, sizeof(double));

    num_threads = omp_get_max_threads();

    thread_mesh = malloc(num_threads * sizeof(double*));
    for (int t = 0; t < num_threads; t++)
        thread_mesh[t] = calloc(grid_size, sizeof(double));

    MPI_Barrier(MPI_COMM_WORLD);
    total_start = MPI_Wtime();

    for (int iter = 0; iter < Maxiter; iter++) {

        /* Interpolation */
        ts = MPI_Wtime();
        interpolate_points(local_points, local_n,
                           local_mesh, thread_mesh,
                           num_threads, grid_size, stride);
        te = MPI_Wtime();
        t_interp += (te - ts);

        /* Reduce */
        ts = MPI_Wtime();
        MPI_Reduce(local_mesh, global_mesh, grid_size,
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        te = MPI_Wtime();
        t_reduce += (te - ts);

        /* Broadcast */
        ts = MPI_Wtime();
        MPI_Bcast(global_mesh, grid_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        te = MPI_Wtime();
        t_bcast += (te - ts);

        memcpy(raw_mesh, global_mesh, grid_size * sizeof(double));

        /* Normalize */
        ts = MPI_Wtime();
        normalize_mesh(global_mesh, grid_size, &min_val, &max_val);
        te = MPI_Wtime();
        t_norm += (te - ts);

        /* Mover */
        ts = MPI_Wtime();
        move_particles(local_points, local_n,
                       global_mesh, stride);
        te = MPI_Wtime();
        t_mover += (te - ts);

        /* Denormalize */
        ts = MPI_Wtime();
        denormalize_mesh(global_mesh, grid_size, min_val, max_val);
        te = MPI_Wtime();
        t_denorm += (te - ts);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    total_end = MPI_Wtime();

    if (rank == 0) {

        printf("\n===== TIMING BREAKDOWN =====\n");
        printf("Total Time        : %lf\n", total_end - total_start);
        printf("Interpolation     : %lf\n", t_interp);
        printf("MPI Reduce        : %lf\n", t_reduce);
        printf("MPI Broadcast     : %lf\n", t_bcast);
        printf("Normalization     : %lf\n", t_norm);
        printf("Mover             : %lf\n", t_mover);
        printf("Denormalization   : %lf\n", t_denorm);

        write_output(raw_mesh);
    }

    free(local_points);
    free(local_mesh);
    free(global_mesh);
    free(raw_mesh);

    for (int t = 0; t < num_threads; t++)
        free(thread_mesh[t]);
    free(thread_mesh);

    if (rank == 0) {
        free(points);
        free(sendcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}
