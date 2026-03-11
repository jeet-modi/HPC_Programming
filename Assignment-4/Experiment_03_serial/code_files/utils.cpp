#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

void interpolation(double *mesh_value, Points *points)
{
    memset(mesh_value,0,sizeof(double)*GRID_X*GRID_Y);

    for(long p=0;p<NUM_Points;p++)
    {
        int ix = (int)(points[p].x * NX);
        int iy = (int)(points[p].y * NY);

        mesh_value[iy*GRID_X + ix] += 1.0;
    }
}

void mover_serial(Points *points,double dx,double dy)
{
    for(long i=0;i<NUM_Points;i++)
    {
        double rx = ((double)rand()/RAND_MAX)*2-1;
        double ry = ((double)rand()/RAND_MAX)*2-1;

        points[i].x += rx*dx;
        points[i].y += ry*dy;
    }
}

void mover_parallel(Points *points,double dx,double dy)
{
#pragma omp parallel for
    for(long i=0;i<NUM_Points;i++)
    {
        unsigned int seed = i;

        double rx = ((double)rand_r(&seed)/RAND_MAX)*2-1;
        double ry = ((double)rand_r(&seed)/RAND_MAX)*2-1;

        points[i].x += rx*dx;
        points[i].y += ry*dy;
    }
}