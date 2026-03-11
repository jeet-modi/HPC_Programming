#ifndef INIT_H
#define INIT_H

typedef struct
{
    double x;
    double y;
} Points;

extern int GRID_X, GRID_Y, NX, NY;
extern long NUM_Points;
extern int Maxiter;
extern double dx, dy;

void initializepoints(Points *points);

#endif