#include <stdlib.h>
#include "init.h"

void initializepoints(Points *points)
{
    for(long i=0;i<NUM_Points;i++)
    {
        points[i].x = (double)rand()/RAND_MAX;
        points[i].y = (double)rand()/RAND_MAX;
    }
}