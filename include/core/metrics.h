#ifndef METRICS_H
#define METRICS_H

#include <stddef.h>

double mean_squared_error(
    const double *actual,
    const double *predicted,
    size_t n
);

#endif