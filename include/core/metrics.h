#ifndef METRICS_H
#define METRICS_H

#include <stddef.h>

double mean_squared_error(
    const double *actual,
    const double *predicted,
    size_t n
);

/* Return the fraction of matching classes, or NAN if either pointer is NULL
 * or n is zero.
 */
double accuracy_score(
    const int *actual,
    const int *predicted,
    size_t n
);

#endif
