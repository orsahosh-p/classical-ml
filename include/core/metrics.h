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

/* Mean binary cross-entropy (natural logarithm).
 * Labels must be 0.0 or 1.0; probabilities must be finite and in [0, 1].
 * Return NAN for invalid inputs, NULL pointers, or n == 0.
 * A confidently wrong probability (0 for label 1, or 1 for label 0)
 * produces positive infinity; perfect predictions produce zero.
 */
double binary_cross_entropy(
    const double *actual,
    const double *probabilities,
    size_t n
);

#endif
