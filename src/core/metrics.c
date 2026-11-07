#include "core/metrics.h"
#include <math.h>

double binary_cross_entropy(
    const double *actual,
    const double *probabilities,
    size_t n
){
    if (actual == NULL || probabilities == NULL || n == 0) {
        return NAN;
    }

    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double y = actual[i];
        double p = probabilities[i];
        if ((y != 0.0 && y != 1.0) || !isfinite(p) || p < 0.0 || p > 1.0) {
            return NAN;
        }

        /* Select the relevant term to avoid 0 * log(0).
         * log1p(-p) accurately computes log(1 - p) for small p.
         */
        if ((y == 1.0 && p == 0.0) || (y == 0.0 && p == 1.0)) {
            total = INFINITY;
        } else {
            total += y == 1.0 ? -log(p) : -log1p(-p);
        }
    }

    return total / n;
}

double mean_squared_error(
    const double *actual,
    const double *predicted,
    size_t n
){
    if(actual == NULL || predicted == NULL){
        return NAN;
    }

    if(n == 0){
        return NAN;
    }

    double error = 0;

    for(size_t i = 0; i < n; i++){
        error += (actual[i] - predicted[i]) * (actual[i] - predicted[i]);
    }

    error /= n;

    return error;
}

double accuracy_score(
    const int *actual,
    const int *predicted,
    size_t n
){

    if(actual == NULL || predicted == NULL || n == 0){
        return NAN;
    }

    size_t count = 0;

    for(size_t i = 0; i < n; i++){
        if(actual[i] == predicted[i]){
            count++;
        }
    }

    return (double)count / n;
}
