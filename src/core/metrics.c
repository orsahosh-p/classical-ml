#include "core/metrics.h"
#include <math.h>

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
