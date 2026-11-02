#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <stddef.h>

typedef struct {
    double weight;
    double bias;
} LinearRegression;

double linear_regression_predict(
    const LinearRegression *model,
    double x
);

void linear_regression_fit(
    LinearRegression *model,
    const double *x,
    const double *y,
    size_t n,
    double learning_rate,
    size_t epochs
);

#endif
