#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <stddef.h>

typedef struct {
    double weight;
    double bias;
} LogisticRegression;

double sigmoid(double z);

double logistic_regression_predict_proba(
    const LogisticRegression *model,
    double x
);

int logistic_regression_predict(
    const LogisticRegression *model,
    double x
);

void logistic_regression_fit(
    LogisticRegression *model,
    const double *x,
    const double *y,
    size_t n,
    double learning_rate,
    size_t epochs
);

#endif
