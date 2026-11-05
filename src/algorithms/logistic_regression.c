#include "algorithms/logistic_regression.h"
#include <math.h>

double sigmoid(double z){

    if(z < 0){
        return exp(z) / (1 + exp(z));
    }

    return 1 / (1 + exp(-z));
}

double logistic_regression_predict_proba(
    const LogisticRegression *model,
    double x
){
    double z = model->weight * x + model->bias;

    return sigmoid(z);
}

int logistic_regression_predict(
    const LogisticRegression *model,
    double x
){
    if(logistic_regression_predict_proba(model, x) >= 0.5){
        return 1;
    }

    return 0;
}

void logistic_regression_fit(
    LogisticRegression *model,
    const double *x,
    const double *y,
    size_t n,
    double learning_rate,
    size_t epochs
){
    if(model == NULL || x == NULL || y == NULL || n == 0 || epochs == 0 || learning_rate <= 0 || !isfinite(learning_rate)){
        return;
    }

    for(size_t epoch = 0; epoch < epochs; epoch++){
        double dw = 0;
        double db = 0;

        for(size_t i = 0; i < n; i++){
            double proba = logistic_regression_predict_proba(model, x[i]);

            double error = proba - y[i];

            dw += error * x[i];
            db += error;
        }

        dw /= n;
        db /= n;

        model->weight -= learning_rate * dw;
        model->bias -= learning_rate * db;
    }
}
