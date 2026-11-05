#include "algorithms/linear_regression.h"

double linear_regression_predict(const LinearRegression *model, double x){
    return model->weight * x + model->bias;
}

void linear_regression_fit(
    LinearRegression *model,
    const double *x,
    const double *y,
    size_t n,
    double learning_rate,
    size_t epochs
){

    if(model == NULL || x == NULL || y == NULL){
        return;
    }

    if(n == 0 || learning_rate <= 0.0 || epochs == 0){
        return;
    }

    for (size_t epoch = 0; epoch < epochs; epoch++) {

        double dw = 0.0;
        double db = 0.0;

        for (size_t i = 0; i < n; i++) {
            double prediction = linear_regression_predict(model, x[i]);

            double error = prediction - y[i];

            dw += 2 * error * x[i];
            db += 2 * error;            
        }

        dw /= n;
        db /= n;

        model->weight -= learning_rate * dw;
        model->bias -= learning_rate * db;

    }
}