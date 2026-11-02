#include <stdio.h>
#include "include/ml.h"

int main(void)
{
    double x[] = {1, 2, 3, 4, 5};
    double y[] = {3, 5, 7, 9, 11};

    LinearRegression model = {0.0, 0.0};

    linear_regression_fit(&model, x, y, 5, 0.01, 10000);

    printf("Weight: %.4f\n", model.weight);
    printf("Bias: %.4f\n", model.bias);
    
    double predictions[5];

    for (size_t i = 0; i < 5; i++) {
        predictions[i] = linear_regression_predict(&model, x[i]);
    }

    double mse = mean_squared_error(y, predictions, 5);

    printf("Training MSE: %.8f\n", mse);

    return 0;
}
