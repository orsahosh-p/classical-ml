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

    const double classification_x[] = {1, 2, 3, 7, 8, 9};
    const double classification_y[] = {0, 0, 0, 1, 1, 1};
    const size_t classification_n = sizeof classification_x / sizeof classification_x[0];
    LogisticRegression classifier = {0.0, 0.0};

    logistic_regression_fit(&classifier, classification_x, classification_y,
                            classification_n, 0.01, 10000);

    printf("\nLogistic regression\n");
    printf("Weight: %.4f\n", classifier.weight);
    printf("Bias: %.4f\n", classifier.bias);

    const double inputs[] = {2, 5, 8};
    for (size_t i = 0; i < sizeof inputs / sizeof inputs[0]; ++i) {
        printf("x=%.0f: probability=%.4f, class=%d\n", inputs[i],
               logistic_regression_predict_proba(&classifier, inputs[i]),
               logistic_regression_predict(&classifier, inputs[i]));
    }

    return 0;
}
