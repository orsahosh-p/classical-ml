#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "algorithms/linear_regression.h"

int main(void)
{
    const double x[] = {1.0, 2.0};
    const double y[] = {3.0, 5.0};
    LinearRegression model = {0.0, 0.0};

    /* Initial predictions are zero, so errors are -3 and -5.
     * dw = (2*(-3)*1 + 2*(-5)*2) / 2 = -13
     * db = (2*(-3) + 2*(-5)) / 2 = -8
     * With learning_rate = 0.1: weight = 1.3, bias = 0.8.
     */
    linear_regression_fit(&model, x, y, 2, 0.1, 1);

    assert(fabs(model.weight - 1.3) < 1e-9);
    assert(fabs(model.bias - 0.8) < 1e-9);
    assert(fabs(linear_regression_predict(&model, x[0]) - 2.1) < 1e-9);
    assert(fabs(linear_regression_predict(&model, x[1]) - 3.4) < 1e-9);

    printf("Linear regression one-epoch test passed!\n");
    return 0;
}
