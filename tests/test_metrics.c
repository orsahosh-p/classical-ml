#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "core/metrics.h"

int main(void)
{
    double actual[] = {3, 5, 7};
    double predicted[] = {2, 3, 5};

    double mse = mean_squared_error(actual, predicted, 3);

    assert(fabs(mse - 3.0) < 1e-9);
    assert(isnan(mean_squared_error(NULL, predicted, 3)));
    assert(isnan(mean_squared_error(actual, predicted, 0)));

    printf("All MSE tests passed!\n");

    return 0;
}