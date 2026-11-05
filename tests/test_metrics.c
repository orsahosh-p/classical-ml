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

    const int actual_classes[] = {0, 0, 1, 1};
    const int partial_classes[] = {0, 1, 1, 1};
    const int wrong_classes[] = {1, 1, 0, 0};

    assert(accuracy_score(actual_classes, partial_classes, 4) == 0.75);
    assert(accuracy_score(actual_classes, actual_classes, 4) == 1.0);
    assert(accuracy_score(actual_classes, wrong_classes, 4) == 0.0);
    assert(isnan(accuracy_score(NULL, partial_classes, 4)));
    assert(isnan(accuracy_score(actual_classes, NULL, 4)));
    assert(isnan(accuracy_score(actual_classes, partial_classes, 0)));

    /* Class labels need not be binary. */
    const int multiclass_actual[] = {2, 4, 7};
    const int multiclass_predicted[] = {2, 7, 7};
    assert(fabs(accuracy_score(multiclass_actual, multiclass_predicted, 3)
                - 2.0 / 3.0) < 1e-9);

    printf("All metrics tests passed!\n");

    return 0;
}
