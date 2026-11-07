#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "core/metrics.h"

static void test_binary_cross_entropy(void)
{
    const double labels[] = {0.0, 1.0};
    const double good[] = {0.1, 0.9};
    const double bad[] = {0.9, 0.1};
    const double uncertain[] = {0.5, 0.5};
    const double wrong[] = {1.0, 0.0};

    assert(fabs(binary_cross_entropy(labels, good, 2) - 0.1053605156578263) < 1e-12);
    assert(fabs(binary_cross_entropy(labels, bad, 2) - 2.302585092994046) < 1e-12);
    assert(fabs(binary_cross_entropy(labels, uncertain, 2) - log(2.0)) < 1e-12);
    assert(binary_cross_entropy(labels, labels, 2) == 0.0);
    assert(binary_cross_entropy(labels, wrong, 2) == INFINITY);
    assert(binary_cross_entropy(labels, wrong, 1) == INFINITY);
    assert(binary_cross_entropy(labels + 1, wrong + 1, 1) == INFINITY);
    assert(isnan(binary_cross_entropy(NULL, good, 2)));
    assert(isnan(binary_cross_entropy(labels, NULL, 2)));
    assert(isnan(binary_cross_entropy(labels, good, 0)));

    const double invalid[] = {-0.1, 1.1, NAN, INFINITY, -INFINITY};
    for (size_t i = 0; i < sizeof invalid / sizeof invalid[0]; ++i) {
        assert(isnan(binary_cross_entropy(labels, &invalid[i], 1)));
        assert(isnan(binary_cross_entropy(&invalid[i], good, 1)));
    }
    const double fractional_label = 0.5;
    assert(isnan(binary_cross_entropy(&fractional_label, good, 1)));

    /* Invalid input still wins over an earlier infinite loss. */
    const double wrong_then_invalid[] = {1.0, NAN};
    assert(isnan(binary_cross_entropy(labels, wrong_then_invalid, 2)));

    const double tiny_probability = 1e-20;
    assert(fabs(binary_cross_entropy(labels, &tiny_probability, 1)
                / tiny_probability - 1.0) < 1e-12);
}

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

    test_binary_cross_entropy();
    printf("All metrics tests passed!\n");

    return 0;
}
