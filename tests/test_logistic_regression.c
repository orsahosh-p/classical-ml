#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "algorithms/logistic_regression.h"

static void test_sigmoid(void)
{
    assert(sigmoid(0.0) == 0.5);
    assert(fabs(sigmoid(1.0) - 0.7310585786300049) < 1e-12);
    assert(fabs(sigmoid(-1.0) - 0.2689414213699951) < 1e-12);
    assert(sigmoid(-1000.0) == 0.0);
    assert(sigmoid(1000.0) == 1.0);
}

static void test_predictions(void)
{
    LogisticRegression model = {0.0, 0.0};
    assert(logistic_regression_predict_proba(&model, -100.0) == 0.5);
    assert(logistic_regression_predict_proba(&model, 100.0) == 0.5);
    assert(logistic_regression_predict(&model, 0.0) == 1);

    model = (LogisticRegression){2.0, -1.0};
    assert(fabs(logistic_regression_predict_proba(&model, 0.0)
                - 0.2689414213699951) < 1e-12);
    assert(fabs(logistic_regression_predict_proba(&model, 1.0)
                - 0.7310585786300049) < 1e-12);
    assert(logistic_regression_predict(&model, 0.0) == 0);
    assert(logistic_regression_predict(&model, 0.5) == 1);
    assert(logistic_regression_predict(&model, 1.0) == 1);
}

static void test_one_epoch(void)
{
    const double x[] = {1.0, 2.0};
    const double y[] = {0.0, 1.0};
    LogisticRegression model = {0.0, 0.0};

    /* Probabilities = {0.5, 0.5}, errors = {0.5, -0.5}.
     * Mean gradients: dw = -0.25, db = 0.
     */
    logistic_regression_fit(&model, x, y, 2, 0.1, 1);
    assert(fabs(model.weight - 0.025) < 1e-12);
    assert(fabs(model.bias) < 1e-12);

    /* A nonzero starting model checks that fit preserves and updates
     * existing parameters, including the bias. Here z = 0, error = -0.5.
     */
    const double single_x[] = {1.0};
    const double single_y[] = {1.0};
    model = (LogisticRegression){1.0, -1.0};
    logistic_regression_fit(&model, single_x, single_y, 1, 0.1, 1);
    assert(fabs(model.weight - 1.05) < 1e-12);
    assert(fabs(model.bias + 0.95) < 1e-12);
}

static void test_invalid_arguments(void)
{
    const double x[] = {1.0};
    const double y[] = {1.0};
    const double invalid_rates[] = {0.0, -0.1, NAN, INFINITY, -INFINITY};
    LogisticRegression model = {0.75, -0.25};

    for (size_t i = 0; i < sizeof invalid_rates / sizeof invalid_rates[0]; ++i) {
        logistic_regression_fit(&model, x, y, 1, invalid_rates[i], 10);
        assert(model.weight == 0.75 && model.bias == -0.25);
    }

    logistic_regression_fit(NULL, x, y, 1, 0.1, 10);
    logistic_regression_fit(&model, NULL, y, 1, 0.1, 10);
    assert(model.weight == 0.75 && model.bias == -0.25);
    logistic_regression_fit(&model, x, NULL, 1, 0.1, 10);
    assert(model.weight == 0.75 && model.bias == -0.25);
    logistic_regression_fit(&model, x, y, 0, 0.1, 10);
    assert(model.weight == 0.75 && model.bias == -0.25);
    logistic_regression_fit(&model, x, y, 1, 0.1, 0);
    assert(model.weight == 0.75 && model.bias == -0.25);
}

static void test_training(void)
{
    const double x[] = {-2.0, -1.0, 1.0, 2.0};
    const double y[] = {0.0, 0.0, 1.0, 1.0};
    LogisticRegression model = {0.0, 0.0};

    logistic_regression_fit(&model, x, y, 4, 0.1, 1000);

    for (size_t i = 0; i < 4; ++i) {
        double probability = logistic_regression_predict_proba(&model, x[i]);
        assert(isfinite(probability));
        assert(probability >= 0.0 && probability <= 1.0);
        assert(logistic_regression_predict(&model, x[i]) == (int)y[i]);
        assert(y[i] == 1.0 ? probability > 0.95 : probability < 0.05);
    }
}

int main(void)
{
    test_sigmoid();
    test_predictions();
    test_one_epoch();
    test_invalid_arguments();
    test_training();
    puts("All logistic regression tests passed!");
    return 0;
}
