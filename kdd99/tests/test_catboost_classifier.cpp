#include <fstream>
#include <iostream>
#include <limits>

#include <gtest/gtest.h>

#include <kdd99/catboost_classifier.h>
#include <helpers.h>

using kdd99::CatboostClassifier;
using std::clog;

TEST(CatboostClassifier, compare_to_python) {
    auto predictor = CatboostClassifier{"../train/model.cbm"};

    auto features = CatboostClassifier::features_t{};

    double y_pred_expected = 0.0;

    std::ifstream test_data{"../train/test_data_catboost.csv"};
    for (;;) {
        test_data >> y_pred_expected;
        if (!read_features(test_data, features)) {
            break;
        }
        auto y_pred = predictor.predict_proba(features);
        EXPECT_NEAR(y_pred_expected, y_pred, std::numeric_limits<float>::epsilon());
    }
}
