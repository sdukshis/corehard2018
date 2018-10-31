#include <fstream>
#include <iostream>
#include <limits>

#include <gtest/gtest.h>

#include <kdd99/logreg_classifier.h>
#include <helpers.h>

using kdd99::LogregClassifier;
using std::clog;

TEST(LogregClassifier, compare_to_python) {
    std::ifstream istream{"../train/logreg_coef.txt"};
    auto coef = read_vector(istream);
    istream.close();

    auto predictor = LogregClassifier{coef};

    auto features = LogregClassifier::features_t{};

    double y_pred_expected = 0.0;

    std::ifstream test_data{"../train/test_data_logreg.csv"};
    for (;;) {
        test_data >> y_pred_expected;
        if (!read_features(test_data, features)) {
            break;
        }
        auto y_pred = predictor.predict_proba(features);
        EXPECT_NEAR(y_pred_expected, y_pred, std::numeric_limits<float>::epsilon());
    }
}
