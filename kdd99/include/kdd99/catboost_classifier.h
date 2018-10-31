#pragma once

#include "classifier.h"

#include <catboost/model_calcer_wrapper.h>

namespace kdd99 {

class CatboostClassifier: public BinaryClassifier {
public:
    CatboostClassifier(const std::string& modepath);

    ~CatboostClassifier() override;
    
    float predict_proba(const features_t&) const override;

private:
    ModelCalcerHandle* model_; 
};

}