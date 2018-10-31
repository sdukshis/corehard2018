#pragma once

#include "classifier.h"

namespace mnist {

#include <tensorflow/c/c_api.h>

class TfClassifier: public Classifier {
public:
    TfClassifier(const std::string& modelpath,
                 const int width,
                 const int height);

    TfClassifier(const TfClassifier&) = delete;

    TfClassifier& operator=(const TfClassifier&) = delete;

    size_t num_classes() const override;

    size_t predict(const features_t&) const override;

    probas_t predict_proba(const features_t&) const override;

protected:
    using tf_graph = std::unique_ptr<TF_Graph, decltype(&TF_DeleteGraph)>;
    using tf_buffer = std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)>;
    using tf_import_graph_def_options = std::unique_ptr<TF_ImportGraphDefOptions, decltype(&TF_DeleteImportGraphDefOptions)>;
    using tf_status = std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)>;
    using tf_session_options = std::unique_ptr<TF_SessionOptions, decltype(&TF_DeleteSessionOptions)>;
    using tf_tensor = std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)>;

protected:
    static tf_buffer read_model(const std::string& modelpath);


    tf_graph graph_{TF_NewGraph(), TF_DeleteGraph};
    tf_buffer graph_def_{nullptr, TF_DeleteBuffer};
    TF_Operation* input_op_ = nullptr;
    TF_Operation* output_op_ = nullptr;
    int width_;
    int height_;
};

}