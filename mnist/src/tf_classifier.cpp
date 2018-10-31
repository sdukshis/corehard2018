#include <mnist/tf_classifier.h>
#include <sstream>
#include <memory>
#include <vector>
#include <fstream>

using mnist::TfClassifier;

static void delete_buffer(void* data, size_t size) {
    delete[] reinterpret_cast<char*>(data);
}

static void dummy_deleter(void* data, size_t length, void* arg) {

}

TfClassifier::TfClassifier(const std::string& modelpath,
                           const int width,
                           const int height)
    : width_{width}
    , height_{height}
    , graph_def_{read_model(modelpath)} {
  
    tf_status status{TF_NewStatus(), TF_DeleteStatus};   
    tf_import_graph_def_options opts{TF_NewImportGraphDefOptions(), TF_DeleteImportGraphDefOptions};

    TF_GraphImportGraphDef(graph_.get(), graph_def_.get(), opts.get(), status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
        std::stringstream ss;
        ss << " Unable to import graph from '" << modelpath << "': " << TF_Message(status.get());
        throw std::invalid_argument{ss.str()};
    }

    size_t pos = 0;
    input_op_ = TF_GraphNextOperation(graph_.get(), &pos);

    TF_Operation* last_op = input_op_, *next_op = nullptr;
    while ((next_op = TF_GraphNextOperation(graph_.get(), &pos)) != nullptr) {
        last_op = next_op;
    }
    output_op_ = last_op;
}


size_t TfClassifier::num_classes() const {
    return 10;
}

size_t TfClassifier::predict(const features_t& feat) const {
    auto proba = predict_proba(feat);
    auto argmax = std::max_element(proba.begin(), proba.end());
    return std::distance(proba.begin(), argmax);
}

TfClassifier::probas_t TfClassifier::predict_proba(const features_t& feat) const {
    assert(width_ * height_ == feat.size());

    std::vector<TF_Output> inputs;
    std::vector<TF_Tensor*> input_values;

    TF_Output input_opout = {input_op_, 0};
    inputs.push_back(input_opout);

    // Create variables to store the size of the input and output variables
    const int num_bytes_in = width_ * height_ * sizeof(float);
    const int num_bytes_out = 10 * sizeof(float);

    // Set input dimensions - this should match the dimensionality of the input in
    // the loaded graph, in this case it's three dimensional.
    int64_t in_dims[] = {1, width_, height_, 1};
    int64_t out_dims[] = {1, 10};

    tf_tensor input{TF_NewTensor(TF_FLOAT, in_dims, 4, reinterpret_cast<void*>(const_cast<float*>(feat.data())), num_bytes_in, &dummy_deleter, 0),
                    TF_DeleteTensor};
    input_values.push_back(input.get());

    std::vector<TF_Output> outputs;
    TF_Output output_opout = {output_op_, 0};
    outputs.push_back(output_opout);

    // Create TF_Tensor* vector
    std::vector<TF_Tensor*> output_values(outputs.size(), nullptr);

    // Similar to creating the input tensor, however here we don't yet have the
    // output values, so we use TF_AllocateTensor()
    tf_tensor output_value{TF_AllocateTensor(TF_FLOAT, out_dims, 2, num_bytes_out), TF_DeleteTensor};
    output_values.push_back(output_value.get());

    tf_status status{TF_NewStatus(), TF_DeleteStatus};
    tf_session_options session_opts{TF_NewSessionOptions(), TF_DeleteSessionOptions};
    TF_Session* session = TF_NewSession(graph_.get(), session_opts.get(), status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
        std::stringstream ss;
        ss << "Unable to create session from graph: " << TF_Message(status.get());
        throw std::runtime_error{ss.str()};
    } 

    TF_SessionRun(session, nullptr,
                &inputs[0], &input_values[0], inputs.size(),
                &outputs[0], &output_values[0], outputs.size(),
                nullptr, 0, nullptr, status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
        std::stringstream ss;
        ss << "Unable to run session from graph: " << TF_Message(status.get());
        throw std::runtime_error{ss.str()};
    }

    probas_t probas;
    float* out_vals = static_cast<float*>(TF_TensorData(output_values[0]));
    for (int i = 0; i < 10; ++i) {
        probas.push_back(*out_vals++);
    }

    TF_CloseSession(session, status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
        std::stringstream ss;
        ss << "Unable to run session from graph: " << TF_Message(status.get());
        throw std::runtime_error{ss.str()};
    }
    TF_DeleteSession(session, status.get());   
    return probas;
}

TfClassifier::tf_buffer TfClassifier::read_model(const std::string& modelpath) {
    std::ifstream model{modelpath};
    model.seekg(0, std::ios_base::seekdir::end);
    size_t size = model.tellg();
    model.seekg(0);

    char* data = new char[size];
    model.read(data, size);

    tf_buffer buf{TF_NewBuffer(), TF_DeleteBuffer};                                                        
    buf->data = data;
    buf->length = size;                                                                    
    buf->data_deallocator = delete_buffer;                                                    
    return buf; 
}