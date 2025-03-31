#pragma once

class Linear {
public:
    float* d_weight;
    float* d_bias;
    float* d_grad_weight;
    float* d_grad_bias;
    int in_features, out_features;

    Linear(int in_f, int out_f);
    void forward(float* input, float* output, int batch_size);
    void backward(float* input, float* grad_output, int batch_size);
    ~Linear();
};
