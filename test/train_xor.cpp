#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include <cmath>
#include "Linear.cuh"

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1 - s);
}

void train_xor(int epochs = 1000) {
    const int input_dim = 2;
    const int hidden_dim = 4;
    const int output_dim = 1;
    const int batch_size = 4;

    float inputs[batch_size * input_dim] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };

    float targets[batch_size] = {0.0f, 1.0f, 1.0f, 0.0f};

    float hidden[batch_size * hidden_dim];
    float outputs[batch_size * output_dim];

    float grad_output[batch_size * output_dim];
    float grad_hidden[batch_size * hidden_dim];

    Linear layer1(input_dim, hidden_dim);
    Linear layer2(hidden_dim, output_dim);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward
        layer1.forward(inputs, hidden, batch_size);
        layer2.forward(hidden, outputs, batch_size);

        // Compute grad_output = dLoss/dOutput (MSE loss)
        float loss = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            float pred = sigmoid(outputs[i]);
            float label = targets[i];
            loss += 0.5f * (pred - label) * (pred - label);
            grad_output[i] = (pred - label) * sigmoid_derivative(outputs[i]);
        }

        // Backward
        layer2.backward(hidden, grad_output, batch_size);
        layer1.backward(inputs, hidden, batch_size);

        if (epoch % 100 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
        }
    }

    std::cout << "Training complete. Final predictions:" << std::endl;
    layer1.forward(inputs, hidden, batch_size);
    layer2.forward(hidden, outputs, batch_size);
    for (int i = 0; i < batch_size; ++i) {
        std::cout << "Input: (" << inputs[i * 2] << ", " << inputs[i * 2 + 1]
                  << ") -> Predicted: " << sigmoid(outputs[i])
                  << ", Target: " << targets[i] << std::endl;
    }
}

int main() {
    train_xor();
    return 0;
}
