#include "CudaHelper.cuh"
#include "Tensor.cuh"
#include <iostream>
#include <stdexcept>

Tensor::Tensor(const std::vector<int> &shape_vec, bool requires_grad, int dev_id)
    : shape_vec(shape_vec), requires_grad(requires_grad), device_id(dev_id), is_device(false) {
    strides = calculate_strides(shape_vec);
    num_of_elements = calculate_size(shape_vec);

    cudaError_t cuda_err;
    cudaSetDevice(device_id);
    cuda_err = cudaMalloc((void **)&data, num_of_elements * sizeof(float));
    checkCudaMallocError("data", cuda_err, device_id);

    if (requires_grad) {
        cuda_err = cudaMalloc((void **)&grad, num_of_elements * sizeof(float));
        checkCudaMallocError("grad", cuda_err, device_id);
    }
}

// @todo: Tensor constructor to create a Tensor from existing host data (copies data to device)

// @todo: Tensor constructor to create a Tensor from existing device data

void Tensor::zero_grad(cudaStream_t stream) {
    if (requires_grad && grad) {
        cudaError_t cuda_err;
        cudaSetDevice(device_id);
        cuda_err = cudaMemsetAsync(grad.get(), 0, num_of_elements * sizeof(float), stream);
        checkCudaMallocError("grad", cuda_err, device_id);
    }
}

void Tensor::backward(cudaStream_t stream) {
    if (!requires_grad)
        return;

    cudaError_t cuda_err;
    cudaSetDevice(device_id);
    cuda_err = cudaMemsetAsync(grad.get(), 1, num_of_elements * sizeof(float), stream);
    checkCudaMallocError("grad", cuda_err, device_id);

    for (const auto &dep : dependencies) {
        dep.backward_fn(grad.get(), stream);
    }

    // void backward() {
    //     if (!requires_grad)
    //         return;
    //     std::fill(grad, grad + size(), 1.0f);
    //     for (auto &dep : dependencies) {
    //         dep.backward_fn(grad);
}
