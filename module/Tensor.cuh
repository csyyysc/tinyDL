#ifndef TENSOR_H
#define TENSOR_H

#include "CudaHelper.cuh"
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

class Tensor;

struct Dependency {
    std::shared_ptr<Tensor> tensor;
    std::function<void(const float *grad_out, cudaStream_t)> backward_fn;

    Dependency(std::shared_ptr<Tensor> t, std::function<void(const float *, cudaStream_t)> fn)
        : tensor(std::move(t)), backward_fn(std::move(fn)) {
    }
};

class Tensor : public std::enable_shared_from_this<Tensor> {
  private:
    int calculate_size(const std::vector<int> &shape_vec) const {
        if (shape_vec.empty()) {
            return 0;
        }
        return std::accumulate(shape_vec.begin(), shape_vec.end(), 1, std::multiplies<int>());
    }

    // N-dimensional strides calculation
    //   Ex: shape = {2, 3, 4}, the strides would be {12, 4, 1}
    std::vector<int> calculate_strides(const std::vector<int> &shape_vec) const {
        std::vector<int> strides_vec(shape_vec.size());
        if (!shape_vec.empty()) {
            strides_vec.back() = 1;
            for (int i = shape_vec.size() - 2; i >= 0; --i) {
                strides_vec[i] = strides_vec[i + 1] * shape_vec[i + 1];
            }
        }
        return strides_vec;
    }

  public:
    std::shared_ptr<float> data;
    std::shared_ptr<float> grad; // @note Allocated only if requires_grad is true

    int num_of_elements;
    int device_id;
    bool is_device;
    bool requires_grad;

    std::vector<int> strides;
    std::vector<int> shape_vec;
    std::vector<Dependency> dependencies;

    // --- Constructors ---
    Tensor(const std::vector<int> &shape_vec, bool requires_grad = false, int dev_id = 0);

    // Constructor to create a Tensor from existing host data (copies data to device)
    Tensor(const float *host_data, const std::vector<int> &shape_vec, bool requires_grad = false, int dev_id = 0);

    // Constructor to create a Tensor from existing device data
    // Note: Ensure the provided ptr was allocated with cudaMalloc and is not managed elsewhere.
    Tensor(float *device_ptr, const std::vector<int> &shape_vec, bool requires_grad = false, int dev_id = 0);

    // --- Methods ---
    int size() const {
        return num_of_elements;
    }
    void zero_grad(cudaStream_t stream = 0);
    void backward(cudaStream_t stream = 0);

    // --- Data Movement ---
    std::vector<float> host_data() const;
    void to_device(int target_device_id = -1, cudaStream_t stream = 0);
    void to_host(cudaStream_t stream = 0);
    void synchronize(cudaStream_t stream = 0);

    // --- Device Management ---
    void set_device(int new_device_id);

    // --- Autograd ---
    std::shared_ptr<Tensor> shared() {
        return shared_from_this();
    }

    // --- Operations ---
    std::shared_ptr<Tensor> add(const Tensor &other, cudaStream_t stream = 0) const;
    std::shared_ptr<Tensor> sub(const Tensor &other, cudaStream_t stream = 0) const;
    std::shared_ptr<Tensor> mul(const Tensor &other, cudaStream_t stream = 0) const;
    std::shared_ptr<Tensor> matmul(const Tensor &other, cudaStream_t stream = 0) const;
};

#endif // TENSOR_H
