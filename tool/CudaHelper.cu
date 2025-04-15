#include <CudaHelper.cuh>
#include <iostream>

void checkCudaError(cudaError_t err, const std::string &msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCudaMallocError(const std::string &entity, cudaError_t err, int device_id) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl;
        std::cerr << "CUDA error: Failed to allocate memory for " << entity << " on device: " << device_id << " - "
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA malloc failed");
    }
}