#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda_runtime.h>
#include <string>
#include <vector>

void checkCudaError(cudaError_t err, const std::string &msg);
void checkCudaMallocError(const std::string &entity, cudaError_t cuda_err, int device_id);

struct GpuDeleter {
    void operator()(float *ptr) const {
        if (ptr) {
            // Consider adding error checking for cudaFree
            cudaFree(ptr);
        }
    }
};

struct CpuDeleter {
    void operator()(float *ptr) const {
        if (ptr) {
            delete[] ptr;
        }
    }
};

#endif // CUDA_HELPER_H
