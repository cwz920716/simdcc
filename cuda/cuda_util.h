#ifndef CUDA_UTIL_H_
#define CUDA_UTIL_H_

#include "glog/logging.h"

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: grid stride looping
#define CUDA_GRID_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: block stride looping
#define CUDA_BLOCK_LOOP(i, n) \
  for (int i = threadIdx.x; \
       i < (n); \
       i += blockDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

#define SWAP(x, y, T) do { T SWAP = x; x = y; y = SWAP; } while (0)

#endif  // CUDA_COMMON_H_
