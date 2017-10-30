#ifndef __CUDA_DYNAMIC_RUNTIME_H
#define __CUDA_DYNAMIC_RUNTIME_H

#include <cuda.h>

extern "C" {

void PrepareDynamicCuda(void);

void DestroyDynamicCuda(void);

cudaError_t
dynamicCudaConfigureCall(dim3 gridDim, dim3 blockDim,
                         size_t sharedMem, cudaStream_t stream);

cudaError_t
dynamicCudaSetupArgument(const void* arg, size_t size, size_t offset);

cudaError_t dynamicCudaLaunchByName(const char* name);

void dynamicCudaRegisterFunction(void *hostFun, const char *name);

cudaError_t dynamicCudaLaunch(void *hostFun);

}  // extern C

#endif
