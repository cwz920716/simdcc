#include "glog/logging.h"
#include "synthesis/base.h"
#include "synthesis/iteratable.h"
#include "synthesis/expand.h"
#include "synthesis/gbar.cuh"

__global__ void type_test(glang::DynArray<float> out, glang::Slice S) {
  if (threadIdx.x == 0) {
    printf("[%d:%d:%d]\n", S.start(), S.end(), S.step());
    printf("%p[%d:%d:%d]\n", out.data(), out.start(), out.end(), out.step());
  }

  auto f = [&] DEVICE (float &x) {
    x = threadIdx.x;
    printf("x[%p]=%f\n", &x, x);
  };

  glang::Expand<float, glang::kBlockScope> expand;
  expand(out, f);
}

int main(void) {
  const int kDataLen = 64;
  glang::Slice S0(0, kDataLen);
  float* device_out;
  cudaMalloc(&device_out, kDataLen * sizeof(float));
  glang::DynArray<float> Out(kDataLen, device_out);

  type_test<<<1, kDataLen + 1>>>(Out, S0);
  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  printf("test end.\n");
  return 0;
}
