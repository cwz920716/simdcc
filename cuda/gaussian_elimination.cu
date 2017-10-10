#include <iostream>
#include <cstdio>
#include "glog/logging.h"
#include "cuda_util.h"

// #include "gpuvm_rt.h"

// One block handles one row
__global__ void ForwardElimination(int p, int n,
                                   const float *A, const float *b,
                                   float *U, float *y) {
  int bid = blockIdx.x;  // Rows
  int tid = threadIdx.x;

  __shared__ float sharedA[6];

  // before_branch_handler(nullptr);
  // atomicExch(&U[0], 1.0f);

  if (bid < p) {
    CUDA_BLOCK_LOOP(c, n) {
      U[bid * n + c] = A[bid * n + c];
    }
    if (tid == 0) {
      y[bid] = b[bid];
    }
    return;
  }

  float pivot_down_scale = 1.0f / A[p * n + p];

  if (bid == p) {
    CUDA_BLOCK_LOOP(c, n) {
      sharedA[c] = A[p * n + c];
    }
    __syncthreads();

    CUDA_BLOCK_LOOP(c, n) {
      float U_pc = sharedA[c] * pivot_down_scale;
      float y_p = b[p] * pivot_down_scale;
      U[bid * n + c] = U_pc;
      if (tid == 0) {
        y[bid] = y_p;
      }
    }
    return;
  }

  float pivot_up_scale = -A[bid * n + p];
  // printf("%f\n", pivot_up_scale);
  CUDA_BLOCK_LOOP(c, n) {
    float U_pc = A[p * n + c] * pivot_down_scale;
    float y_p = b[p] * pivot_down_scale;
    float u_ = A[bid * n + c] + pivot_up_scale * U_pc;
    float y_ = b[bid] + pivot_up_scale * y_p;
    // printf("%f %f %f %f\n", U_pc, y_p, u_, y_);
    U[bid * n + c] = u_;
    if (tid == 0) {
      y[bid] = y_;
    }
  }

  return;
}

void printArray2D(float *A, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f ", A[i * n + j]);
    }
    printf(";\n");
  }

  printf("\n");
}

int main(void) {
  int n = 6;

  float hA[] = {
    1.00, 0.00, 0.00,  0.00,  0.00, 0.00,
    1.00, 0.63, 0.39,  0.25,  0.16, 0.10,
    1.00, 1.26, 1.58,  1.98,  2.49, 3.13,
    1.00, 1.88, 3.55,  6.70, 12.62, 23.80,
    1.00, 2.51, 6.32, 15.88, 39.90, 100.28,
    1.00, 3.14, 9.87, 31.01, 97.41, 306.02
  };
	float hb[] = { -0.01, 0.61, 0.91, 0.99, 0.60, 0.02 };

  float *A, *U, *b, *y;
  const int a_size = n * n * sizeof(float);
  const int b_size = n * sizeof(float);
  CUDA_CHECK(cudaMalloc(&A, a_size));
  CUDA_CHECK(cudaMalloc(&U, a_size));
  CUDA_CHECK(cudaMalloc(&b, b_size));
  CUDA_CHECK(cudaMalloc(&y, b_size));

  cudaMemcpy(A, hA, a_size, cudaMemcpyHostToDevice);
  cudaMemcpy(b, hb, b_size, cudaMemcpyHostToDevice);

  float *pA = A, *pb = b, *pU = U, *py = y;
  printArray2D(hA, n);
  for (int p = 0; p < n; p++) {
    ForwardElimination<<< 6, 6 >>>(p, n, pA, pb, pU, py);
    SWAP(pA, pU, float *);
    SWAP(pb, py, float *);
    cudaMemcpy(hA, pA, a_size, cudaMemcpyDeviceToHost);
    printArray2D(hA, n);
  }

  cudaMemcpy(hA, pA, a_size, cudaMemcpyDeviceToHost);
  printArray2D(hA, n);

  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(U));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(y));

  return 0;
}
