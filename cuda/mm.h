#pragma once

namespace MM {

#define mm_Kb (16)

// Expect mm_Kb*mm_Kb block

__device__ float blas_microkernel(float *Ab, float *Bb) {
  // int Cid = threadIdx.x * mm_Kb + threadIdx.y;
  float c = 0.0f;
  for (int i = 0; i < mm_Kb; i++) {
    int Aid = threadIdx.x * mm_Kb + i;
    int Bid = i * mm_Kb + threadIdx.y;
    c += Ab[Aid] * Bb[Bid];
  }
  return c;
}

__global__ void blas_macrokernel(int K, float *A, float *B, float *C) {
  __shared__ float Apack[mm_Kb*mm_Kb], Bpack[mm_Kb*mm_Kb];
  int Arow = blockIdx.x * mm_Kb;
  int Bcol = blockIdx.y * mm_Kb;
  int Crow = blockIdx.x * mm_Kb;
  int Ccol = blockIdx.y * mm_Kb;
  int tid_blk = threadIdx.x * mm_Kb + threadIdx.y;
  float c = 0.0f;
  int Kbs = K / mm_Kb;
  for (int Kid = 0; Kid < Kbs; Kid++) {
    int Acol = Kid * mm_Kb;
    int Brow = Kid * mm_Kb;
    Apack[tid_blk] = A[(Arow + threadIdx.x) * K + (Acol + threadIdx.y)];
    Bpack[tid_blk] = B[(Brow + threadIdx.x) * K + (Bcol + threadIdx.y)];
    __syncthreads();
    c += blas_microkernel(Apack, Bpack);
    __syncthreads();
  }
  C[(Crow + threadIdx.x) * K + (Ccol + threadIdx.y)] += c;
}

__global__ void mm(int K, float *A, float *B, float *C) {
  int rowId = blockIdx.x;
  int colId = threadIdx.x;
  for (int i = 0; i < K; i++) {
    C[rowId * K + colId] += A[rowId * K + i] * B[i * K + colId];
  }
}

TEST(MM, Prepare) {
  const int K = mm_Kb * mm_Kb;
  device_vector<float> A(K*K, 1.0f);
  device_vector<float> B(K*K, 1.0f);
  srand(0);
  for (int i = 0; i < K*K; i++) {
    A[i] = rand() * 1.0f / RAND_MAX;
    B[i] = rand() * 1.0f / RAND_MAX;
  }
  
  device_vector<float> C(K*K, 0.0f);
  device_vector<float> C_ref(K*K, 0.0f);
  dim3 grid(K/mm_Kb, K/mm_Kb, 1);
  dim3 block(mm_Kb, mm_Kb, 1);
  // for (int i = 0; i < K / mm_Kb; i++) {
  blas_macrokernel <<< grid, block >>> (K, PTR(A), PTR(B), PTR(C));
  // }
  mm <<< K, K >>> (K, PTR(A), PTR(B), PTR(C_ref));
  for (int i = 0; i < K*K; i++) {
    CHECK_LE(fabs(C[i] - C_ref[i]), 0.0002f) << "at iter " << i;
  }
}

TEST(MM, Time_nothing) {
  const int K = 1024;
  device_vector<float> A(K*K, 1.0f);
  device_vector<float> B(K*K, 1.0f);
  device_vector<float> C_ref(K*K, 0.0f);
  srand(0);
  for (int i = 0; i < mm_Kb; i++) {
    A[i] = rand() * 1.0f / RAND_MAX;
    B[i] = rand() * 1.0f / RAND_MAX;
  }
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
  mm <<< K, K >>> (K, PTR(A), PTR(B), PTR(C_ref));
  CUDA_CHECK( cudaGetLastError() );
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
std::cout << "Time: " << milliseconds << " (ms)." << endl;
}

TEST(MM, Time_tiled) {
  const int K = 1024;
  device_vector<float> A(K*K, 1.0f);
  device_vector<float> B(K*K, 1.0f);
  device_vector<float> C(K*K, 0.0f);
  srand(0);
  for (int i = 0; i < mm_Kb; i++) {
    A[i] = rand() * 1.0f / RAND_MAX;
    B[i] = rand() * 1.0f / RAND_MAX;
  }
  dim3 grid(K/mm_Kb, K/mm_Kb, 1);
  dim3 block(mm_Kb, mm_Kb, 1);
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
  // for (int i = 0; i < K / mm_Kb; i++) {
  blas_macrokernel <<< grid, block >>> (K, PTR(A), PTR(B), PTR(C));
  // }
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
std::cout << "Time: " << milliseconds << " (ms)." << endl;
}

}  // namespace MM


