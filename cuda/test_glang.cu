#include "glog/logging.h"
#include "synthesis/base.h"
#include "synthesis/iteratable.h"
#include "synthesis/expand.h"
#include "synthesis/enlist.h"
#include "synthesis/for.h"
#include "synthesis/gbar.cuh"
#include "synthesis/pool.h"
#include "synthesis/gather.h"
#include "cuda_util.h"

using namespace glang;

__global__ void type_test(glang::DynArray<float> out, glang::Slice S) {
  if (threadIdx.x == 0) {
    printf("[%d:%d:%d]\n", S.start(), S.end(), S.step());
    // printf("%p[%d:%d:%d]\n", out.data(), out.start(), out.end(), out.step());
  }

  auto f = [&] DEVICE (float &x) {
    x = threadIdx.x;
    // printf("x[%p]=%f\n", &x, x);
  };

  glang::Expand<float, glang::kBlockScope> expand;
  expand(out, f);

  glang::Parfor<float, glang::kBlockScope> parfor;
  parfor(out, f);
}

template<int NUM_THREADS>
__global__ void spmv_test(glang::DynArray<int> R, glang::DynArray<float> M) {
  typedef glang::BlockPool<glang::kBlockScope, NUM_THREADS> BlockPool;
  __shared__ typename BlockPool::TempStorage temp_storage;
  BlockPool pool(temp_storage);

  glang::Slice RSlice(R.start(), R.end());
  auto Rc = R;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  auto visit = [&] DEVICE (int i) {
    printf("[%d] visit col %d\n", tid, i);
  };

  auto f = [&] DEVICE (int i) {
    glang::Slice S(Rc[i], Rc[i+1]);
    int r, t;
    pool.claim(Rc[i + 1] - Rc[i], r, t);
    printf("[%d] visit row %d[%d:%d:%d] alloc resources at [%d<-%d]\n",
        tid, i, S.start(), S.end(), S.step(), r, t);

    glang::ForEach<int> for_each;
    for_each(S, visit);
  };

  glang::Expand<int, glang::kDeviceScope> expand;
  expand(RSlice, f);
}

template<int NUM_THREADS>
__global__ void warp_enlist_test(glang::DynArray<int> R, glang::DynArray<float> M) {
  typedef WarpEnlist<Slice, NUM_THREADS> Enlist;
  __shared__ typename Enlist::TempStorage temp_storage;
  Enlist enlist(temp_storage);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  auto visit = [&] DEVICE (int i) {
    printf("[%d] visit col %d\n", tid, i);
  };

  auto warp_visit = [&] DEVICE (int leader, glang::Slice S) {
    if (lane_id() == 0) {
      printf("Warp[%d] visit Lane %d [%d:%d:%d]\n", glang::warp_id(), leader, S.start(), S.end(), S.step());
    }
    glang::Parfor<int, glang::kWarpScope> parfor;
    parfor(S, visit);
  };

  glang::Slice RSlice(R.start(), R.end());
  auto Rc = R;

  auto f = [&] DEVICE (int i) {
    glang::Slice S(Rc[i], Rc[i+1]);
    enlist.until(S.size() > 0, S, warp_visit);
  };

  glang::Expand<int, glang::kDeviceScope> expand;
  expand(RSlice, f);
}

template<int NUM_THREADS>
__global__ void block_enlist_test(glang::DynArray<int> R, glang::DynArray<float> M) {
  typedef BlockEnlist<Slice, NUM_THREADS> Enlist;
  __shared__ typename Enlist::TempStorage temp_storage;
  Enlist enlist(temp_storage);

  // int tid = threadIdx.x + blockIdx.x * blockDim.x;

  auto visit = [&] DEVICE (int i) {
    // printf("[%d] visit col %d\n", tid, i);
  };

  auto block_visit = [&] DEVICE (int leader, glang::Slice S) {
    if (threadIdx.x == 0) {
      printf("Block[%d] visit Leader %d [%d:%d:%d]\n", blockIdx.x, leader, S.start(), S.end(), S.step());
    }
    glang::Parfor<int, glang::kBlockScope> parfor;
    parfor(S, visit);
  };

  glang::Slice RSlice(R.start(), R.end());
  auto Rc = R;

  auto f = [&] DEVICE (int i) {
    glang::Slice S(Rc[i], Rc[i+1]);
    if (S.size() <= 0) {
      //printf("[%d] idle.\n", tid);
    }
    enlist.until(S.size() > 0, S, block_visit);
  };

  glang::Expand<int, glang::kDeviceScope> expand;
  expand(RSlice, f);
}

template<int NUM_THREADS>
__global__ void pool_test() {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.x, j, t;
  typedef glang::BlockPool<glang::kWarpScope, NUM_THREADS> BlockPool;
  __shared__ typename BlockPool::TempStorage temp_storage;
  BlockPool(temp_storage).claim(i, j, t);
  printf("[%d] alloc %d at %d\n", tid, i, j);
}

template<int NUM_THREADS>
__global__ void gather_test(glang::DynArray<int> R, glang::DynArray<float> M) {
  typedef Gather<int, NUM_THREADS> Gather;
  __shared__ typename Gather::TempStorage temp_storage;
  Gather gather(temp_storage);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  auto visit = [&] DEVICE (int i) {
    printf("[%d] visit col %d\n", tid, i);
  };

  glang::Slice RSlice(R.start(), R.end());
  auto Rc = R;

  auto f = [&] DEVICE (int i) {
    glang::Slice S(Rc[i], Rc[i+1]);
    gather(S, visit);
  };

  glang::Expand<int, glang::kDeviceScope> expand;
  expand(RSlice, f);
}

template<class T>
glang::DynArray<T> makeDA(int len, T *hData, T default_v) {
  T *dData;
  CUDA_CHECK(cudaMalloc(&dData, len * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(dData, hData, len * sizeof(T), cudaMemcpyHostToDevice));
  return glang::DynArray<T>(len, dData, default_v);
}

int main(void) {
  const int kDataLen = 64;
  glang::Slice S0(0, kDataLen);
  float* device_out;
  cudaMalloc(&device_out, kDataLen * sizeof(float));
  glang::DynArray<float> Out(kDataLen, device_out);

  type_test<<<1, kDataLen + 1>>>(Out, S0);
  cudaDeviceSynchronize();

  pool_test<128> <<<1, 128>>>();
  cudaDeviceSynchronize();

  int hR[] = {0, 3, 4, 8, 10, 20, 26, 32, 32, 67, 78, 88, 91, 93, 95, 97, 111, 124, 150, 180, 190, 195, 196, 197, 199, 201, 222, 252, 299, 310, 356, 365, 388, 446, 450};
  float hM[256];
  auto dR = makeDA<int>(sizeof(hR)/sizeof(int), hR, 0);
  auto dM = makeDA<float>(256, hM, 0);
  spmv_test<6> <<<1, 6>>>(dR, dM);
  cudaDeviceSynchronize();
  printf("===Test WarpEnlist===\n");
  warp_enlist_test<64> <<<1, 64>>>(dR, dM);
  cudaDeviceSynchronize();
  printf("===Test BlockEnlist===\n");
  block_enlist_test<8> <<<8, 8>>>(dR, dM);
  cudaDeviceSynchronize();
  printf("===Test Gather===\n");
  gather_test<8> <<<8, 8>>>(dR, dM);
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
