#include <cstdio>

#include "cuda_util.h"
#include "gpuvm_rt.h"
#include "sassi_runtime/sassi_dictionary.hpp"

struct BranchCounter {
  uint64_t address;
  int32_t branchType;                    // The branch type.
  int32_t taggedUnanimous;               // Branch had .U modifier, so compiler knows...
  unsigned long long totalBranches;
  unsigned long long takenThreads;
  unsigned long long takenNotThreads;
  unsigned long long divergentBranches;   // Not all branches go the same way.
  unsigned long long activeThreads;       // Number of active threads.
};                                        

__device__ sassi::dictionary<uint64_t, BranchCounter> *sassiStats;
__device__ int *XXX;

__device__ void before_branch_handler(struct CondBranchParams *ptr) {
  if (ptr == NULL) {
    return;
  }

  printf("Thread (%d, %d) at branch %d, %s, %s, %p\n", blockIdx.x, threadIdx.x,
              ptr->id, ptr->taken ? "taken" : "not-taken",
              ptr->is_conditional ? "br" : "jmp", XXX);
  XXX[95] = 0;
  return;
}

void before_main_handler(void) {
  int *XXX_h;
  CUDA_CHECK(cudaMalloc(&XXX_h, 100*sizeof(int)));
  CUDA_CHECK(cudaMemcpyToSymbol(XXX, &XXX_h, sizeof(XXX_h)));
  printf("XXX H: %p\n", XXX_h);
}
