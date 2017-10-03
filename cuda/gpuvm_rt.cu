#include <cstdio>

// #include <sm_20_intrinsics.h>

#include "cuda_util.h"
#include "gpuvm_rt.h"
#include "sassi_runtime/sassi_dictionary.hpp"

struct BranchCounter {
  uint64_t id;
  int32_t branchType;                    // The branch type.
  unsigned long long totalBranches;
  unsigned long long takenThreads;
  unsigned long long takenNotThreads;
  unsigned long long divergentBranches;   // Not all branches go the same way.
  unsigned long long activeThreads;       // Number of active threads.
};                                        

__device__ sassi::dictionary<uint64_t, BranchCounter> *sassiStats;

__device__ void before_branch_handler(struct CondBranchParams *ptr) {
  if (ptr == NULL) {
    return;
  }

  printf("Thread (%d, %d) at branch %d, %s, %s, %p\n", blockIdx.x, threadIdx.x,
              ptr->id, ptr->taken ? "taken" : "not-taken",
              ptr->is_conditional ? "br" : "jmp", sassiStats);
  // XXX[95] = 0;
  return;
}

__device__ void before_mem_handler(struct MemParams *ptr) {
  if (ptr == NULL) {
    return;
  }

  void *Addr = (void *) ptr->address;

  printf("Thread (%d, %d) is %s %ld bits from %p (AP: %d)\n",
              blockIdx.x, threadIdx.x, ptr->write ? "storing" : "loading",
              ptr->size_in_bits, Addr, __isGlobal(Addr) ? 1 : ptr->addr_space);
  return;
}

void before_main_handler(void) {
  auto sassiStats_h = new sassi::dictionary<uint64_t, BranchCounter>();
  CUDA_CHECK(cudaMemcpyToSymbol(sassiStats, &sassiStats_h, sizeof(sassiStats_h)));
  printf("XXX H: %p\n", sassiStats_h);
}

DO_NOTHING(after_main, void)
DO_NOTHING(before_kernel, void)
DO_NOTHING(after_kernel, void)
