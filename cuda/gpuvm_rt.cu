#include "gpuvm_rt.h"

#include "sassi_runtime/sassi_dictionary.hpp"

#include <cstdio>

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

static __managed__ sassi::dictionary<uint64_t, BranchCounter> *sassiStats;

__device__ void before_branch_handler(struct CondBranchParams *ptr) {
  if (ptr == NULL) {
    return;
  }

  printf("Thread %d at branch %d, X=%p\n", threadIdx.x, ptr->id, sassiStats);
  return;
}
