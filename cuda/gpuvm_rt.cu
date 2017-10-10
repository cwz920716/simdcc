#include <assert.h>
#include <inttypes.h>
#include <map>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <unistd.h>

// #include <sm_20_intrinsics.h>

#include "cuda_util.h"
#include "gpuvm_rt.h"
#include "sassi_runtime/sassi_dictionary.hpp"
#include "sassi_runtime/sassi_intrinsics.h"

struct BranchCounter {
  uint64_t id;
  bool uniform;                    // The branch type.
  unsigned long long totalBranches;
  unsigned long long takenThreads;
  unsigned long long takenNotThreads;
  unsigned long long divergentBranches;   // Not all branches go the same way.
  unsigned long long activeThreads;       // Number of active threads.
};                                        

__device__ sassi::dictionary<uint64_t, BranchCounter> *sassiStats;
sassi::dictionary<uint64_t, BranchCounter> *sassiStats_h;

__device__ void before_branch_handler(struct CondBranchParams *brp) {
  if (brp == NULL) {
    return;
  }

  printf("Thread (%d, %d) at branch %d, %s, %s, %p\n", blockIdx.x, threadIdx.x,
              brp->id, brp->taken ? "taken" : "not-taken",
              brp->is_conditional ? "br" : "jmp", sassiStats);

  // Find out thread index within the warp.
  int threadIdxInWarp = get_laneid();

  // Get masks and counts of 1) active threads in this warp,
  // 2) threads that take the branch, and
  // 3) threads that do not take the branch.
  int active = __ballot(1);
  bool dir = brp->taken;
  int taken = __ballot(dir == true);
  int ntaken = __ballot(dir == false);

  int numActive = __popc(active);
  int numTaken = __popc(taken);
  int numNotTaken = __popc(ntaken);

  bool divergent = (numTaken != numActive && numNotTaken != numActive);

  // The first active thread in each warp gets to write results.
  if ((__ffs(active)-1) == threadIdxInWarp) {
    // Get the address, we'll use it for hashing.
    uint64_t instId = brp->id;
    
    // Looks up the counters associated with 'instAddr', but if no such entry
    // exits, initialize the counters in the lambda.
    BranchCounter *stats = (*sassiStats).getOrInit(instId,
        [instId,brp](BranchCounter* v) {
          v->id = instId;
          v->uniform = !brp->is_conditional;
        });

    // Why not sanity check the hash map?
    assert(stats->id == instId);
    assert(numTaken + numNotTaken == numActive);

    // Increment the various counters that are associated
    // with this instruction appropriately.
    atomicAdd(&(stats->totalBranches), 1ULL);
    atomicAdd(&(stats->activeThreads), numActive);
    atomicAdd(&(stats->takenThreads), numTaken);
    atomicAdd(&(stats->takenNotThreads), numNotTaken);
    atomicAdd(&(stats->divergentBranches), divergent);
  }

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
  sassiStats_h = new sassi::dictionary<uint64_t, BranchCounter>();
  CUDA_CHECK(cudaMemcpyToSymbol(sassiStats, &sassiStats_h, sizeof(sassiStats_h)));
  printf("sassiStats H: %p\n", sassiStats_h);

  
}

void after_main_handler(void) {
  FILE *fRes = fopen("sassi-branch.txt", "w");

  fprintf(fRes, "%-16.16s %-10.10s %-10.10s %-10.10s %-10.10s %-10.10s %-8.8s\n",
      "Id", "Total/32", "Dvrge/32", "Active", "Taken", "NTaken", 
	    "Uni");

  sassiStats_h->map([fRes](uint64_t& id, BranchCounter& val) {
	  assert(val.id == id);
	
	  fprintf(fRes,
		  "%-16lu %-10.llu %-10.llu %-10.llu %-10.llu %-10.llu %-8.d\n",
		  id,
		  val.totalBranches, 
		  val.divergentBranches,
		  val.activeThreads,
		  val.takenThreads,
		  val.takenNotThreads,
		  val.uniform
		);
  });
  
  fclose(fRes);
}

DO_NOTHING(before_kernel, void)
DO_NOTHING(after_kernel, void)
