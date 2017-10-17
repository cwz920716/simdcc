#include <assert.h>
#include <inttypes.h>
#include <map>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <unistd.h>

// #include <sm_20_intrinsics.h>

// #define DEBUG

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

/// The number of bits we need to shift off to get the cache line address.
#define LINE_BITS   5

// The width of a warp.
#define WARP_SIZE   32

/// The counters that we will use to record our statistics.
__device__ unsigned long long *memdiverge_counters;
unsigned long long *memdiverge_counters_h;
#define MEMDIVERGE_SIZE (WARP_SIZE + 1)

bool device_reset = false;

__device__ void before_branch_handler(struct CondBranchParams *brp) {
  if (brp == NULL) {
    return;
  }

#ifdef DEBUG
  printf("Thread (%d, %d) at branch %d, %s, %s, %p\n", blockIdx.x, threadIdx.x,
              brp->id, brp->taken ? "taken" : "not-taken",
              brp->is_conditional ? "br" : "jmp", sassiStats);
#endif  // DEBUG

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

#ifdef DEBUG
  void *Addr = (void *) ptr->address;

  printf("Thread (%d, %d) is %s %ld bits from %p (AP: %d)\n",
              blockIdx.x, threadIdx.x, ptr->write ? "storing" : "loading",
              ptr->size_in_bits, Addr, __isGlobal(Addr) ? 1 : ptr->addr_space);
#endif  // DEBUG

  intptr_t addrAsInt = ptr->address;
  // Don't look at shared or local memory.
  if (__isGlobal((void*)addrAsInt)) { 
    // The number of unique addresses across the warp 
    unsigned unique = 0;   // for the instrumented instruction.

    // Shift off the offset bits into the cache line.
    intptr_t lineAddr =  addrAsInt >> LINE_BITS;

    int workset = __ballot(1);
    int firstActive = __ffs(workset) - 1;
    int numActive = __popc(workset);
    while (workset) {
      // Elect a leader, get its line, see who all matches it.
      int leader = __ffs(workset) - 1;
      intptr_t leadersAddr = __broadcast<intptr_t>(lineAddr, leader);
      int notMatchesLeader = __ballot(leadersAddr != lineAddr);

      // We have accounted for all values that match the leader's.
      // Let's remove them all from the workset.
      workset = workset & notMatchesLeader;
      unique++;
      assert(unique <= 32);
    }

    assert(unique > 0 && unique <= 32);

    // Each thread independently computed 'numActive' and 'unique'.
    // Let's let the first active thread actually tally the result.
    int threadsLaneId = get_laneid();
    if (threadsLaneId == firstActive) {
      atomicAdd(&(memdiverge_counters[numActive* MEMDIVERGE_SIZE + unique]), 1LL);
    }
  }
  return;
}

void before_main_handler(void) {
  sassiStats_h = new sassi::dictionary<uint64_t, BranchCounter>();
  CUDA_CHECK(
      cudaMemcpyToSymbol(sassiStats, &sassiStats_h, sizeof(sassiStats_h)));

  printf("sassiStats H: %p\n", sassiStats_h);

  const int memdiverge_size =
      MEMDIVERGE_SIZE * MEMDIVERGE_SIZE * sizeof(unsigned long long);
  CUDA_CHECK(cudaMallocManaged(&memdiverge_counters_h, memdiverge_size););
  CUDA_CHECK(
      cudaMemcpyToSymbol(memdiverge_counters, &memdiverge_counters_h,
                         sizeof(memdiverge_counters_h)));

  printf("memdiverge_counters H: %p\n", memdiverge_counters_h);
}

void finalize(void) {
  FILE *fRes = fopen("sassi-branch.txt", "w");

  fprintf(fRes, "%-16.16s %-10.10s %-10.10s %-10.10s %-10.10s %-10.10s %-8.8s\n",
      "Id", "Total/32", "Dvrge/32", "Active", "Taken", "NTaken", 
	    "Uni");

  sassiStats_h->map([fRes](uint64_t& id, BranchCounter& val) {
	  CHECK(val.id == id);

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

  FILE *rf = fopen("sassi-memdiverge.txt", "w");
  fprintf(rf, "Active x Diverged:\n");
  for (unsigned m = 0; m <= WARP_SIZE; m++) {
    fprintf(rf, "%-2d> ", m);
    for (unsigned u = 0; u <= WARP_SIZE; u++) {
      fprintf(rf, "%-10llu ", memdiverge_counters_h[m * MEMDIVERGE_SIZE + u]);
    }
    fprintf(rf, "\n");
  }
  fprintf(rf, "\n");
  fclose(rf);
}

void after_main_handler(void) {
  if (device_reset) {
    return;
  }

  finalize();
}

void before_reset_handler(void) {
  if (device_reset) {
    return;
  }

  printf("Cuda device reset.\n");
  finalize();
  device_reset = true;
}

DO_NOTHING(before_kernel, void)
DO_NOTHING(after_kernel, void)
