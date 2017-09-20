#include "gpuvm_rt.h"
#include <cstdio>

__device__ void before_branch_handler(char *ptr) {
  if (ptr != NULL) {
    *ptr = 0;
  }

  // printf("XXX\n");
  return;
}
