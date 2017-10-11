#ifndef __GPUVM_RT_H
#define __GPUVM_RT_H

#define DO_NOTHING(name, ty) \
    void name ## _handler(ty) {}

#include "types.inc"

extern "C" {

__device__ void before_branch_handler(struct CondBranchParams *ptr);
__device__ void before_mem_handler(struct MemParams *ptr);

void before_main_handler(void);
void after_main_handler(void);
void before_kernel_handler(void);
void after_kernel_handler(void);
void before_reset_handler(void);

}

#endif  // __GPUVM_RT_H
