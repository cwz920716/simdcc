#ifndef __GPUVM_RT_H
#define __GPUVM_RT_H

extern "C" {

struct CondBranchParams {
  int32_t id;
  bool taken;
  bool is_conditional;
};  // struct CondBranchParams

__device__ void before_branch_handler(struct CondBranchParams *ptr);
void before_main_handler(void);

}

#endif  // __GPUVM_RT_H
