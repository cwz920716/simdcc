#ifndef __COMPILER_CUDA_UTILS_H
#define __COMPILER_CUDA_UTILS_H

#include "llvm_utils.h"

namespace gpuvm {

inline bool IsCudaLaunch(const llvm::Instruction *inst) {
  if (inst == nullptr) {
    return false;
  }

  LLVM_STRING(cudaLaunch);
  LLVM_STRING(dynamicCudaLaunch);
  if (auto call = llvm::dyn_cast<llvm::CallInst>(inst)) {
    if (auto callee = call->getCalledFunction()) {
      if (callee->getName() == cudaLaunch ||
          callee->getName() == dynamicCudaLaunch) {
        return true;
      }
    }
  }

  return false;
}

}  // namespace gpuvm

#endif  // __COMPILER_CUDA_UTILS_H
