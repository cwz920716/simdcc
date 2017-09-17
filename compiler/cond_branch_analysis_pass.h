#ifndef _CUDA_COND_BRANCH_ANALYSIS_Pass_H
#define _CUDA_COND_BRANCH_ANALYSIS_Pass_H

#include <llvm/Pass.h>
#include <llvm/IR/Instructions.h>

#include "base.h"
#include "kernel_info_pass.h"

namespace gpuvm {

class CondBranchAnalysisPass: public llvm::ModulePass {
 public:
  static char ID;
  CondBranchAnalysisPass() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module&);
  // We don't modify the program, so we preserve all analyses
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const;
};

}  // namespace gpuvm

#endif  // #ifndef _CUDA_COND_BRANCH_ANALYSIS_Pass_H
