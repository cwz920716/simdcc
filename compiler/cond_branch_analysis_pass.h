#ifndef _CUDA_COND_BRANCH_ANALYSIS_Pass_H
#define _CUDA_COND_BRANCH_ANALYSIS_Pass_H

#include <llvm/Pass.h>
#include <llvm/IR/Instructions.h>

#include "base.h"
#include "kernel_info_pass.h"

#define COND_BRANCH_PARAMS_TYPENAME "CondBranchParams"

namespace gpuvm {

class CondBranchAnalysisPass: public llvm::ModulePass {
 public:
  static char ID;
  CondBranchAnalysisPass() : llvm::ModulePass(ID), num_branches_(0) {}

  bool runOnModule(llvm::Module&);
  // We don't modify the program, so we preserve all analyses
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const;

 private:
  int64_t num_branches_;

  llvm::StructType *
  CondBranchAnalysisPass::DefineHandlerParamsType(llvm::Module& module)
};

}  // namespace gpuvm

#endif  // #ifndef _CUDA_COND_BRANCH_ANALYSIS_Pass_H
