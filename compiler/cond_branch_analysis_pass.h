#ifndef _CUDA_COND_BRANCH_ANALYSIS_Pass_H
#define _CUDA_COND_BRANCH_ANALYSIS_Pass_H

#include <llvm/Pass.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/InstVisitor.h>
#include <llvm/IR/IRBuilder.h>

#include "base.h"
#include "kernel_info_pass.h"

#define COND_BRANCH_PARAMS_TYPENAME "CondBranchParams"
#define BEFORE_BRANCH_HANDLER_FUNCNAME "before_branch_handler"

namespace gpuvm {

class CondBranchAnalysisPass: public llvm::ModulePass {
 public:
  CondBranchAnalysisPass() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module&);
  // We don't modify the program, so we preserve all analyses
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const;

  static char ID;

 private:
  llvm::StructType *DefineHandlerParamsType(llvm::Module& module);

  InstStatistics branch_stat_;
};

}  // namespace gpuvm

#endif  // #ifndef _CUDA_COND_BRANCH_ANALYSIS_Pass_H
