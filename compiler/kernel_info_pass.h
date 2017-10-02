#ifndef _COMPILER_KERNEL_INFO_PASS_H
#define _COMPILER_KERNEL_INFO_PASS_H

#include <llvm/Pass.h>
#include <llvm/IR/Instructions.h>

#include "base.h"

namespace gpuvm {

enum ModuleType {
  kNvidiaPtx64 = 0,
  kNvidiaPtx32,
  kNonPtx
};

class KernelInfoPass: public llvm::ModulePass {
 public:
  static char ID;
  KernelInfoPass() : llvm::ModulePass(ID) {}

  bool IsCudaModule(llvm::Module&) const;
  bool runOnModule(llvm::Module&);
  // We don't modify the program, so we preserve all analyses
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const;
};

}  // namespace gpuvm

#endif  // #ifndef _COMPILER_KERNEL_INFO_PASS_H
