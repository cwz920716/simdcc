#ifndef _COMPILER_MEMORY_ANALYSIS_PASS_H
#define _COMPILER_MEMORY_ANALYSIS_PASS_H

#include <llvm/Pass.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/InstVisitor.h>
#include <llvm/IR/IRBuilder.h>

#include "base.h"
#include "kernel_info_pass.h"

#define MEM_PARAMS_TYPENAME "MemParams"
#define BEFORE_MEM_HANDLER_FUNCNAME "before_mem_handler"

namespace gpuvm {

class MemoryAnalysisPass: public llvm::ModulePass {
 public:
  MemoryAnalysisPass() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module&);
  // We don't modify the program, so we preserve all analyses
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const;

  static char ID;

 private:
  llvm::StructType *DefineHandlerParamsType(llvm::Module& module);
};

}  // namespace gpuvm

#endif  // _COMPILER_MEMORY_ANALYSIS_PASS_H
