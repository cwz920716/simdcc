#ifndef _COMPILER_HOST_INSTRUMENTATION_PASS_H
#define _COMPILER_HOST_INSTRUMENTATION_PASS_H

#include <llvm/Pass.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/InstVisitor.h>
#include <llvm/IR/IRBuilder.h>

#define BEFORE_MAIN_HANDLER_FUNCNAME "before_main_handler"
#define AFTER_MAIN_HANDLER_FUNCNAME "after_main_handler"
#define BEFORE_DEVICE_RESET_HANDLER_FUNCNAME "before_reset_handler"
#define BEFORE_KERNEL_HANDLER_FUNCNAME "before_kernel_handler"
#define AFTER_KERNEL_HANDLER_FUNCNAME "after_kernel_handler"

namespace gpuvm {

class HostInstrumentationPass : public llvm::ModulePass {
 public:
  static char ID;
  HostInstrumentationPass() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module&);
  // We don't modify the program, so we preserve all analyses
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const;

 private:
  bool IsCudaLaunch(const llvm::Instruction *inst);
  bool IsCudaDeviceReset(const llvm::Instruction *inst);
};

}  // namespave gpuvm

#endif // _COMPILER_HOST_INSTRUMENTATION_PASS_H
