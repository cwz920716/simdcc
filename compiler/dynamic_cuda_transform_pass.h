#ifndef _COMPILER_DYNAMIC_CUDA_TRANSFORM_PASS_H
#define _COMPILER_DYNAMIC_CUDA_TRANSFORM_PASS_H

#include <llvm/Pass.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/InstVisitor.h>
#include <llvm/IR/IRBuilder.h>

#include <map>
#include <vector>

#define PREPARE_DYN_CUDA_FUNCNAME "PrepareDynamicCuda"
#define DESTROY_DYN_CUDA_FUNCNAME "DestroyDynamicCuda"
#define DYN_CUDA_CONFIG_CALL_FUNCNAME "dynamicCudaConfigureCall"
#define DYN_CUDA_SETUP_ARGS_FUNCNAME "dynamicCudaSetupArgument"
#define DYN_CUDA_LAUNCH_FUNCNAME "dynamicCudaLaunchByName"
#define DYN_CUDA_REGISTER_FUNC_FUNCNAME "dynamicCudaRegisterFunction"

namespace gpuvm {

class DynamicCudaTransformPass : public llvm::ModulePass {
 public:
  static char ID;
  DynamicCudaTransformPass() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module&);
  // We don't modify the program, so we preserve all analyses
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const;

 private:
  bool IsCudaLaunch(const llvm::Instruction *inst);
  bool IsCudaDeviceReset(const llvm::Instruction *inst);
};

}  // namespave gpuvm

#endif // _COMPILER_DYNAMIC_CUDA_TRANSFORM_PASS_H
