#ifndef _KERNEL_INFO_PASS_H
#define _KERNEL_INFO_PASS_H

#include <llvm/Pass.h>
#include <llvm/IR/Instructions.h>

#define NVPTX_NVIDIA_CUDA "nvptx-nvidia-cuda"
#define NVPTX64_NVIDIA_CUDA "nvptx64-nvidia-cuda"

#define NVPTX_DATA_LAYOUT "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
#define NVPTX64_DATA_LAYOUT "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

#define NVVM_ANNOTATIONS "nvvm.annotations"

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

  bool runOnModule(llvm::Module&);
  // We don't modify the program, so we preserve all analyses
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const;
};

}  // namespace gpuvm

#endif
