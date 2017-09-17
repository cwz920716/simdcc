#include <glog/logging.h>

#include "cond_branch_analysis_pass.h"

namespace gpuvm {

bool CondBranchAnalysisPass::runOnModule(llvm::Module& module) {
  const KernelInfoPass &kernel_info = getAnalysis<KernelInfoPass>();
  if (!kernel_info.IsCudaModule(module)) {
    LOG(FATAL) << "CondBranchAnalysisPass Cannot run on non-cuda module!";
    return false;
  }

  
  return true;
}

void CondBranchAnalysisPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const
{
  AU.addRequired<KernelInfoPass>();
	AU.setPreservesCFG();
}

// LLVM uses the address of this static member to identify the pass, so the
// initialization value is unimportant.
char CondBranchAnalysisPass::ID = 0;

static llvm::RegisterPass<CondBranchAnalysisPass> X("CondBranchAnalysisPass",
    "Dynamic analysis of conditional branches in cuda module.");

}  // namespace gpuvm
