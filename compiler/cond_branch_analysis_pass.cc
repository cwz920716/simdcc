#include <glog/logging.h>

#include "cond_branch_analysis_pass.h"

namespace gpuvm {

llvm::StructType *
CondBranchAnalysisPass::DefineHandlerParamsType(llvm::Module& module) {
  auto type_cond_brach_params = module.getTypeByName(COND_BRANCH_PARAMS_TYPENAME);
  if (type_cond_brach_params) {
    return type_cond_brach_params;
  }

  llvm::LLVMContext &ctx = module.getContext();

  auto type_int32 = llvm::IntegerType::get(TheContext, 32);
  auto type_int8 = llvm::IntegerType::get(TheContext, 8);

  SmallVector<Type *, 3> type_fields[] = {type_int32, type_int8, type_int8};
  return llvm::StructType::create(ctx, type_fields,
                                  COND_BRANCH_PARAMS_TYPENAME);
}

bool CondBranchAnalysisPass::runOnModule(llvm::Module& module) {
  const KernelInfoPass &kernel_info = getAnalysis<KernelInfoPass>();
  if (!kernel_info.IsCudaModule(module)) {
    LOG(FATAL) << "CondBranchAnalysisPass Cannot run on non-cuda module!";
    return false;
  }

  // Add a type for CondBranchParams.
  auto type_cond_brach_params = DefineHandlerParamsType(module);

  // Declare a device function prototype:
  // __device__ void before_branch_handler(CondBranchParams *);

  /*
   * For (each function in module):
   *   Allocate a new instance of CondBranchParams on stack
   *   For (each instruction in module):
   *     If (not branch):
   *       continue
   *     Set up CondBranchParams
   *     Call before_branch_handler() before branch
   */

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
