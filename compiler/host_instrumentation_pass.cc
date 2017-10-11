#include <glog/logging.h>

#include "host_instrumentation_pass.h"
#include "llvm_utils.h"

namespace gpuvm {

bool HostInstrumentationPass::IsCudaLaunch(const llvm::Instruction *inst) {
  if (inst == nullptr) {
    return false;
  }

  LLVM_STRING(cudaLaunch);
  if (auto call = llvm::dyn_cast<llvm::CallInst>(inst)) {
    if (auto callee = call->getCalledFunction()) {
      if (callee->getName() == cudaLaunch) {
        return true;
      }
    }
  }

  return false;
}

bool HostInstrumentationPass::IsCudaDeviceReset(const llvm::Instruction *inst) {
  if (inst == nullptr) {
    return false;
  }

  LLVM_STRING(cudaDeviceReset);
  if (auto call = llvm::dyn_cast<llvm::CallInst>(inst)) {
    if (auto callee = call->getCalledFunction()) {
      if (callee->getName() == cudaDeviceReset) {
        return true;
      }
    }
  }

  return false;
}

bool HostInstrumentationPass::runOnModule(llvm::Module& module) {
  llvm::ArrayRef<llvm::Type *> void_arg_types;
  llvm::LLVMContext &ctx = module.getContext();
  auto type_void = llvm::Type::getVoidTy(ctx);
  auto type_main_handler =
      llvm::FunctionType::get(type_void, void_arg_types, false);
  llvm::StringRef before_main_handler_func_name(BEFORE_MAIN_HANDLER_FUNCNAME);
  auto before_main_handler =
      module.getOrInsertFunction(before_main_handler_func_name,
                                 type_main_handler);
  llvm::StringRef after_main_handler_func_name(AFTER_MAIN_HANDLER_FUNCNAME);
  auto after_main_handler =
      module.getOrInsertFunction(after_main_handler_func_name,
                                 type_main_handler);

  LLVM_STRING(main);
  llvm::ArrayRef<llvm::Value *> void_arg;

  for (auto &func : module) {
    // instrument at main 
    if (func.getName() == main) {
      llvm::BasicBlock &entry = func.getEntryBlock();
      llvm::IRBuilder<> builder(entry.getFirstNonPHI());
      builder.CreateCall(before_main_handler, void_arg);

      for (auto &bb : func) {
        auto terminator = bb.getTerminator();
        if (llvm::ReturnInst *ret =
                llvm::dyn_cast<llvm::ReturnInst>(terminator)) {
          builder.SetInsertPoint(ret);
          builder.CreateCall(after_main_handler, void_arg);
        }

        if (llvm::UnreachableInst *unreachable =
                llvm::dyn_cast<llvm::UnreachableInst>(terminator)) {
          llvm::Instruction *pred = nullptr;
          for (auto it = bb.begin(); it != bb.end(); ++it) {
            // do not insert call back if the bb only has
            // phi nodes and unreachables
            llvm::Instruction *inst = &(*it);
            if (llvm::PHINode *phi = llvm::dyn_cast<llvm::PHINode>(inst)) {
              continue;
            }

            if (inst == unreachable && pred != nullptr) {
              builder.SetInsertPoint(pred);
              builder.CreateCall(after_main_handler, void_arg);
            }
            pred = inst;
          }
        }

        if (llvm::ResumeInst  *resume =
                llvm::dyn_cast<llvm::ResumeInst >(terminator)) {
          LOG(WARNING)
              << "after_main_handler() for resume not implemented yet";
        }

        if (llvm::CleanupReturnInst  *cleanupret =
                llvm::dyn_cast<llvm::CleanupReturnInst >(terminator)) {
          LOG(WARNING)
              << "after_main_handler() for cleanupret not implemented yet";
        }
      }
    }
  }

  // instrument cuda launch
  llvm::StringRef
      before_kernel_handler_func_name(BEFORE_KERNEL_HANDLER_FUNCNAME);
  auto before_kernel_handler =
      module.getOrInsertFunction(before_kernel_handler_func_name,
                                 type_main_handler);
  llvm::StringRef
      after_kernel_handler_func_name(AFTER_KERNEL_HANDLER_FUNCNAME);
  auto after_kernel_handler =
      module.getOrInsertFunction(after_kernel_handler_func_name,
                                 type_main_handler);
  // instrument device reset
  llvm::StringRef
      before_reset_handler_func_name(BEFORE_DEVICE_RESET_HANDLER_FUNCNAME);
  auto before_reset_handler =
      module.getOrInsertFunction(before_reset_handler_func_name,
                                 type_main_handler);

  for (auto &func : module) {
    for (auto &bb : func) {
      llvm::Instruction *pred = nullptr;
      llvm::IRBuilder<> builder(bb.getFirstNonPHI());
      for (auto it = bb.begin(); it != bb.end(); ++it) {
        llvm::Instruction *inst = &(*it);
        if (IsCudaLaunch(inst)) {
          builder.SetInsertPoint(inst);
          builder.CreateCall(before_kernel_handler, void_arg);
        }

        if (IsCudaLaunch(pred)) {
          builder.SetInsertPoint(inst);
          builder.CreateCall(after_kernel_handler, void_arg);
        }

        if (IsCudaDeviceReset(inst)) {
          builder.SetInsertPoint(inst);
          builder.CreateCall(before_reset_handler, void_arg);
        }

        pred = inst;
      }
    } 
  }
}

void HostInstrumentationPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const
{
	AU.setPreservesCFG();
}

// LLVM uses the address of this static member to identify the pass, so the
// initialization value is unimportant.
char HostInstrumentationPass::ID = 1;

static llvm::RegisterPass<HostInstrumentationPass> X("HostInstrumentationPass",
    "Instrument cuda host module.");

}  // namespace gpuvm
