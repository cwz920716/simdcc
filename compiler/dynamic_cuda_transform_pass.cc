#include <glog/logging.h>

#include "dynamic_cuda_transform_pass.h"
#include "cuda_utils.h"

namespace gpuvm {

class CudaLaunchVisitor: public llvm::InstVisitor<CudaLaunchVisitor> {
 public:
  CudaLaunchVisitor(llvm::Function *func) :
      parent_(func) {}

  void visitCallInst(llvm::CallInst &call) {
    if (IsCudaLaunch(&call)) {
      // do something...
      CHECK(call.getNumArgOperands() > 0);
    }
  }

 private:
  llvm::Function *parent_;
};

class FunctionRenamer {
 public:
  FunctionRenamer() {}

  FunctionRenamer &map(const std::string &old_name, const std::string &new_name) {
    names_map_[old_name] = new_name;
    return *this;
  }

  void runOnModule(llvm::Module& module) {
    for (auto &func : module) {
      llvm::StringRef fname = func.getName();
      auto fname_str = fname.str();
      if (names_map_.find(fname_str) != names_map_.end()) {
        func.setName(names_map_[fname_str]);
      }
    }
  }

 private:
  std::map<std::string, std::string> names_map_;
};

bool DynamicCudaTransformPass::runOnModule(llvm::Module& module) {
  llvm::ArrayRef<llvm::Type *> void_arg_types;
  llvm::LLVMContext &ctx = module.getContext();
  auto type_void = llvm::Type::getVoidTy(ctx);
  auto type_void_handler =
      llvm::FunctionType::get(type_void, void_arg_types, false);
  llvm::StringRef prepare_dyn_cuda_func_name(PREPARE_DYN_CUDA_FUNCNAME);
  auto prepare_dyn_cuda =
      module.getOrInsertFunction(prepare_dyn_cuda_func_name,
                                 type_void_handler);
  llvm::StringRef destroy_dyn_cuda_func_name(DESTROY_DYN_CUDA_FUNCNAME);
  auto destroy_dyn_cuda =
      module.getOrInsertFunction(destroy_dyn_cuda_func_name,
                                 type_void_handler);

  FunctionRenamer func_renamer;
  func_renamer.map("cudaConfigureCall", "dynamicCudaConfigureCall")
              .map("cudaSetupArgument", "dynamicCudaSetupArgument")
              .map("cudaLaunch", "dynamicCudaLaunch")
              .runOnModule(module);

  LLVM_STRING(main);
  llvm::ArrayRef<llvm::Value *> void_arg;

  for (auto &func : module) {
    // instrument at main 
    if (func.getName() == main) {
      llvm::BasicBlock &entry = func.getEntryBlock();
      llvm::IRBuilder<> builder(entry.getFirstNonPHI());
      builder.CreateCall(prepare_dyn_cuda, void_arg);

      for (auto &bb : func) {
        auto terminator = bb.getTerminator();
        if (llvm::ReturnInst *ret =
                llvm::dyn_cast<llvm::ReturnInst>(terminator)) {
          builder.SetInsertPoint(ret);
          builder.CreateCall(destroy_dyn_cuda, void_arg);
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
              builder.CreateCall(destroy_dyn_cuda, void_arg);
            }
            pred = inst;
          }
        }

        if (llvm::ResumeInst  *resume =
                llvm::dyn_cast<llvm::ResumeInst >(terminator)) {
          LOG(WARNING)
              << "destroy_dyn_cuda() for resume not implemented yet";
        }

        if (llvm::CleanupReturnInst  *cleanupret =
                llvm::dyn_cast<llvm::CleanupReturnInst >(terminator)) {
          LOG(WARNING)
              << "destroy_dyn_cuda() for cleanupret not implemented yet";
        }
        // End-Of-Instrumentation
      }
    }

    CudaLaunchVisitor cuda_launch_visitor(&func);
    cuda_launch_visitor.visit(func);
  }
}

void DynamicCudaTransformPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const
{
	AU.setPreservesCFG();
}

// LLVM uses the address of this static member to identify the pass, so the
// initialization value is unimportant.
char DynamicCudaTransformPass::ID = 1;

static llvm::RegisterPass<DynamicCudaTransformPass> X("DynamicCudaTransformPass",
    "Tranform cuda host module to use dynamic cuda.");

}  // namespace gpuvm
