#include <glog/logging.h>

#include "cond_branch_analysis_pass.h"

namespace gpuvm {

class BranchInstVisitor: public llvm::InstVisitor<BranchInstVisitor> {
 public:
  BranchInstVisitor(llvm::Function *func, InstStatistics *br_stat) :
      parent(func), stat(br_stat), cond_branch_params(nullptr) {}

  llvm::AllocaInst *GetCondBranchParams() {
    if (cond_branch_params != nullptr) {
      return cond_branch_params;
    }

    llvm::Module *module = parent->getParent();
    llvm::StructType *type = module->getTypeByName(COND_BRANCH_PARAMS_TYPENAME);
    llvm::BasicBlock &entry = parent->getEntryBlock();
    llvm::IRBuilder<> builder(entry.getFirstNonPHI());
    cond_branch_params = builder.CreateAlloca(type);
    return cond_branch_params;
  }

  void visitBranchInst(llvm::BranchInst &br) {
    // llvm::errs() << "Visit " << br << "\n";

    llvm::Module *module = parent->getParent();
    llvm::LLVMContext &ctx = module->getContext();
    llvm::StructType *type_cond_branch_params =
        module->getTypeByName(COND_BRANCH_PARAMS_TYPENAME);
    llvm::Type *type_int32 = llvm::IntegerType::get(ctx, 32);
    llvm::Type *type_int8 = llvm::IntegerType::get(ctx, 8);

    // get branch id.
    int br_id = stat->Record(&br);
    // allocate argument on stack.
    auto argument = GetCondBranchParams();
    CHECK(argument != nullptr);
    // get callback function.
    llvm::Function *br_handler =
        module->getFunction(BEFORE_BRANCH_HANDLER_FUNCNAME);
    CHECK(br_handler != nullptr);

    // set up callback arguments.
    llvm::IRBuilder<> builder(&br);
    auto ptr_bid = builder.CreateConstGEP2_32(type_cond_branch_params,
                                              argument, 0, 0);
    auto value_bid = llvm::ConstantInt::getSigned(type_int32, br_id);
    builder.CreateStore(value_bid, ptr_bid);

    auto ptr_taken = builder.CreateConstGEP2_32(type_cond_branch_params,
                                                argument, 0, 1);
    llvm::Value *value_taken = llvm::ConstantInt::get(type_int8, 1);
    if (br.isConditional()) {
      value_taken = builder.CreateZExt(br.getCondition(), type_int8);
    }
    builder.CreateStore(value_taken, ptr_taken);

    auto ptr_is_conditional =
        builder.CreateConstGEP2_32(type_cond_branch_params, argument, 0, 2);
    auto value_is_conditional =
        llvm::ConstantInt::get(type_int8, br.isConditional());
    builder.CreateStore(value_is_conditional, ptr_is_conditional);

    // invoke callback.
    llvm::SmallVector<llvm::Value *, 1> args(1, argument);
    builder.CreateCall(br_handler, args);
  }

  // visit indirect branch inst

  llvm::Function *parent;
  InstStatistics *stat;
  llvm::AllocaInst *cond_branch_params;
};

llvm::StructType *
CondBranchAnalysisPass::DefineHandlerParamsType(llvm::Module& module) {
  auto type_cond_brach_params = module.getTypeByName(COND_BRANCH_PARAMS_TYPENAME);
  if (type_cond_brach_params) {
    return type_cond_brach_params;
  }

  llvm::LLVMContext &ctx = module.getContext();

  llvm::Type *type_int32 = llvm::IntegerType::get(ctx, 32);
  llvm::Type *type_int8 = llvm::IntegerType::get(ctx, 8);

  llvm::StringRef type_name(COND_BRANCH_PARAMS_TYPENAME);
  llvm::Type *type_array[] = {type_int32, type_int8, type_int8};
  llvm::ArrayRef<llvm::Type *> struct_types(type_array, 3);
  return llvm::StructType::create(struct_types, type_name);
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
  llvm::LLVMContext &ctx = module.getContext();
  llvm::SmallVector<llvm::Type *, 1> param_types(1,
      llvm::PointerType::get(type_cond_brach_params, GENERIC_ADDR_SPACE));
  llvm::StringRef func_name(BEFORE_BRANCH_HANDLER_FUNCNAME);
  auto type_void = llvm::Type::getVoidTy(ctx);
  auto type_func = llvm::FunctionType::get(type_void, param_types, false);
  auto before_branch_handler =
      module.getOrInsertFunction(func_name, type_func);

  /*
   * For (each function in module):
   *   Allocate a new instance of CondBranchParams on stack
   *   For (each instruction in module):
   *     If (not branch):
   *       continue
   *     Set up CondBranchParams
   *     Call before_branch_handler() before branch
   */
  for (llvm::Function &func : module) {
    if (func.isDeclaration()) {
      continue;
    }

    BranchInstVisitor br_visitor(&func, &branch_stat_);
    LOG(INFO) << "Run BranchInstVisitor On " << func.getName().str();
    br_visitor.visit(func);
  }

  LOG(INFO) << "Branch Description:\n" << branch_stat_.DebugStr("branch");

  return true;
}

void CondBranchAnalysisPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const
{
  AU.addRequired<KernelInfoPass>();
	AU.setPreservesCFG();
}

// LLVM uses the address of this static member to identify the pass, so the
// initialization value is unimportant.
char CondBranchAnalysisPass::ID = 1;

static llvm::RegisterPass<CondBranchAnalysisPass> X("CondBranchAnalysisPass",
    "Dynamic analysis of conditional branches in cuda module.");

}  // namespace gpuvm
