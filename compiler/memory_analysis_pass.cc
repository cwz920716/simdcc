#include <glog/logging.h>

#include <cxxabi.h>
#include "memory_analysis_pass.h"
#include "llvm_utils.h"

namespace gpuvm {

class MemoryAccessVisitor: public llvm::InstVisitor<MemoryAccessVisitor> {
 public:
  MemoryAccessVisitor(llvm::Function *func) :
      parent(func), mem_params(nullptr) {}

  llvm::AllocaInst *GetMemParams() {
    if (mem_params != nullptr) {
      return mem_params;
    }

    llvm::Module *module = parent->getParent();
    llvm::StructType *type = module->getTypeByName(MEM_PARAMS_TYPENAME);
    llvm::BasicBlock &entry = parent->getEntryBlock();
    llvm::IRBuilder<> builder(entry.getFirstNonPHI());
    mem_params = builder.CreateAlloca(type);
    return mem_params;
  }

  void insertMemHandler(llvm::Instruction &inst, llvm::Value *pointer, 
                        bool is_write, bool is_atomic = false) {
    llvm::Module *module = parent->getParent();
    llvm::LLVMContext &ctx = module->getContext();
    const llvm::DataLayout &dl = module->getDataLayout();

    llvm::StructType *type_mem_params =
        module->getTypeByName(MEM_PARAMS_TYPENAME);
    llvm::Type *type_int64 = llvm::IntegerType::get(ctx, 64);
    llvm::Type *type_int32 = llvm::IntegerType::get(ctx, 32);
    llvm::Type *type_int8 = llvm::IntegerType::get(ctx, 8);

    // allocate argument on stack.
    auto argument = GetMemParams();
    CHECK(argument != nullptr);
    // get callback function.
    llvm::Function *mem_handler =
        module->getFunction(BEFORE_MEM_HANDLER_FUNCNAME);
    CHECK(mem_handler != nullptr);

    // set up callback arguments.
    llvm::IRBuilder<> builder(&inst);
    llvm::Type *type = pointer->getType();
    CHECK(type->isPointerTy());
    auto type_ptr = llvm::dyn_cast<llvm::PointerType>(type);
    auto type_access = type_ptr->getPointerElementType();

    auto ptr_address = builder.CreateConstGEP2_32(type_mem_params,
                                                  argument, 0, 0);
    auto value_address =
        builder.CreatePtrToInt(pointer, type_int64);
    builder.CreateStore(value_address, ptr_address);

    auto ptr_size = builder.CreateConstGEP2_32(type_mem_params,
                                               argument, 0, 1);
    llvm::Value *value_size =
        llvm::ConstantInt::get(type_int64, dl.getTypeSizeInBits(type_access));
    builder.CreateStore(value_size, ptr_size);

    auto ptr_ap = builder.CreateConstGEP2_32(type_mem_params,
                                               argument, 0, 2);
    llvm::Value *value_ap =
        llvm::ConstantInt::get(type_int32, type_ptr->getPointerAddressSpace());
    builder.CreateStore(value_ap, ptr_ap);

    auto ptr_is_write = builder.CreateConstGEP2_32(type_mem_params,
                                                   argument, 0, 3);
    auto value_is_write =
        llvm::ConstantInt::get(type_int8, is_write);
    builder.CreateStore(value_is_write, ptr_is_write);

    // invoke callback.
    llvm::SmallVector<llvm::Value *, 1> args(1, argument);
    builder.CreateCall(mem_handler, args);
  }

  void visitCallInst(llvm::CallInst &call) {
    auto func = call.getCalledFunction();
    if (func == nullptr) {
      LOG(WARNING) << "Indirect function call not supported yet.";
      return;
    }

    auto name = func->getName();
    LLVM_STRING(nvvm);
    LLVM_STRING(atomic);
    if (name.contains(nvvm) && name.contains(atomic)) {
      LOG(INFO) << "Calling atomic " << std::string(name);
      CHECK(call.getNumArgOperands() > 0);
      insertMemHandler(call, call.getArgOperand(0), true, true);
    }
  }

  void visitAtomicCmpXchg(llvm::AtomicCmpXchgInst &cmpxchg) {
    insertMemHandler(cmpxchg, cmpxchg.getPointerOperand(), true, true);
  }

  void visitAtomicRMW(llvm::AtomicRMWInst &atomicrmw) {
    insertMemHandler(atomicrmw, atomicrmw.getPointerOperand(), true, true);
  }

  void visitStoreInst(llvm::StoreInst &st) {
    insertMemHandler(st, st.getPointerOperand(), true);
  }

  void visitLoadInst(llvm::LoadInst &ld) {
    insertMemHandler(ld, ld.getPointerOperand(), false);
  }

  llvm::Function *parent;
  llvm::AllocaInst *mem_params;
};


llvm::StructType *
MemoryAnalysisPass::DefineHandlerParamsType(llvm::Module& module) {
  auto type_mem_params = module.getTypeByName(MEM_PARAMS_TYPENAME);
  if (type_mem_params) {
    return type_mem_params;
  }

  llvm::LLVMContext &ctx = module.getContext();

  llvm::Type *type_int64 = llvm::IntegerType::get(ctx, 64);
  llvm::Type *type_int32 = llvm::IntegerType::get(ctx, 32);
  llvm::Type *type_int8 = llvm::IntegerType::get(ctx, 8);

  llvm::StringRef type_name(MEM_PARAMS_TYPENAME);
  llvm::Type *type_array[] = {type_int64, type_int64, type_int32, type_int8};
  llvm::ArrayRef<llvm::Type *> struct_types(type_array, 4);
  return llvm::StructType::create(struct_types, type_name);
}

bool MemoryAnalysisPass::runOnModule(llvm::Module& module) {
  const KernelInfoPass &kernel_info = getAnalysis<KernelInfoPass>();
  if (!kernel_info.IsCudaModule(module)) {
    LOG(FATAL) << "MemoryAnalysisPass Cannot run on non-cuda module!";
    return false;
  }

  // Add a type for MemParams.
  auto type_mem_params = DefineHandlerParamsType(module);

  // Declare a device function prototype:
  // __device__ void before_branch_handler(CondBranchParams *);
  llvm::LLVMContext &ctx = module.getContext();
  llvm::SmallVector<llvm::Type *, 1> param_types(1,
      llvm::PointerType::get(type_mem_params, GENERIC_ADDR_SPACE));
  llvm::StringRef func_name(BEFORE_MEM_HANDLER_FUNCNAME);
  auto type_void = llvm::Type::getVoidTy(ctx);
  auto type_func = llvm::FunctionType::get(type_void, param_types, false);
  auto before_mem_handler =
      module.getOrInsertFunction(func_name, type_func);

  for (llvm::Function &func : module) {
    if (func.isDeclaration()) {
      continue;
    }

    MemoryAccessVisitor mem_visitor(&func);
    // LOG(INFO) << "Run BranchInstVisitor On " << func.getName().str();
    mem_visitor.visit(func);
  }
}

void MemoryAnalysisPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const
{
  AU.addRequired<KernelInfoPass>();
	AU.setPreservesCFG();
}

// LLVM uses the address of this static member to identify the pass, so the
// initialization value is unimportant.
char MemoryAnalysisPass::ID = 1;

static llvm::RegisterPass<MemoryAnalysisPass> X("MemoryAnalysisPass",
    "Dynamic analysis of memory accesses in cuda module.");

}  // namespace
