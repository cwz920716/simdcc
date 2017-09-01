#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace llvm;

static LLVMContext TheContext;
static IRBuilder<> Builder(TheContext);
static std::unique_ptr<Module> TheModule;

int main() {
  TheModule = llvm::make_unique<Module>("ptx tutorial", TheContext);

  StringRef data_layout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
  StringRef target_triple("nvptx64-nvidia-cuda");
  TheModule->setDataLayout(data_layout);
  TheModule->setTargetTriple(target_triple);

  auto Int32 = IntegerType::get(TheContext, 32);
  ArrayRef< Type *> param_types;
  auto read_tid_x_type = FunctionType::get(Int32, param_types, false);
  StringRef read_tid_x_name("llvm.nvvm.read.ptx.sreg.tid.x");
  llvm::AttributeList attr_list;
  attr_list = attr_list.addAttribute(TheContext, 0, Attribute::ReadNone);
  attr_list = attr_list.addAttribute(TheContext, 1, Attribute::NoUnwind);
  auto read_tid_x = TheModule->getOrInsertFunction(read_tid_x_name, read_tid_x_type, attr_list);

  StringRef nvvm_annotations("nvvm.annotations");
  auto nvvm_annotations_md = TheModule->getOrInsertNamedMetadata(nvvm_annotations);

  auto Void = Type::getVoidTy(TheContext);
  auto Float = Type::getFloatTy(TheContext);
  auto FloatGlobalPtr = PointerType::get(Float, 1);
  SmallVector<Type *, 3> kernel_param_types(3, FloatGlobalPtr);
  auto kernel_type = FunctionType::get(Void, kernel_param_types, false);
  StringRef kernel_name("kernel");
  std::map<std::string, Value *> NamedValues;
  Function *kernel_func = Function::Create(kernel_type, Function::ExternalLinkage, kernel_name, TheModule.get());
  std::string names[] = {"A", "B", "C"};
  int i = 0;
  for(auto &arg : kernel_func->args()) {
    NamedValues[names[i]] = &arg;
    arg.setName(names[i++]);
  }

  BasicBlock *BB = BasicBlock::Create(TheContext, "entry", kernel_func);
  Builder.SetInsertPoint(BB);
  auto get_tid_x = Builder.CreateCall(read_tid_x, None, "id");
  get_tid_x->setTailCall();
  get_tid_x->setDoesNotAccessMemory();
  get_tid_x->setDoesNotThrow();
  auto ptrA = Builder.CreateGEP(Float, NamedValues["A"], get_tid_x);
  ptrA->setName("ptrA");
  auto ptrB = Builder.CreateGEP(Float, NamedValues["B"], get_tid_x);
  ptrB->setName("ptrB");
  auto ptrC = Builder.CreateGEP(Float, NamedValues["C"], get_tid_x);
  ptrC->setName("ptrC");
  auto valA = Builder.CreateLoad(ptrA, "valA");
  valA->setAlignment(4);
  auto valB = Builder.CreateLoad(ptrB, "valB");
  valB->setAlignment(4);
  auto valC = Builder.CreateFAdd(valA, valB, "valC");
  auto storeC = Builder.CreateStore(valC, ptrC);
  storeC->setAlignment(4);
  Builder.CreateRetVoid();

  auto md0_0 = ValueAsMetadata::getConstant(kernel_func);
  StringRef kernel_str("kernel");
  auto md0_1 = MDString::get(TheContext, kernel_str);
  APInt i1(32, 1);
  auto One = ConstantInt::get(Int32, i1);
  auto md0_2 = ValueAsMetadata::getConstant(One);
  Metadata *metas[] = { md0_0, md0_1, md0_2 };
  ArrayRef<Metadata *> metas_(metas, 3);
  auto md0 = MDNode::get(TheContext, metas_);
  nvvm_annotations_md->addOperand(md0); 

  TheModule->print(outs(), nullptr);

  return 0;
}
