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
static std::map<std::string, Value *> NamedValues;

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
  llvm::AttributeSet attr_list;
  attr_list.addAttribute(TheContext, 0, Attribute::ReadNone);
  attr_list.addAttribute(TheContext, 1, Attribute::NoUnwind);
  auto read_tid_x = TheModule->getOrInsertFunction(read_tid_x_name, read_tid_x_type, attr_list);

  StringRef nvvm_annotations("nvvm.annotations");
  auto nvvm_annotations_md = TheModule->getOrInsertNamedMetadata(nvvm_annotations);

  auto Void = Type::getVoidTy(TheContext);
  auto kernel_type = FunctionType::get(Void, param_types, false);
  StringRef kernel_name("kernel");
  Function *kernel_func = Function::Create(kernel_type, Function::ExternalLinkage, kernel_name, TheModule.get());
  BasicBlock *BB = BasicBlock::Create(TheContext, "entry", kernel_func);
  Builder.SetInsertPoint(BB);
  Builder.CreateRetVoid();

  // auto md0_0 = MDNode::get(TheContext, ValueAsMetadata::get(TheModule->getFunction(kernel_name)));
  auto md0_0 = ValueAsMetadata::getConstant(kernel_func);
  StringRef kernel_str("kernel");
  auto md0_1 = MDString::get(TheContext, kernel_str);
  Metadata *metas[] = { md0_0, md0_1 };
  ArrayRef<Metadata *> metas_(metas, 2);
  auto md0 = MDNode::get(TheContext, metas_);
  nvvm_annotations_md->addOperand(md0); 

  TheModule->print(outs(), nullptr);

  return 0;
}
