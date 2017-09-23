#include <glog/logging.h>

#include <llvm/IR/AssemblyAnnotationWriter.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FormattedStream.h>

#include "kernel_info_pass.h"

using llvm::isa;
using llvm::cast;
using llvm::dyn_cast;

namespace gpuvm {

namespace {

ModuleType GetModuleType(const llvm::Module &M) {
  auto target_desc = M.getTargetTriple();
  auto data_layout_string = M.getDataLayoutStr();

  if (target_desc == NVPTX64_NVIDIA_CUDA) {
    // CHECK(data_layout_string == NVPTX64_DATA_LAYOUT) << data_layout_string;
    return kNvidiaPtx64;
  } else if (target_desc == NVPTX_NVIDIA_CUDA) {
    CHECK(false) << "NVPTX-32bit is not implemented!";
  }

  return kNonPtx;
}

std::string MD2String(const llvm::Metadata *md) {
  if (auto md_string = dyn_cast<llvm::MDString>(md)) {
    return md_string->getString().str();
  }

  return "";
}

}  // namespace

bool KernelInfoPass::IsCudaModule(llvm::Module& M) const {
  auto mod_type = GetModuleType(M);
  if (mod_type != kNvidiaPtx64 && mod_type != kNvidiaPtx32) {
    return false;
  }

  return true;
}

bool KernelInfoPass::runOnModule(llvm::Module& M) {
  auto mod_type = GetModuleType(M);
  if (mod_type != kNvidiaPtx64) {
    return false;
  }

  LOG(INFO) << "A Ptx64 LLVM Module is found: " << M.getName().str();
  llvm::NamedMDNode *nvvm_annotations = M.getNamedMetadata(NVVM_ANNOTATIONS);
  for (auto md : nvvm_annotations->operands()) {
    if (md->getNumOperands() == 3) {
      auto md0 = md->getOperand(0).get();
      auto md1 = md->getOperand(1).get();
      auto md2 = md->getOperand(2).get();
      if (md0 == nullptr) {
        continue;
      }

      if (auto func_md = dyn_cast<llvm::ConstantAsMetadata>(md0)->getValue()) {
        if (auto func = dyn_cast<llvm::Function>(func_md)) {
          CHECK(isa<llvm::MDString>(md1) && isa<llvm::ConstantAsMetadata>(md2));
          CHECK(MD2String(md1) == "kernel");
        }
      }
    }
  }

  return true;
}

void KernelInfoPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const
{
	AU.setPreservesAll();
}

// LLVM uses the address of this static member to identify the pass, so the
// initialization value is unimportant.
char KernelInfoPass::ID = 0;

static llvm::RegisterPass<KernelInfoPass> X("KernelInfoPass",
    "Info Cuda kernel Functions.");

}  // namespace gpuvm
