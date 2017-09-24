#ifndef _CUDA_BASE_H
#define _CUDA_BASE_H

#define NVPTX_NVIDIA_CUDA "nvptx-nvidia-cuda"
#define NVPTX64_NVIDIA_CUDA "nvptx64-nvidia-cuda"

#define NVPTX_DATA_LAYOUT "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
#define NVPTX64_DATA_LAYOUT "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

#define NVVM_ANNOTATIONS "nvvm.annotations"

#define GENERIC_ADDR_SPACE (0)
#define LOCAL_ADDR_SPACE (5)

#include <iostream>
#include <string>
#include <functional>

#include <llvm/Pass.h>
#include <llvm/IR/Instructions.h>

extern "C" {

#define FILE_NAME_MAX 128

enum InstructionType {
  kMemRead = 0,
  kMemWrite,
  kCondBranch,
  kDefault
};

struct InstParams {
  int32_t id;
  int32_t type;
  bool will_execute;
  int32_t line_no;
  char file_name[FILE_NAME_MAX];
};  // struct InstParams

struct CondBranchParams {
  int32_t id;
  bool taken;
  bool is_conditional;
};  // struct CondBranchParams

}  // extern "C"

namespace gpuvm {

using std::to_string;

class InstStatistics {
 public:
  InstStatistics() {}

  inline int Record(llvm::Instruction *inst) {
    int id = instructions.size();
    inst_id[inst] = id;
    instructions.push_back(inst);
    return id;
  }

  inline int GetInstructionId(llvm::Instruction *inst) {
    if (inst_id.find(inst) != inst_id.end()) {
      return -1;
    }

    return inst_id[inst];
  }

  inline std::string DebugStr(void) {
    std::string db = "";
    for (auto inst : instructions) {
      int id = inst_id[inst];
      db += "branch(" + to_string(id) + "): line "
            + to_string(inst->getDebugLoc().getLine()) + "\n";
    }
    return db;
  }

  std::vector<llvm::Instruction *> instructions;
  std::map<llvm::Instruction *, int> inst_id;
};

}  // namespace gpuvm

#endif
