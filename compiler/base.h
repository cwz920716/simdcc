#ifndef _CUDA_BASE_H
#define _CUDA_BASE_H

#define NVPTX_NVIDIA_CUDA "nvptx-nvidia-cuda"
#define NVPTX64_NVIDIA_CUDA "nvptx64-nvidia-cuda"

#define NVPTX_DATA_LAYOUT "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
#define NVPTX64_DATA_LAYOUT "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

#define NVVM_ANNOTATIONS "nvvm.annotations"

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
  bool is_backedge;
};  // struct CondBranchParams

}  // extern "C"

#endif
