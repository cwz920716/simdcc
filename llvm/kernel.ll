; ModuleID = 'ptx tutorial'
source_filename = "ptx tutorial"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

define void @kernel(float addrspace(1)* %A, float addrspace(1)* %B, float addrspace(1)* %C) {
entry:
  %id = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  %ptrA = getelementptr float, float addrspace(1)* %A, i32 %id
  %ptrB = getelementptr float, float addrspace(1)* %B, i32 %id
  %ptrC = getelementptr float, float addrspace(1)* %C, i32 %id
  %valA = load float, float addrspace(1)* %ptrA, align 4
  %valB = load float, float addrspace(1)* %ptrB, align 4
  %valC = fadd float %valA, %valB
  store float %valC, float addrspace(1)* %ptrC, align 4
  ret void
}

attributes #0 = { nounwind readnone }

!nvvm.annotations = !{!0}

!0 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*)* @kernel, !"kernel", i32 1}
