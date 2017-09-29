#ifndef __COMPILER_LLVM_UTILS_H
#define __COMPILER_LLVM_UTILS_H

#define LLVM_STRING(name) \
  llvm::StringRef name(#name)

#endif  // __COMPILER_LLVM_UTILS_H
