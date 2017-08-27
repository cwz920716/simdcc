#ifndef MEMORY_H
#define MEMORY_H

#include <stdint.h>

namespace simdsim {

class BasicMemory {
 public:
  BasicMemory(int size) : size_(size), data_(nullptr) {}

  int Size() const { return size_; }
  int8_t ReadByte(int addr) const;
  void WriteByte(int addr, int8_t data);
  void Mmap(void *ptr);

 protected:
  int size_;
  void *data_;
};

}  // namespace simdsim

#endif
