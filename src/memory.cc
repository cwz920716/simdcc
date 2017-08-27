#include "memory.h"

namespace simdsim {

int8_t BasicMemory::ReadByte(int addr) const {
  int8_t *ptr = static_cast<int8_t *>(data_);
  return ptr[addr];
}

void BasicMemory::WriteByte(int addr, int8_t data) {
  int8_t *ptr = static_cast<int8_t *>(data_);
  ptr[addr] = data;
}

void BasicMemory::Mmap(void *ptr) {
  data_ = ptr;
}

}  // namespace simdsim
