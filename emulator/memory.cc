#include "memory.h"

namespace simdsim {

int8_t BasicMemory::ReadByte(int64_t addr) const {
  int8_t *ptr = static_cast<int8_t *>(data_);
  return ptr[addr];
}

void BasicMemory::WriteByte(int64_t addr, int8_t data) {
  int8_t *ptr = static_cast<int8_t *>(data_);
  ptr[addr] = data;
}

void BasicMemory::Mmap(void *ptr) {
  data_ = ptr;
}

static void
ComputeDefaultStride(int64_t dim, const int64_t *shape, int64_t *strides) {
  int64_t product = 1;
  for (int i = dim - 1; i >= 0; i--) {
    strides[i] = product;
    product *= shape[i];
  }
}

void
NdMemory::SetShape(const int64_t *shape, const int64_t *strides) {
  memcpy(shape_, shape, dim_ * sizeof(int64_t));
  if (strides) {
    memcpy(strides_, strides, dim_ * sizeof(int64_t));
  } else {
    ComputeDefaultStride(dim_, shape_, strides_);
  }
}

}  // namespace simdsim
