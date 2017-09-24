#ifndef _MEMORY_H
#define _MEMORY_H

#include <stdint.h>

namespace simdsim {

class BasicMemory {
 public:
  BasicMemory(int size): size_(size), data_(nullptr) {}

  int Size() const { return size_; }
  int8_t ReadByte(int addr) const;
  void WriteByte(int addr, int8_t data);
  void Mmap(void *ptr);

 protected:
  int size_;
  void *data_;
};

void ComputeDefaultStride(int dim, int *shape, int *strides);

enum Type {
  Int8,
  Int16,
  Int32,
  Int64,
  Float32,
  Float64,
};

// A mmeory model for Nd-array
class NdMemory: public BasicMemory {
 public:
  NdMemory(int size, Type type, int dim):
      BasicMemory(size), type_(type), dim_(dim) {}

  void SetShape(const int *shape, const int *strides = nullptr);

  template<class T>
  void ReadElement(const int *addr, T *data) const;
  template<class T>
  void WriteElement(const int *addr, T data);

 private:
  Type type_;
  int dim_;
  int *shape_;
  int *strides_;
};

}  // namespace simdsim

#endif  // _MEMORY_H
