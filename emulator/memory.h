#ifndef _MEMORY_H
#define _MEMORY_H

#include <cstring>
#include <string>
#include <stdint.h>
#include <stdlib.h>
#include <vector>

namespace simdsim {

class BasicMemory {
 public:
  BasicMemory(int64_t size): size_(size), data_(nullptr) {}

  int Size() const { return size_; }
  int8_t ReadByte(int64_t addr) const;
  void WriteByte(int64_t addr, int8_t data);
  void Mmap(void *ptr);

 protected:
  int64_t size_;
  void *data_;
};

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
  NdMemory(int64_t size, Type type, int dim):
      BasicMemory(size), type_(type), dim_(dim) {
    shape_ = new int64_t[dim];
    strides_ = new int64_t[dim];
  }

  ~NdMemory() {
    delete shape_;
    delete strides_;
  }

  void SetShape(const int64_t *shape, const int64_t *strides = nullptr);

  template<class T>
  void ReadElement(const int64_t *addr, T *data) const;
  template<class T>
  void WriteElement(const int64_t *addr, T data);

 private:
  Type type_;
  int dim_;
  int64_t *shape_;
  int64_t *strides_;
};

}  // namespace simdsim

#endif  // _MEMORY_H
