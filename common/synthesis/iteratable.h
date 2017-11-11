#ifndef __COMMON_SYNTHESIS_ITERATABLE_H
#define __COMMON_SYNTHESIS_ITERATABLE_H

#include "synthesis/base.h"

namespace glang {

template<typename T>
class Iteratable {
 public:
  virtual GLANG int start() const = 0;
  virtual GLANG int end() const = 0;
  virtual GLANG int step() const = 0;

  virtual GLANG int size() const {
    return (end() - start()) / step();
  }

  virtual GLANG int translate(int idx) {
    return start() + idx * step();
  }

  virtual GLANG T reference(int idx) = 0;

  virtual GLANG T operator[](int idx) {
    return reference(idx);
  }

  virtual GLANG bool readonly() const {
    return false;
  }
};

class Slice: public Iteratable<int> {
 public:
  GLANG Slice():
    start_(-1), end_(-1), step_(1) {}
  GLANG Slice(int start, int end, int step = 1):
    start_(start), end_(end), step_(step) {
    if (end_ < start_) {
      end_ = start_;
    }
  }

  virtual GLANG int start() const {
    return start_;
  }

  virtual GLANG int end() const {
    return end_;
  }

  virtual GLANG int step() const {
    return step_;
  }

  virtual GLANG int reference(int idx) {
    return translate(idx);
  }

  virtual GLANG bool readonly() const {
    return true;
  }

 private:
  int start_, end_, step_;
  int key_;
};

template<typename T>
class DynArray: public Iteratable<T> {
 public:
  DynArray():
    size_(0), data_(nullptr) {}
  DynArray(int size, T *data):
    size_(size), data_(data) {}
  DynArray(int size, T *data, T &halo):
    size_(size), data_(data), halo_(halo) {}

  virtual GLANG int start() const {
    return 0;
  }

  virtual GLANG int end() const {
    return size_;
  }

  virtual GLANG int step() const {
    return 1;
  }

  virtual GLANG T *data() const {
    return data_;
  }

  virtual GLANG T reference(int idx) {
    if (!valid(idx)) {
      return halo_;
    }

    return data_[idx];
  }

 protected:
  virtual GLANG bool valid(int idx) const { 
    return idx >= start() && idx < end();
  }

  int size_;
  T *data_;
  T halo_;
};

}  // namespace glang

#endif  // __COMMON_SYNTHESIS_ITERATABLE_H
