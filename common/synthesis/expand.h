#ifndef __COMMON_SYNTHESIS_EXPAND_H
#define __COMMON_SYNTHESIS_EXPAND_H

#include "synthesis/base.h"
#include "synthesis/iteratable.h"

#include <nvfunctional>

namespace glang {

template<typename T, Scope S>
struct Expand {
 public:
  DEVICE void operator() (Iteratable<T> *Input,
                          nvstd::function<void(int)> exec) {
    int idx = self_id<S>() * Input->step() + Input->start();
    exec(idx);
  }

  DEVICE void operator() (Slice Input,
                          nvstd::function<void(T &)> exec) {
    int idx = self_id<S>() * Input.step() + Input.start();
    exec(idx);
  }

  DEVICE void operator() (DynArray<T> Input,
                          nvstd::function<void(T &)> exec) {
    int idx = self_id<S>() * Input.step() + Input.start();
    T item = Input.reference(idx);
    exec(item);
  }

 private:
};

}  // namespace glang

#endif  // __COMMON_SYNTHESIS_EXPAND_H
