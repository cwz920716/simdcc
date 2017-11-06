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
    int idx = self_id() + Input->start();
    exec(idx);
  }

  DEVICE void operator() (DynArray<T> Input,
                          nvstd::function<void(T &)> exec) {
    int idx = self_id() + Input.start();
    T &item = Input.reference(idx);
    exec(item);
  }

 private:
  DEVICE int self_id() {
    switch(S) {
      case kWarpScope: return lane_id();
      case kBlockScope: return threadIdx.x;
      case kDeviceScope: return blockIdx.x * blockDim.x + threadIdx.x;
    }
    return 0;
  }
};

}  // namespace glang

#endif  // __COMMON_SYNTHESIS_EXPAND_H
