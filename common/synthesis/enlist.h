#ifndef __COMMON_SYNTHESIS_ENLIST_H
#define __COMMON_SYNTHESIS_ENLIST_H

#include "synthesis/base.h"

#include <nvfunctional>

namespace glang {

template<typename T, int BLOCK_DIM_X>
struct WarpEnlist {
 public:
  static constexpr int WARP_PER_BLOCK = round_up(BLOCK_DIM_X, WARP_SZ) / WARP_SZ;

  struct TempStorage {
    T comm[WARP_PER_BLOCK];
  };

  DEVICE WarpEnlist(TempStorage &ts): ts_(ts) {}

  DEVICE void until(bool valid,
                    T &arg,
                    nvstd::function<void(int,T &)> exec) {
    while(true) {
      auto ret = (*this)(valid, arg, exec);
      if (ret == false) {
        break;
      }
    }

    return;
  }

  DEVICE bool operator() (bool &valid,
                          T &arg,
                          nvstd::function<void(int,T &)> exec) {
    auto activemask = __ballot(valid);
    if (activemask == 0) {
      return false;
    }

    auto leader = __ffs(activemask) - 1;

    if (lane_id() == leader) {
      ts_.comm[warp_id()] = arg;
      valid = false;
    }

    exec(leader, ts_.comm[warp_id()]);
    return true;
  }

 private:
  TempStorage &ts_;
};

template<typename T, int BLOCK_DIM_X>
struct BlockEnlist {
 public:
  struct TempStorage {
    int leader;
    T comm;
  };

  DEVICE BlockEnlist(TempStorage &ts): ts_(ts) {}

  DEVICE void until(bool valid,
                    T &arg,
                    nvstd::function<void(int,T &)> exec) {
    while(true) {
      auto ret = (*this)(valid, arg, exec);
      __syncthreads();

      if (ret == false) {
        break;
      }
    }

    __syncthreads();
    return;
  }

  DEVICE bool operator() (bool &valid,
                          T &arg,
                          nvstd::function<void(int,T &)> exec) {
    if (threadIdx.x == 0) {
      ts_.leader = BLOCK_DIM_X;
    }
    __syncthreads();

    if (valid) {
      ts_.leader = threadIdx.x;
    }
    __syncthreads();

    if (ts_.leader == BLOCK_DIM_X) {
      return false;
    }
    __syncthreads();

    if (ts_.leader == threadIdx.x) {
      ts_.comm = arg;
      valid = false;
    }
    __syncthreads();

    exec(ts_.leader, ts_.comm);
    return true;
  }

 private:
  TempStorage &ts_;
};

}  // namespace glang

#endif  // __COMMON_SYNTHESIS_ENLIST_H
