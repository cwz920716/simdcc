#ifndef __COMMON_SYNTHESIS_BASE_H
#define __COMMON_SYNTHESIS_BASE_H

#include <stdio.h>

#include <iostream>
#include <cstdio>

#define GLANG      __host__ __device__
#define HOST       __host__
#define DEVICE     __device__

#include "glog/logging.h"

#define RAISE(msg) printf(msg)

namespace glang {

enum Scope {
  kWarpScope,
  kBlockScope,
  kDeviceScope,
};

DEVICE
unsigned int round_up(unsigned int value, unsigned int round_to) {
    return (value + (round_to - 1)) & ~(round_to - 1);
}

#define WARP_SZ 32

DEVICE int lane_id(void) { return threadIdx.x % WARP_SZ; }

template<Scope S>
DEVICE int self_id() {
  switch(S) {
    case kWarpScope: return lane_id();
    case kBlockScope: return threadIdx.x;
    case kDeviceScope: return blockIdx.x * blockDim.x + threadIdx.x;
  }
  return 0;
}

template<Scope S>
DEVICE int scope_size() {
  switch(S) {
    case kWarpScope: return WARP_SZ;
    case kBlockScope: return blockDim.x;
    case kDeviceScope: return gridDim.x * blockDim.x;
  }
  return 0;
}

}  // glang

#endif  // __COMMON_SYNTHESIS_BASE_H
