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

#define WARP_SZ 32
DEVICE
inline int lane_id(void) { return threadIdx.x % WARP_SZ; }

}  // glang

#endif  // __COMMON_SYNTHESIS_BASE_H
