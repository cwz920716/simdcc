#ifndef __COMMON_SYNTHESIS_POOL_H
#define __COMMON_SYNTHESIS_POOL_H

#include "synthesis/base.h"
#include "cub/cub.cuh"
#include "cub/util_type.cuh"

namespace glang {

template<Scope S, int BLOCK_DIM_X>
struct BlockPool {
 public:
  typedef int Int;

  /// Define the delegate type for the desired algorithm
  typedef typename cub::If<(S == kWarpScope),
      cub::WarpScan<Int>,
      cub::BlockScan<Int, BLOCK_DIM_X>>::Type Scan;
  typedef typename cub::If<(S == kWarpScope),
      cub::WarpScan<Int>,
      cub::BlockScan<Int, BLOCK_DIM_X>>::Type::TempStorage ScanTempStorage;

  static constexpr int POOL_SZ = BLOCK_DIM_X / ((S == kWarpScope) ? WARP_SZ : BLOCK_DIM_X);

  struct TempStorage {
    ScanTempStorage scan_storage[POOL_SZ];
    Int total_tokens[POOL_SZ];
  };

  DEVICE BlockPool(TempStorage &ts): ts_(ts) {}

  DEVICE void claim(Int items, Int &rsv_rank, Int &total) {
    Scan(ts_.scan_storage[pool_id()]).ExclusiveSum(items, rsv_rank, total);
  }

 private:
  DEVICE int pool_id() {
    return self_id<kBlockScope>() / scope_size<S>();
  }

  TempStorage &ts_;
};

}  // namespace glang

#endif  // __COMMON_SYNTHESIS_POOL_H
