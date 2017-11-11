#ifndef __COMMON_SYNTHESIS_GATHER_H
#define __COMMON_SYNTHESIS_GATHER_H

#include "synthesis/base.h"
#include "synthesis/iteratable.h"
#include "synthesis/pool.h"

namespace glang {

#define thread_id (threadIdx.x)

template<typename T, int BLOCK_DIM_X>
struct Gather {
 public:
  typedef BlockPool<kBlockScope, BLOCK_DIM_X> BlockPool;

  struct TempStorage {
    typename BlockPool::TempStorage pool_ts;
    T comm[BLOCK_DIM_X];
  };

  DEVICE Gather(TempStorage &ts): ts_(ts) {}

  DEVICE void operator()(Iteratable<T> &input, nvstd::function<void(T &)> exec) {
    BlockPool pool(ts_.pool_ts);
    int rsv_rank, total;
    // printf("[%d] take size %d\n", thread_id, input.size());
    pool.claim(input.size(), rsv_rank, total);
    int cta_progress = 0;
    int items = input.size(), gathered = 0;
    int remain;
    while ((remain = total - cta_progress) > 0) {
      while (rsv_rank - cta_progress < BLOCK_DIM_X && gathered < items) {
        ts_.comm[rsv_rank - cta_progress] = input[gathered];
        rsv_rank++;
        gathered++;
      }
      __syncthreads();
      if (thread_id < min(remain, BLOCK_DIM_X)) {
        exec(ts_.comm[thread_id]);
      }
      cta_progress += BLOCK_DIM_X;
      __syncthreads();
    }
  }

 private:
  TempStorage &ts_;
};

}  // namespace glang

#endif  // __COMMON_SYNTHESIS_GATHER_H
