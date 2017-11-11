#ifndef __COMMON_SYNTHESIS_FOR_H
#define __COMMON_SYNTHESIS_FOR_H

#include "synthesis/base.h"
#include "synthesis/iteratable.h"

#include <nvfunctional>

namespace glang {

template<typename T>
struct ForEach {
 public:
  DEVICE void operator() (Iteratable<T> *Input,
                          nvstd::function<void(int)> exec) {
    for (int i = Input->start(); i < Input->end(); i += Input->step()) {
      exec(i);
      step_watch();
    }
  }

  DEVICE void operator() (Slice Input,
                          nvstd::function<void(T &)> exec) {
    for (int i = Input.start(); i < Input.end(); i += Input.step()) {
      exec(i);
      step_watch();
    }
  }

  DEVICE void operator() (DynArray<T> Input,
                          nvstd::function<void(T &)> exec) {
    for (int i = Input.start(); i < Input.end(); i += Input.step()) {
      T item = Input.reference(i);
      exec(item);
      step_watch();
    }
  }

 private:
  DEVICE void step_watch() {
    // RAISE("step.\n");
  }
};

template<typename T, Scope S, bool NO_DIVERGE=false>
struct Parfor {
#define declare_loop_vars(i_start, i_end, i_step) \
    auto start = (i_step) * self_id<S>() + (i_start); \
    auto end = end_cond(i_end); \
    auto step = (i_step) * each_step();

 public:
  DEVICE void operator() (Iteratable<T> *Input,
                          nvstd::function<void(int)> exec) {
    declare_loop_vars(Input->start(), Input->end(), Input->step());
    for (int i = start; i < end; i += step) {
      exec(i);
    }
  }

  DEVICE void operator() (Slice Input,
                          nvstd::function<void(T &)> exec) {
    declare_loop_vars(Input.start(), Input.end(), Input.step());
    for (int i = start; i < end; i += step) {
      exec(i);
    }
  }

  DEVICE void operator() (DynArray<T> Input,
                          nvstd::function<void(T &)> exec) {
    declare_loop_vars(Input.start(), Input.end(), Input.step());
    for (int i = start; i < end; i += step) {
      T item = Input.reference(i);
      exec(item);
    }
  }

 private:
  DEVICE int each_step() {
    return scope_size<S>();
  }

  DEVICE int end_cond(int end) {
    if (NO_DIVERGE) {
      return round_up(end, each_step());
    } else {
      return end;
    }
  }
};

}  // namespace glang

#endif  // __COMMON_SYNTHESIS_FOR_H
