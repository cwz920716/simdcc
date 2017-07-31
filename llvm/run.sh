#! /bin/sh

clang++ -g -O3 tutorial.cc `llvm-config --cxxflags --ldflags --system-libs --libs core`
./a.out > kernel.ll
llc -mcpu=sm_35 kernel.ll -o kernel.ptx
