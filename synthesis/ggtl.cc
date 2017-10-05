// GPU Graph Traversal Language

namespace ggtl {

enum DataType {
  Int8,
  Int32,
  Int64,
  Float,
  Float64,
  Array,
  // Assist types
  Struct,
  NdArray,
  // GGTL Specific types
  WorkList,
  Task,
};

enum AddressSpace {
  Generic,
  

enum ParallelScope {
  Sequential,
  Warp,
  Threadblock,
  Device,
};

enum ParallelOperator {
  ParallelFor,
  RescourceAlloc,
  AtomicOp,
  InclusiveScan,
  ExclusiveScan,
  Reduce,
  Load,
  Store,
};


}  // namespace ggtl
