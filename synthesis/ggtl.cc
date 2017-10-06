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

enum AddrSpace {
  Generic,
  Global,
  Internal,
  Shared,
  Constant,
  Local,
};

enum Scope {
  Thread,
  Warp,
  Threadblock,
  Device,
};

enum Operator {
  CXXRestricted,  // restricted straight line cxx statement operate on local data, no memory allocation/dereference
  Load,
  Store,
  AtomicOp,
  // Control flow
  ParallelFor,
  For,
  Singleton,
  UniformIf,
  If,
  // Helper 
  InclusiveScan,
  ExclusiveScan,
  Reduce,
  RescourceAllocation,
};


}  // namespace ggtl
