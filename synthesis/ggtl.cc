// GPU Graph Traversal Language

#include <string>
#include <sstream>
#include <iostream>

using namespace std;

namespace ggtl {

enum DataType {
  Bool,
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
  Iteratable,
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
  MemFence,
  // Control flow
  ParallelFor,
  For,
  Enlist,
  Singleton,
  UniformIf,
  If,
  Barrier,
  // Helper 
  InclusiveScan,
  ExclusiveScan,
  Reduce,
  RescourceAllocation,
};

class Value {
 public:
  Value(int id, DataType type, Scope scope, const string &name): id_(id), type_(type), scope_(scope), name_(name) {}

 private:
  int id_;
  DataType type_;
  Scope scope_;
  string name_;
};

static int valuesi = 0;
static int value_id_gen(void) {
  return values++;
}

}  // namespace ggtl
