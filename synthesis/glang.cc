// GPU Graph Traversal Language (glang)

#include <string>
#include <sstream>
#include <iostream>
#include <cinttypes>

#include <glog/logging.h>

using namespace std;

namespace glang {

enum DataType {
  Nil,
  Bool,
  Integer,
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

enum Scope {
  Thread,
  Warp,
  Threadblock,
  Device,
  Constant,
};

enum Operator {
  CXXRestricted,  // restricted straight line cxx statement operate on local
                  // data, no memory allocation/dereference
  Load,
  Store,
  AtomicOp,
  MemFence,
  // Control flow
  ParallelFor,
  For,
  For_uni,
  While,
  While_uni,
  If,
  If_uni,
  // Intrinsics
  Any,
  Ballot,
  // Programming Paradiagms
  Enlist,
  Singleton,
  Barrier,
  // Helper 
  InclusiveScan,
  ExclusiveScan,
  Reduce,
  RescourceAllocation,
};

static int64_t values = 0;
static int64_t value_id_gen(void) {
  return values++;
}

class Value {
 public:
  Value(int64_t id, DataType type, Scope scope, const string &name): id_(id), type_(type), scope_(scope), name_(name) {}
  Value(DataType type, Scope scope, const string &name): id_(-1), type_(type), scope_(scope), name_(name) { id_ = value_id_gen(); }
  Value(DataType type, Scope scope): id_(-1), type_(type), scope_(scope) { id_ = value_id_gen(); }

  int id() const { return id_; }
  string name() const { return name_; }
  DataType type() const { return type_; }
  Scope scope() const { return scope_; }

 protected:
  int64_t id_;
  DataType type_;
  Scope scope_;
  string name_;
};

// non-const int OR symbolic int
class IntValue: public Value {
 public:
  IntValue(Scope scope, string name, int bitwidth, bool is_constant = false):
      Value(Integer, scope, name), bitwidth_(bitwidth),
      is_constant_(is_constant) {}

  int bitwidth() const { return bitwidth_; }
  bool isConstant() const { return is_constant_; }

  virtual int64_t value() const {
    CHECK(false) << "Non-const IntValue cannot return value!";
    return 0;
  }

 private:
  int bitwidth_;
  bool is_constant_;
};

class ConstantInt: public IntValue {
 public:
  ConstantInt(int64_t value, int bitwidth = 32):
      IntValue(Constant, to_string(value), bitwidth, true), value_(value) {}

  int64_t value() const { return value_; }

  static ConstantInt *Zero, *One;

 private:
  int64_t value_;
};

ConstantInt *CreateConstInt(int64_t v, int bw = 32) {
  auto p = new ConstantInt(v, bw);
  // store p into a global map to reduce memory
  return p;
}

class IteratableValue: public Value {
 public:
  IteratableValue(Scope scope, const string &name):
      Value(Iteratable, scope, name) {}

  virtual IntValue *start(void) const = 0;
  virtual IntValue *end(void) const = 0;
  virtual IntValue *step(void) const = 0;

  virtual Value *reference(IntValue *iter) = 0;
};

class Slice: public IteratableValue {
 public:
  Slice(IntValue *start, IntValue *end, IntValue *step,
        Scope scope, const string &name):
      IteratableValue(scope, name), start_(start), end_(end), step_(step) {}

  virtual IntValue *start(void) const { return start_; }
  virtual IntValue *end(void) const { return end_; }
  virtual IntValue *step(void) const { return step_; }

  virtual Value *reference(IntValue *iter) {
    CHECK(iter);
    return iter;
  }

 private:
  IntValue *start_, *end_, *step_; 
};

Slice *CreateConstSlice(int start, int end, int step = 1) {
  auto c_start = CreateConstInt(start);
  auto c_end = CreateConstInt(end);
  auto c_step = CreateConstInt(step);

  string name = "slice";
  name += "[" + c_start->name() + ":"
          + c_end->name() + ":" + c_step->name() + "]";
  auto slice = new Slice(c_start, c_end, c_step, Constant, name);
  CHECK(slice != nullptr);
  return slice;
}

class DynArray: public IteratableValue {
 public:
  DynArray(Scope scope, string name, DataType dtype, IntValue *length):
      IteratableValue(scope, name), dtype_(dtype), length_(length) {}

  virtual IntValue *start(void) const { return ConstantInt::Zero; }
  virtual IntValue *end(void) const { return length_; }
  virtual IntValue *step(void) const { return ConstantInt::One; }

 private:
  DataType dtype_;
  IntValue *length_;
};

class Operation: public Value {
 public:
  Operation(Scope scope, Operator op, DataType type = Nil):
    Value(type, scope), op_(op) {}

 private:
  Operator op_;
};

ConstantInt *ConstantInt::Zero, *ConstantInt::One;

void InitGlang(void) {
  ConstantInt::Zero = CreateConstInt(0);
  ConstantInt::One = CreateConstInt(1);
}

}  // namespace glang

using namespace glang;

int main(int argc, char **argv) {
  // google::InitGoogleLogging(argv[0]);
  auto c0 = CreateConstInt(0);
  LOG(INFO) << "c0 = " << c0->name();
  auto s0 = CreateConstSlice(0, 128);
  LOG(INFO) << "s0 = " << s0->name();
  return 0;
}
