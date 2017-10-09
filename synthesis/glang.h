#ifndef _SYNTHESIS_GLANG_H
#define _SYNTHESIS_GLANG_H

// GPU Graph Traversal Language (glang)

#include <string>
#include <sstream>
#include <iostream>
#include <cinttypes>
#include <map>
#include <vector>

#include <glog/logging.h>

#define THREAD_IDX "thread_idx"

using namespace std;

namespace glang {

enum TypeCode {
  Nil,
  Integer,
  Float,
  Double,
  Pointer,
  Array,
  Struct,
  NdArray,
  // GGTL Specific types
  Iteratable,
  WorkList,
  Task,
};

class Type {
 public:
  Type(TypeCode typecode, string name = "anonymous"):
    typecode_(typecode), name_(name) {}

  virtual string nice_str() {
    std::string ty = "typeof(";
    ty += to_string(typecode_) + ")";
    return ty;
  }

 protected:
  TypeCode typecode_;
  string name_;
};

using DataType = Type *;
extern DataType NilTy;

class IntType: public Type {
 public:
  IntType(int bitwidth): Type(Integer), bitwidth_(bitwidth) {}

  virtual string nice_str() {
    if (bitwidth_ == 1) {
      return "bool";
    }

    std::string ty = "int";
    ty += to_string(bitwidth_) + "_t";
    return ty;
  }

  static IntType *GetIntegerTy(int bw = 32);

 private:
  int bitwidth_;
};

class PointerType: public Type {
 public:
  PointerType(DataType dtype): Type(Pointer), dtype_(dtype) {}

  virtual string nice_str() {
    std::string ty = dtype_->nice_str() + "*";
    return ty;
  }

  DataType dtype() const { return dtype_; }

  static PointerType *GetPointerTy(DataType dtype);

 private:
  DataType dtype_;
};

class IteratableType: public Type {
 public:
  IteratableType(const std::string &name, DataType dtype):
      Type(Iteratable, name), dtype_(dtype) {}

  virtual string nice_str() {
    std::string ty = name_ + "<" + dtype_->nice_str() + ">";
    return ty;
  }

  DataType dtype() const { return dtype_; }

 private:
  DataType dtype_;
};

enum Scope {
  Thread,
  Warp,
  ThreadBlock,
  Device,
  Constant,
};

extern string ScopeDesc(Scope s);

extern int64_t value_id_gen(void);

class Value {
 public:
  Value(int64_t id, DataType type, Scope scope, const string &name): id_(id), type_(type), scope_(scope), name_(name) {}
  Value(DataType type, Scope scope, const string &name): id_(-1), type_(type), scope_(scope), name_(name) { id_ = value_id_gen(); }
  Value(DataType type, Scope scope): id_(-1), type_(type), scope_(scope) { id_ = value_id_gen(); }

  int id() const { return id_; }
  string name() const { return name_; }
  DataType type() const { return type_; }
  Scope scope() const { return scope_; }

  virtual string nice_str() { return name_; }

 protected:
  int64_t id_;
  DataType type_;
  Scope scope_;
  string name_;
};

// non-const int OR symbolic int
class IntValue: public Value {
 public:
  IntValue(Scope scope, string name, int bitwidth = 32, bool is_constant = false):
      Value(IntType::GetIntegerTy(bitwidth), scope, name),
      is_constant_(is_constant) {}

  static IntValue *CreateIntValue(Scope s, string name, int bw = 32);

  bool isConstant() const { return is_constant_; }

  virtual int64_t value() const {
    CHECK(false) << "Non-const IntValue cannot return value!";
    return 0;
  }

 private:
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

class PointerValue: public Value {
 public:
  PointerValue(Scope scope, string name, DataType dtype, int ap = 0):
      Value(PointerType::GetPointerTy(dtype), scope, name),
      addr_space_(ap) {}

  static PointerValue *CreatePtrValue(Scope s, string name, DataType dtype);

  int addr_space() const { return addr_space_; }
  DataType dtype() const {
    PointerType *ty = dynamic_cast<PointerType *> (type_);
    return ty->dtype();
  }

 private:
  int addr_space_;
};

class IteratableValue: public Value {
 public:
  IteratableValue(DataType type, Scope scope, const string &name):
      Value(type, scope, name) {}

  virtual IntValue *start(void) const = 0;
  virtual IntValue *end(void) const = 0;
  virtual IntValue *step(void) const = 0;

  virtual Value *reference(IntValue *iter) = 0;
};

extern IteratableType *SliceTy;
class Slice: public IteratableValue {
 public:
  Slice(IntValue *start, IntValue *end, IntValue *step,
        Scope scope, const string &name):
      IteratableValue(SliceTy, scope, name), start_(start), end_(end), step_(step) {}

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

enum Operator {
  // cxx operators
  Index,
  Assign,
  For,
  While,
  If,
  // parallel operators
  ParallelLoad,
  AtomicOp,
  MemFence,
  ParallelFor,
  For_uni,
  While_uni,
  If_uni,
  Any,
  Ballot,
  Enlist,
  Singleton,
  Barrier,
  InclusiveScan,
  ExclusiveScan,
  Reduce,
  RescourceAllocation,
};

class Operation: public Value {
 public:
  Operation(Scope scope, Operator op, DataType type = NilTy):
      Value(type, scope), op_(op) {}

  Operator op() const { return op_; }
  virtual string nice_str() { return "UnknowOp " + std::to_string(op_); }

 protected:
  Operator op_;
};

class IndexOp: public Operation {
 public:
  IndexOp(PointerValue *base, IntValue *offset):
      Operation(Thread, Index, base->dtype()), base_(base), offset_(offset) {}

  PointerValue *base() const { return base_; }
  IntValue *offset() const { return offset_; }
  string nice_str() {
    std::string comment = "/* ";
    comment += ScopeDesc(scope_) + " */ ";
    return comment + base_->name() + "[" + offset_->name() + "]";
  }

 private:
  PointerValue *base_;
  IntValue *offset_;
};

class ParforOp: public Operation {
 public:
  ParforOp(Scope scope, Value *it, IteratableValue *container):
      Operation(scope, ParallelFor), iterator_(it), container_(container) {}

  void appendOp(Operation *op) {
    body_.push_back(op);
  }

 private:
   Value *iterator_;
   IteratableValue *container_;
   vector<Operation *> body_;
};

class DynArray: public IteratableValue {
 public:
  DynArray(DataType type, Scope scope, string name, 
           PointerValue *data, IntValue *length):
      IteratableValue(type, scope, name), data_(data), length_(length) {}

  DynArray(IteratableType *type, Scope scope, string name):
      IteratableValue(type, scope, name),
      data_(PointerValue::CreatePtrValue(scope, name + ".data", type->dtype())),
      length_(IntValue::CreateIntValue(scope, name + ".length")) {}

  static IteratableType *GetDynArrayTy(DataType dtype);

  virtual IntValue *start(void) const { return ConstantInt::Zero; }
  virtual IntValue *end(void) const { return length_; }
  virtual IntValue *step(void) const { return ConstantInt::One; }

  Value *data() const { return data_; }
  Value *length() const { return length_; }

  virtual Value *reference(IntValue *iter) {
    return new IndexOp(data_, iter);
  }

  DataType dtype() const {
    IteratableType *ty = dynamic_cast<IteratableType *> (type_);
    return ty->dtype();
  }

 private:
  PointerValue *data_;
  IntValue *length_;
};

void InitGlang(void);

}  // namespace glang

#endif  // _SYNTHESIS_GLANG_H
