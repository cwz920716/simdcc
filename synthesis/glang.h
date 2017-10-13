#ifndef _SYNTHESIS_GLANG_H
#define _SYNTHESIS_GLANG_H

// GPU Primitive Language (glang) for BFS Graph Traversal

#include <string>
#include <sstream>
#include <iostream>
#include <cinttypes>
#include <map>
#include <vector>

#include <glog/logging.h>

#define GLOBAL_ID_STR       std::string("global_id")
#define TB_ID_STR           std::string("threadblock_id")
#define THREAD_ID_STR       std::string("thread_id")
#define WARP_ID_STR         std::string("warp_id")
#define LANE_ID_STR         std::string("lane_id")

#define GLOBAL_SIZE_STR     std::string("GLOBAL_SIZE")
#define TB_SIZE_STR         std::string("THREADBLOCK_SIZE")
#define WARP_SIZE_STR       std::string("WARP_SIZE")

#define SHARED "__shared__"
#define DEVICE "__device__"

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
  Callable,
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
    if (typecode_ == 0) {
      return "nil";
    }

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

class CallableType: public Type {
 public:
  CallableType(string name, DataType ret_type = NilTy):
      Type(Callable), name_(name), ret_type_(ret_type) {}

  string nice_str() { return "Callable"; }
  string name() const { return name_; }
  DataType ret_type() const { return ret_type_; }

 private:
  string name_;
  DataType ret_type_;
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
  Value(int64_t id, DataType type, Scope scope, const string &name):
      id_(id), type_(type), scope_(scope), name_(name) {}
  Value(DataType type, Scope scope, const string &name):
      id_(-1), type_(type), scope_(scope), name_(name) { id_ = value_id_gen(); }
  Value(DataType type, Scope scope):
      id_(-1), type_(type), scope_(scope) { id_ = value_id_gen(); }

  int id() const { return id_; }
  string name() const { return name_; }
  DataType type() const { return type_; }
  Scope scope() const { return scope_; }

  virtual string comment_str() {
    std::string comment = "/* ";
    comment += ScopeDesc(scope_) + " */ ";
    return comment;
  }

  virtual string nice_str() {
    auto ref_name = name_;
    // TODO(wcui): Warp scope do not work with vector/array types.
    if (scope_ == Warp) {
      ref_name += "[" + WARP_ID_STR + "]";
    }
    return ref_name;
  }

 protected:
  int64_t id_;
  DataType type_;
  Scope scope_;
  string name_;
};

// non-const int OR symbolic int
class IntValue: public Value {
 public:
  IntValue(Scope scope, string name, int bitwidth = 32,
           bool is_constant = false):
      Value(IntType::GetIntegerTy(bitwidth), scope, name),
      is_constant_(is_constant) {}

  static IntValue *CreateIntValue(Scope s, string name, int bw = 32);

  static IntValue *UniqueId(Scope s) {
    switch (s) {
      case Device: return global_id;
      case ThreadBlock: return thread_id;
      case Warp: return lane_id;
      default:
        LOG(FATAL) << "Do not support scope " << ScopeDesc(s);
        return nullptr;
    }
  }

  bool isConstant() const { return is_constant_; }

  virtual bool isStatic() const { return false; }

  virtual int64_t value() const {
    CHECK(false) << "Non-const IntValue cannot return value!";
    return 0;
  }

  virtual bool Equals(int64_t v) const {
    if (isStatic() && isConstant()) {
      return value() == v;
    }

    return false;
  }

  static IntValue *global_id, *threadblock_id, *thread_id, *warp_id, *lane_id;
  static IntValue *WARP_SIZE, *TB_SIZE, *GLOBAL_SIZE;

 protected:
  bool is_constant_;
};

class ConstantInt: public IntValue {
 public:
  ConstantInt(int64_t value, int bitwidth = 32):
      IntValue(Constant, to_string(value), bitwidth, true), value_(value) {}

  static ConstantInt *CreateConstInt(int64_t v, int bw = 32);

  virtual bool isStatic() const { return true; }
  virtual int64_t value() const { return value_; }

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

// Parallel iteratable
class IteratableValue: public Value {
 public:
  IteratableValue(DataType type, Scope scope, const string &name):
      Value(type, scope, name) {}

  virtual IntValue *start(void) const = 0;
  virtual IntValue *end(void) const = 0;
  virtual IntValue *step(void) const = 0;

  virtual bool need_reference() const { return true; }
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

  bool need_reference() const { return false; }
  virtual Value *reference(IntValue *iter) {
    CHECK(iter);
    return iter;
  }

 private:
  IntValue *start_, *end_, *step_; 
};

class FunctionValue: public Value {
 public:
  FunctionValue(CallableType *func_type):
      Value(func_type, Constant, func_type->name()), func_type_(func_type) {}

  DataType ret_type() const { return func_type_->ret_type(); }

  static FunctionValue *declareFunction(string name, DataType ret_type = NilTy);

 private:
  CallableType *func_type_;
};

enum Operator {
  // cxx operators
  Declare,
  Index,
  Assign,
  Binary,
  Call,
  For,
  While,
  If,
  BasicBlock,
  // parallel operators
  ParallelLoad,
  AtomicOp,
  MemFence,
  ParallelFor,
  For_uni,
  While_uni,
  If_uni,
  Any,
  Broadcast,
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
  virtual string nice_str() { return "Op" + std::to_string(op_); }
  virtual string cxx_code() { return nice_str(); }

 protected:
  Operator op_;
};

class DeclareOp: public Operation {
 public:
  DeclareOp(Value *var, Value *init = nullptr, Scope scope = Thread):
      Operation(scope, Declare), var_(var), init_(init) {}

  Value *var() const { return var_; }
  Value *init() const { return init_; }

  string nice_str() {
    auto decl = var_->type()->nice_str() + " " + var_->nice_str();
    if (init_) {
      decl += " = " + init_->nice_str();
    }
    return decl;
  }

  string cxx_code() {
    if (scope_ == Thread) return nice_str();

    if (scope_ == ThreadBlock) {
      auto decl = string(SHARED) + " " +
                  var_->type()->nice_str() + " " + var_->name();
      if (init_) {
        decl += " = " + init_->nice_str();
      }
      return decl;
    }

    // TODO(wcui): Fix declare for vector/array types
    if (scope_ == Warp) {
      CHECK(init_ == nullptr) << "not support __warp__ initializer yet.";
      auto decl = string(SHARED) + " " +
                  var_->type()->nice_str() + " " + var_->name() +
                  "[" + WARP_SIZE_STR + "]";
      return decl;
    }

    if (scope_ == Device) {
      auto decl = string(DEVICE) + " " +
                  var_->type()->nice_str() + " " + var_->name();
      if (init_) {
        decl += " = " + init_->nice_str();
      }
      return decl;
    }
  }

 private:
  Value *var_;
  Value *init_;
};

class IndexOp: public Operation {
 public:
  IndexOp(PointerValue *base, IntValue *offset):
      Operation(Thread, Index, base->dtype()), base_(base), offset_(offset) {}

  PointerValue *base() const { return base_; }
  IntValue *offset() const { return offset_; }
  string nice_str() {
    return base_->nice_str() + "[" + offset_->nice_str() + "]";
  }

 private:
  PointerValue *base_;
  IntValue *offset_;
};

class AssignOp: public Operation {
 public:
  AssignOp(Value *lhs, Value *rhs):
      Operation(Thread, Assign, lhs->type()), lhs_(lhs), rhs_(rhs) {}

  string nice_str() {
    return lhs_->name() + " = " + rhs_->nice_str();
  }

 private:
  Value *lhs_, *rhs_;
};

#define BIN_OP_LT "<"
#define BIN_OP_EQ "=="
#define BIN_OP_ADD "+"
#define BIN_OP_MUL "*"
#define BIN_OP_INC "+="

class BinaryOp: public Operation {
 public:
  BinaryOp(Value *lhs, Value *rhs, string bin_op, DataType type = nullptr):
      Operation(Thread, Binary, (type != nullptr) ? type : lhs->type()),
      lhs_(lhs), rhs_(rhs), bin_op_(bin_op) {}

  string nice_str() {
    return lhs_->nice_str() + " " + bin_op_ + " " + rhs_->nice_str();
  }

 private:
  Value *lhs_, *rhs_;
  string bin_op_;
};

class CallOp: public Operation {
 public:
  CallOp(FunctionValue *function, std::vector<Value *> &args):
      Operation(Thread, Call, function->ret_type()),
      function_(function), args_(args) {}

  FunctionValue *function() const { return function_; }

  string nice_str() {
    string res = function_->nice_str() + "(";
    for (int i = 0; i < args_.size(); i++) {
      if (i > 0) {
        res += ", ";
      }
      res += args_[i]->nice_str();
    }
    res += ")";
    return res;
  }

 private:
  FunctionValue *function_;
  std::vector<Value *> args_;
};

class ParforOp: public Operation {
 public:
  ParforOp(Scope scope, Value *it, IteratableValue *container):
      Operation(scope, ParallelFor), iterator_(it), container_(container) {}

  void appendOp(Operation *op) {
    body_.push_back(op);
  }

  vector<Operation *> &body() {
    return body_;
  }

  Value *start() {
    IntValue *start_id = nullptr;
    switch(scope_) {
      case Warp: start_id = IntValue::lane_id; break;
      case ThreadBlock: start_id = IntValue::thread_id; break;
      case Device: start_id = IntValue::global_id; break;
      default: break;
    }
    CHECK(start_id);

    auto stride = container_->step();
    Value *start_stride = start_id;
    if (!stride->Equals(1)) {
      start_stride = new BinaryOp(start_id, stride, BIN_OP_MUL);
    }

    auto offset = container_->start();
    if (offset->Equals(0)) {
      return start_stride; 
    } else {
      return new BinaryOp(start_stride, offset, BIN_OP_ADD);
    }
  }

  Value *step() {
    IntValue *par_limit = nullptr;
    switch(scope_) {
      case Warp: par_limit = IntValue::WARP_SIZE; break;
      case ThreadBlock: par_limit = IntValue::TB_SIZE; break;
      case Device: par_limit = IntValue::GLOBAL_SIZE; break;
      default: break;
    }
    CHECK(par_limit);
    auto stride = container_->step();
    if (stride->Equals(1)) {
      return par_limit; 
    } else {
      return new BinaryOp(par_limit, stride, BIN_OP_MUL);
    }
  }

  string nice_str() {
    IntValue *i = new IntValue(Thread, i_name());
    Value *ref = container_->reference(i);

    DeclareOp *decl_it = nullptr;
    if (container_->need_reference()) {
      decl_it = new DeclareOp(iterator_, ref);
    }

    auto decl_i = new DeclareOp(i, start());
    auto end_cond = new BinaryOp(i, container_->end(), BIN_OP_LT);
    auto incr_i = new BinaryOp(i, step(), BIN_OP_INC);

    std::string res = comment_str() + "\nfor(" + decl_i->nice_str() +
                      "; " + end_cond->nice_str() + "; " + incr_i->nice_str() +
                      ") { " + (decl_it? (decl_it->nice_str() + "; ") : "");
    for (auto stmt : body_) {
      res += stmt->nice_str(); + "; ";
    }
    res += "}";
    return res;
  }

 private:
  string i_name() {
    string tmp = iterator_->name();
    if (container_->need_reference()) {
      tmp += "_i";
    }
    return std::move(tmp);
  }

   Value *iterator_;
   IteratableValue *container_;
   vector<Operation *> body_;
};

class WhileOp: public Operation {
 public:
  WhileOp(Value *cond):
      Operation(Thread, While), cond_(cond) {}

  void appendOp(Operation *op) {
    body_.push_back(op);
  }

  vector<Operation *> &body() {
    return body_;
  }

  string nice_str() {
    std::string res = comment_str();
    res += "while (" +cond_->nice_str() + ")";
    res += " { ";
    for (auto stmt : body_) {
      res += stmt->nice_str(); + "; ";
    }
    res += "}";
    return res;
  }

 private:
  Value *cond_;
  vector<Operation *> body_;
};

class IfOp: public Operation {
 public:
  IfOp(Value *cond):
      Operation(Thread, If), cond_(cond) {}

  void appendThenOp(Operation *op) {
    then_body_.push_back(op);
  }

  void appendElseOp(Operation *op) {
    else_body_.push_back(op);
  }

  vector<Operation *> &then_body() {
    return then_body_;
  }

  vector<Operation *> &else_body() {
    return else_body_;
  }

  bool hasElse() {
    return !else_body_.empty();
  }

  string nice_str() {
    std::string res = "";
    res += "if (" +cond_->nice_str() + ")";
    res += " { ";
    for (auto stmt : then_body_) {
      res += stmt->nice_str() + "; ";
    }
    res += "}";
    if (hasElse()) {
      res += "else {";
      for (auto stmt : else_body_) {
        res += stmt->nice_str() + "; ";
      }
      res += "}";
    }
    return res;
  }

 private:
  Value *cond_;
  vector<Operation *> then_body_;
  vector<Operation *> else_body_;
};

class BarrierOp: public Operation {
 public:
  BarrierOp(Scope scope): Operation(scope, Barrier) {}

  string nice_str() {
    if (scope_ == Warp) {
      return "/* __warp_sync(); */";
    }

    if (scope_ == ThreadBlock) {
      return "__synthreads()";
    }

    if (scope_ == Device) {
      return "__gpu_sync()";
    }

    LOG(FATAL) << "Barrier do not support scope " << ScopeDesc(scope_);
    return "";
  }
};

class AnyOp: public Operation {
 public:
  AnyOp(Scope scope, Value *cond):
      Operation(scope, Any, IntType::GetIntegerTy()), cond_(cond) {}

  string nice_str() {
    if (scope_ == Warp) {
      return "__ballot(" + cond_->nice_str() + ")";
    }

    if (scope_ == ThreadBlock) {
      return "__threadblock_any(" + cond_->nice_str() + ")";
    }

    if (scope_ == Device) {
      return "__gpu_any(" + cond_->nice_str() + ")";
    }

    LOG(FATAL) << "Any do not support scope " << ScopeDesc(scope_);
    return "";
  }

 private:
  Value *cond_;
};

class BroadcastOp: public Operation {
 public:
  BroadcastOp(Scope scope, IntValue *id, Value *obj, Value *target):
    Operation(scope, Broadcast, obj->type()),
    id_(id), obj_(obj), target_(target) {}

  string nice_str() {
    auto unique_id = IntValue::UniqueId(scope_);
    auto cmp_id = new BinaryOp(unique_id, id_, BIN_OP_EQ);
    auto if_ = new IfOp(cmp_id);
    if_->appendThenOp(new AssignOp(target_, obj_));
    auto sync = new BarrierOp(scope_);
    return if_->nice_str() + " " + sync->nice_str();
  }

 private:
  IntValue *id_;
  Value *obj_, *target_;
};

class EnlistOp: public Operation {
 public:
  EnlistOp(Scope scope, Value *cond, Value *broadcast):
    Operation(scope, Enlist), cond_(cond), broadcast_(broadcast) {}

  void appendOp(Operation *op) {
    body_.push_back(op);
  }

  vector<Operation *> &body() {
    return body_;
  }

 private:
  Value *cond_;
  Value *broadcast_;
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
