#include "glang.h"

namespace glang {

DataType NilTy = new Type(Nil, "nil");

static map<int, IntType *> int_types;

IntType *IntType::GetIntegerTy(int bw) {
  auto res = int_types[bw];
  if (res == nullptr) {
    int_types[bw] = new IntType(bw);
  }

  return int_types[bw];
}

static map<DataType, PointerType *> ptr_types;

PointerType *PointerType::GetPointerTy(DataType dtype) {
  auto res = ptr_types[dtype];
  if (res == nullptr) {
    ptr_types[dtype] = new PointerType(dtype);
  }

  return ptr_types[dtype];
}

string ScopeDesc(Scope s) {
  switch(s) {
    case Thread: return "__thread__";
    case Warp: return "__warp__";
    case ThreadBlock: return "__threadblock__";
    case Device: return "__device__";
    case Constant: return "__constant__";
    default: return "__error_scope__";
  }

  return "__fatal_error__";
}

static int64_t values = 0;
int64_t value_id_gen(void) {
  return values++;
}

IntValue *IntValue::global_id,
         *IntValue::threadblock_id, *IntValue::thread_id,
         *IntValue::warp_id, *IntValue::lane_id;
IntValue *IntValue::WARP_SIZE, *IntValue::TB_SIZE, *IntValue::GLOBAL_SIZE;

IntValue *IntValue::CreateIntValue(Scope s, string name, int bw) {
  auto p = new IntValue(s, name, bw);
  return p;
}

static map<int, map<int64_t, ConstantInt *> > const_int_values;

ConstantInt *ConstantInt::CreateConstInt(int64_t v, int bw) {
  auto res = const_int_values[bw][v];
  if (res == nullptr) {
    // store p into a global map to reduce memory
    const_int_values[bw][v] = new ConstantInt(v, bw);
  }
  return const_int_values[bw][v];
}

PointerValue *
PointerValue::CreatePtrValue(Scope s, string name, DataType dtype) {
  auto p = new PointerValue(s, name, dtype);
  return p;
}

IteratableType *SliceTy =
    new IteratableType("slice", IntType::GetIntegerTy());
 
Slice *CreateConstSlice(int start, int end, int step = 1) {
  auto c_start = ConstantInt::CreateConstInt(start);
  auto c_end = ConstantInt::CreateConstInt(end);
  auto c_step = ConstantInt::CreateConstInt(step);

  string name = "slice";
  name += "[" + c_start->name() + ":"
          + c_end->name() + ":" + c_step->name() + "]";
  auto slice = new Slice(c_start, c_end, c_step, Constant, name);
  CHECK(slice != nullptr);
  return slice;
}

static map<DataType, IteratableType *> darray_types;

IteratableType *DynArray::GetDynArrayTy(DataType dtype) {
  auto res = darray_types[dtype];
  if (res == nullptr) {
    darray_types[dtype] = new IteratableType("darray", dtype);
  }

  return darray_types[dtype];
}

FunctionValue *
FunctionValue::declareFunction(string name, DataType ret_type) {
  auto ty = new CallableType(name, ret_type);
  return new FunctionValue(ty);
}

ConstantInt *ConstantInt::Zero, *ConstantInt::One;

void InitGlang(void) {
  ConstantInt::Zero = ConstantInt::CreateConstInt(0);
  ConstantInt::One = ConstantInt::CreateConstInt(1);
  IntValue::global_id = IntValue::CreateIntValue(Thread, GLOBAL_ID_STR);
  IntValue::threadblock_id = IntValue::CreateIntValue(Thread, TB_ID_STR);
  IntValue::thread_id = IntValue::CreateIntValue(Thread, THREAD_ID_STR);
  IntValue::warp_id = IntValue::CreateIntValue(Thread, WARP_ID_STR);
  IntValue::lane_id = IntValue::CreateIntValue(Thread, LANE_ID_STR);
  IntValue::WARP_SIZE = IntValue::CreateIntValue(Constant, WARP_SIZE_STR);
  IntValue::TB_SIZE = IntValue::CreateIntValue(Constant, TB_SIZE_STR);
  IntValue::GLOBAL_SIZE = IntValue::CreateIntValue(Constant, GLOBAL_SIZE_STR);
}

}  // namespace glang

using namespace glang;

void log_value(Value *v) {
  LOG(INFO) << v->type()->nice_str() << " " << v->nice_str();
}

void log_op(Operation *op) {
  LOG(INFO) << op->cxx_code();
}

int main(int argc, char **argv) {
  // google::InitGoogleLogging(argv[0]);
  InitGlang();

  auto C0 = ConstantInt::CreateConstInt(0);
  CHECK_EQ(C0, ConstantInt::Zero);
  log_value(C0);
  auto tid = new IntValue(Thread, THREAD_ID_STR, 32, true);
  log_value(tid);
  auto S0 = CreateConstSlice(1, 129, 4);
  log_value(S0);
  auto A0 = new DynArray( DynArray::GetDynArrayTy(IntType::GetIntegerTy()),
                          Device, "Vertices" );
  log_value(A0);
  log_value(A0->data());
  log_value(A0->length());
  log_value(A0->reference(C0));

  auto I = new IntValue(Thread, "i");
  auto decl_i = new DeclareOp(I, A0->reference(I), ThreadBlock);
  log_op(decl_i);

  auto V = new IntValue(Thread, "v");
  auto v_from_ai = new AssignOp(V, A0->reference(I));
  log_op(v_from_ai);

  auto pf0 = new ParforOp(ThreadBlock, V, A0);
  auto visit = FunctionValue::declareFunction("visit");
  log_value(visit);
  std::vector<Value *> args; args.push_back(V);
  auto call = new CallOp(visit, args);
  log_op(call);
  pf0->appendOp(call);
  log_op(pf0);

  auto J = new IntValue(Thread, "j");
  auto pf1 = new ParforOp(Warp, J, S0);
  log_op(pf1);

  auto B0 = new BroadcastOp(ThreadBlock, V, I, J);
  log_op(B0);

  return 0;
}
