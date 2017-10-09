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
    case ThreadBlock: return "__threadvlock__";
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

ConstantInt *CreateConstInt(int64_t v, int bw = 32) {
  auto p = new ConstantInt(v, bw);
  // store p into a global map to reduce memory
  return p;
}

IntValue *IntValue::CreateIntValue(Scope s, string name, int bw) {
  auto p = new IntValue(s, name, bw);
  return p;
}

PointerValue *
PointerValue::CreatePtrValue(Scope s, string name, DataType dtype) {
  auto p = new PointerValue(s, name, dtype);
  return p;
}

IteratableType *SliceTy =
    new IteratableType("slice", IntType::GetIntegerTy());
 
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

static map<DataType, IteratableType *> darray_types;

IteratableType *DynArray::GetDynArrayTy(DataType dtype) {
  auto res = darray_types[dtype];
  if (res == nullptr) {
    darray_types[dtype] = new IteratableType("darray", dtype);
  }

  return darray_types[dtype];
}

ConstantInt *ConstantInt::Zero, *ConstantInt::One;

void InitGlang(void) {
  ConstantInt::Zero = CreateConstInt(0);
  ConstantInt::One = CreateConstInt(1);
}

}  // namespace glang

using namespace glang;

void log_value(Value *v) {
  LOG(INFO) << v->type()->nice_str() << " " << v->nice_str();
}

int main(int argc, char **argv) {
  // google::InitGoogleLogging(argv[0]);
  InitGlang();

  auto c0 = CreateConstInt(0);
  log_value(c0);
  auto tid = new IntValue(Thread, THREAD_IDX, 32, true);
  log_value(tid);
  auto s0 = CreateConstSlice(0, 128);
  log_value(s0);
  auto A0 = new DynArray( DynArray::GetDynArrayTy(IntType::GetIntegerTy()),
                          Device, "Vertices" );
  log_value(A0);
  log_value(A0->data());
  log_value(A0->length());
  log_value(A0->reference(c0));

  return 0;
}
