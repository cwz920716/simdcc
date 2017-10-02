extern "C" {

struct InstParams {
  int32_t id;
  int32_t type;
  bool will_execute;
  int32_t line_no;
};  // struct InstParams

struct CondBranchParams {
  int32_t id;
  bool taken;
  bool is_conditional;
};  // struct CondBranchParams

struct MemParams {
  int64_t address;
  int64_t size_in_bits;
  int32_t addr_space;
  bool write;
};  // struct MemParams

}  // extern "C"
