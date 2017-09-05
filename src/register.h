#ifndef _REGISTER_H
#define _REGISTER_H

#include <stdint.h>

namespace simdsim {

class VirtualRegister {
 public:
  bool IsScalar() const;
  bool IsPositional() const;

 private:
  type_;  // register type
  int bit_width_;  // bit width
};

class Register32Bit: public VirtualRegister {
 private:
  int32_t data_;
};

}  // namespace simdsim

#endif  // _REGISTER_H
