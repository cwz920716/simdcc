#include "memory.h"
#include "glog/logging.h"

int main(void) {
  char data[1024];
  simdsim::BasicMemory mem(1024);
  mem.Mmap(data);

  mem.WriteByte(111, 111);
  CHECK(mem.ReadByte(111) == 111);

  return 0;
}
