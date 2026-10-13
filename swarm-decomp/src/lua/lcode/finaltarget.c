#include "Proto.h"
#include "OpCode.h"

static int finaltarget (Instruction *code, int i) {
  int count;
  for (count = 0; count < 100; count++) {
    Instruction pc = code[i];
    if ((pc & 0x7f) != 56)
      break;
    else
      i += (int)(pc >> 7) - 0xffffff + 1;
  }
  return i;
}
