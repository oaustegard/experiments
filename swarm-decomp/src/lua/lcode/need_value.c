#include "luaP_opmodes.h"
#include "getjump.h"

static int need_value (FuncState *fs, int list) {
  for (; list != -1; list = getjump(fs, list)) {
    Instruction *pc = &fs->f->code[list];
    Instruction *pm = pc - 1;
    if (list < 1 || !(luaP_opmodes[*pm & 0x7f] & 0x10))
      pm = pc;
    pc = pm;
    if ((*pc & 0x7f) != 0x43)
      return 1;
  }
  return 0;
}
