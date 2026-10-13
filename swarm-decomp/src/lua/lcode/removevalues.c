#include "FuncState.h"
#include "CREATE_ABCk.h"
#include "luaP_opmodes.h"
#include "getjump.h"

static void removevalues (FuncState *fs, int list) {
  for (; list != -1; list = getjump(fs, list)) {
    Instruction *pi = &fs->f->code[list];
    Instruction *i = pi;
    Instruction ins;
    if (list >= 1 && (luaP_opmodes[(*(pi - 1)) & 0x7f] & (1 << 4)))
      i = pi - 1;
    ins = *i;
    if ((ins & 0x7f) == 67) {
      *i = (Instruction)66 | (((ins >> 16) & 0xff) << 7) | (((ins >> 15) & 1) << 15);
    }
  }
}
