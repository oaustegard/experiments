#include "FuncState.h"
#include "CREATE_ABCk.h"
#include "lopcodes.h"
#include "fn_luaO_ceillog2.h"

void luaK_settablesize (FuncState *fs, int pc, int ra, int asize, int hsize) {
  Instruction *inst = &fs->f->code[pc];
  int rb = (hsize != 0) ? luaO_ceillog2(hsize) + 1 : 0;
  int extra = asize / 256;
  int k = (extra > 0);
  *inst = CREATE_ABCk(19, ra, rb, asize, k);
  *(inst + 1) = CREATE_Ax(OP_EXTRAARG, extra);
}
