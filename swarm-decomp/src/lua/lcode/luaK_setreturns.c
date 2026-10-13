#include "FuncState.h"
#include "expdesc.h"
#include "fn_luaK_reserveregs.h"
#include "CREATE_ABCk.h"

void luaK_setreturns (FuncState *fs, expdesc *e, int nresults) {
  Instruction *pc = &fs->f->code[e->u.info];
  Instruction ins = (*pc & 0xffffff) | ((Instruction)(nresults + 1) << 24);
  if (e->k != VCALL) {
    ins = (ins & 0xffff807f) | ((Instruction)fs->freereg << 7);
    *pc = ins;
    luaK_reserveregs(fs, 1);
  }
  else
    *pc = ins;
}
