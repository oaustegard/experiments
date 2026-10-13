#include "FuncState.h"
#include "expdesc.h"
#include "fn_luaK_exp2anyreg.h"
#include "fn_luaK_reserveregs.h"
#include "freeexp.h"
#include "codeABRK.h"

void luaK_self (FuncState *fs, expdesc *e, expdesc *key) {
  int ereg;
  luaK_exp2anyreg(fs, e);
  ereg = e->u.info;
  freeexp(fs, e);
  e->u.info = fs->freereg;
  e->k = VNONRELOC;
  luaK_reserveregs(fs, 2);
  codeABRK(fs, 0x14, e->u.info, ereg, key);
  freeexp(fs, key);
}
