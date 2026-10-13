#include "fn_luaK_dischargevars.h"
#include "fn_luaK_reserveregs.h"
#include "exp2reg.h"
#include "freeexp.h"

void luaK_exp2nextreg (FuncState *fs, expdesc *e) {
  luaK_dischargevars(fs, e);
  freeexp(fs, e);
  luaK_reserveregs(fs, 1);
  exp2reg(fs, e, fs->freereg - 1);
}
