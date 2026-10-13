#include "fn_luaY_nvarstack.h"
#include "fn_luaK_exp2nextreg.h"
#include "fn_luaK_dischargevars.h"
#include "exp2reg.h"

int luaK_exp2anyreg (FuncState *fs, expdesc *e) {
  luaK_dischargevars(fs, e);
  if (e->k == VNONRELOC) {
    if (e->t == e->f) return e->u.info;
    if (e->u.info >= luaY_nvarstack(fs)) {
      exp2reg(fs, e, e->u.info);
      return e->u.info;
    }
  }
  luaK_exp2nextreg(fs, e);
  return e->u.info;
}
