#include "FuncState.h"
#include "expdesc.h"
#include "fn_luaK_exp2anyreg.h"
#include "fn_luaK_indexed.h"
#include "isKstr.h"
#include "isCint.h"
#include "str2K.h"

void luaK_indexed (FuncState *fs, expdesc *t, expdesc *k) {
  if (k->k == VKSTR)
    str2K(fs, k);
  if (t->k == VUPVAL && !isKstr(fs, k))
    luaK_exp2anyreg(fs, t);
  if (t->k == VUPVAL) {
    t->u.ind.t = t->u.info;
    t->u.ind.idx = k->u.info;
    t->k = VINDEXUP;
  }
  else {
    t->u.ind.t = (t->k == VLOCAL) ? t->u.var.ridx : t->u.info;
    if (isKstr(fs, k)) {
      t->u.ind.idx = k->u.info;
      t->k = VINDEXSTR;
    }
    else if (isCint(k)) {
      t->u.ind.idx = (short)k->u.ival;
      t->k = VINDEXI;
    }
    else {
      t->u.ind.idx = luaK_exp2anyreg(fs, k);
      t->k = VINDEXED;
    }
  }
}
