#include "FuncState.h"
#include "ConsControl.h"
#include "fn_luaK_exp2nextreg.h"
#include "fn_luaK_setlist.h"
#include "fn_luaK_setreturns.h"
#include "expkind.h"

static void lastlistfield (FuncState *fs, ConsControl *cc) {
  if (cc->tostore == 0)
    return;
  if (cc->v.k != VCALL && cc->v.k != VVARARG) {
    if (cc->v.k != VVOID)
      luaK_exp2nextreg(fs, &cc->v);
    luaK_setlist(fs, cc->t->u.info, cc->na, cc->tostore);
  }
  else {
    luaK_setreturns(fs, &cc->v, -1);
    luaK_setlist(fs, cc->t->u.info, cc->na, -1);
    cc->na--;
  }
  cc->na += cc->tostore;
}
