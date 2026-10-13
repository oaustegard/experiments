#include "FuncState.h"
#include "ConsControl.h"
#include "fn_luaK_exp2nextreg.h"
#include "fn_luaK_setlist.h"

static void closelistfield (FuncState *fs, ConsControl *cc) {
  if (cc->v.k == VVOID)
    return;
  luaK_exp2nextreg(fs, &cc->v);
  cc->v.k = VVOID;
  if (cc->tostore == 50) {
    luaK_setlist(fs, cc->t->u.info, cc->na, cc->tostore);
    cc->na += cc->tostore;
    cc->tostore = 0;
  }
}
