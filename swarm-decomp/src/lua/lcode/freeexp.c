#include "FuncState.h"
#include "expdesc.h"
#include "fn_luaY_nvarstack.h"

static void freeexp (FuncState *fs, expdesc *e) {
  if (e->k == VNONRELOC) {
    int r = e->u.info;
    if (r >= luaY_nvarstack(fs)) fs->freereg--;
  }
}
