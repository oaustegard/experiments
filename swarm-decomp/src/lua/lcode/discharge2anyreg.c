#include "FuncState.h"
#include "expdesc.h"
#include "fn_luaK_reserveregs.h"
#include "discharge2reg.h"

static void discharge2anyreg (FuncState *fs, expdesc *e) {
  if (e->k != VNONRELOC) {
    luaK_reserveregs(fs, 1);
    discharge2reg(fs, e, fs->freereg - 1);
  }
}
