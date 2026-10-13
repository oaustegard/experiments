#include "FuncState.h"
#include "fn_luaY_nvarstack.h"
#include "freereg.h"

static void freereg(FuncState *fs, int reg) {
  if (reg >= luaY_nvarstack(fs))
    fs->freereg--;
}
