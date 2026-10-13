#include "FuncState.h"
#include "fn_luaX_next.h"
#include "reglevel.h"

int luaY_nvarstack (FuncState *fs) {
  return reglevel(fs, fs->nactvar);
}
