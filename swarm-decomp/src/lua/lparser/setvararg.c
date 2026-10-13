#include "FuncState.h"
#include "fn_luaK_codeABCk.h"

static void setvararg (FuncState *fs, int nparams) {
  fs->f->is_vararg = 1;
  luaK_codeABCk(fs, 0x51, nparams, 0, 0, 0);
}
