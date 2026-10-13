#include "FuncState.h"
#include "fn_luaK_code.h"
#include "fn_luaK_codeABx.h"
#include "lopcodes.h"

static int luaK_codek (FuncState *fs, int reg, int k) {
  if (k <= MAXARG_Bx)
    return luaK_codeABx(fs, OP_LOADK, reg, k);
  else {
    luaK_codeABx(fs, OP_LOADKX, reg, 0);
    return luaK_code(fs, CREATE_Ax(OP_EXTRAARG, k));
  }
}
