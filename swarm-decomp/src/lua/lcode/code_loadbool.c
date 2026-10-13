#include "fn_luaK_code.h"

static int code_loadbool (FuncState *fs, int A, int op) {
  fs->lasttarget = fs->pc;
  return luaK_code(fs, (op) | (A << 7));
}
