#include "fn_luaK_exp2anyreg.h"
#include "fn_luaK_code.h"


static int exp2RK (FuncState *fs, expdesc *e) {
  if (luaK_exp2K(fs, e)) return 1;
  else {
    luaK_exp2anyreg(fs, e);
    return 0;
  }
}

static void codeABRK (FuncState *fs, int o, int a, int b, expdesc *ec) {
  int k = exp2RK(fs, ec);
  luaK_code(fs, CREATE_ABCk(o, a, b, ec->u.info, k));
}
