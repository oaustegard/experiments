#include "FuncState.h"
#include "fn_luaK_exp2anyregup.h"
#include "fn_luaK_indexed.h"
#include "fn_luaX_next.h"
#include "fn_luaX_syntaxerror.h"

static void error_expected (LexState *ls, int token);

static void fieldsel (LexState *ls, expdesc *v) {
  FuncState *fs = ls->fs;
  expdesc key;
  luaK_exp2anyregup(fs, v);
  luaX_next(ls);
  if (ls->t.token != 0x123)
    error_expected(ls, 0x123);
  TString *s = ls->t.seminfo.ts;
  luaX_next(ls);
  key.u.strval = s;
  key.f = key.t = -1;
  key.k = VKSTR;
  luaK_indexed(fs, v, &key);
}
