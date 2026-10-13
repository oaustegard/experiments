#include "LexState.h"
#include "FuncState.h"
#include "expdesc.h"
#include "fn_luaK_storevar.h"
#include "fn_luaK_fixline.h"
#include "fn_luaX_next.h"
#include "singlevar.h"
#include "fieldsel.h"
#include "body.h"
#include "check_readonly.h"

static int funcname (LexState *ls, expdesc *v) {
  int ismethod = 0;
  singlevar(ls, v);
  while (ls->t.token == '.')
    fieldsel(ls, v);
  if (ls->t.token == ':') {
    ismethod = 1;
    fieldsel(ls, v);
  }
  return ismethod;
}

static void funcstat (LexState *ls, int line) {
  int ismethod;
  expdesc v, b;
  luaX_next(ls);
  ismethod = funcname(ls, &v);
  body(ls, &b, ismethod, line);
  check_readonly(ls, &v);
  luaK_storevar(ls->fs, &v, &b);
  luaK_fixline(ls->fs, line);
}
