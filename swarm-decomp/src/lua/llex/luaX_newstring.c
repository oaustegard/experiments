#include "llex_macros.h"
#include "lstate.h"
#include "fn_luaS_newlstr.h"
#include "fn_luaH_getstr.h"
#include "fn_luaH_finishset.h"
#include "fn_luaC_step.h"
#include "fn_luaX_newstring.h"

TString *luaX_newstring(LexState *ls, const char *str, size_t l) {
  lua_State *L = ls->L;
  TString *ts = luaS_newlstr(L, str, l);
  const TValue *o = luaH_getstr(ls->h, ts);
  if ((o->tt_ & 0xf) != 0)
    ts = (TString *)((Node *)o)->u.key_val.gc;
  else {
    TValue *stv = L->top++;
    stv->value_.gc = (GCObject *)ts;
    stv->tt_ = ts->tt | 0x40;
    luaH_finishset(L, ls->h, stv, o, stv);
    if (L->l_G->GCdebt > 0)
      luaC_step(L);
    L->top--;
  }
  return ts;
}
