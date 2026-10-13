#include "lstate.h"
#include "fn_luaS_newlstr.h"
#include "fn_luaV_concat.h"
#include "fn_luaC_step.h"
#include "lua.h"

void lua_concat (lua_State *L, int n) {
  if (n > 0)
    luaV_concat(L, n);
  else {
    StkId top = L->top;
    struct TString *ts = luaS_newlstr(L, "", 0);
    top->value_.gc = (GCObject *)ts;
    top->tt_ = ((GCObject *)ts)->tt | 0x40;
    L->top++;
  }
  if (L->l_G->GCdebt > 0)
    luaC_step(L);
}
