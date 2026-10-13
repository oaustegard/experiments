#include "lstate.h"
#include "fn_luaS_new.h"
#include "fn_luaC_step.h"
#include "lua.h"

const char *lua_pushstring (lua_State *L, const char *s) {
  struct TString *ts;
  if (s == NULL)
    L->top->tt_ = 0;
  else {
    ts = luaS_new(L, s);
    L->top->value_.gc = (GCObject *)ts;
    L->top->tt_ = ((GCObject *)ts)->tt | 0x40;
    s = (const char *)ts + 0x18;
  }
  L->top++;
  if (L->l_G->GCdebt > 0)
    luaC_step(L);
  return s;
}
