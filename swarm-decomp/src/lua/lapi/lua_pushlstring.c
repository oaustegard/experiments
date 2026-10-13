#include "lstate.h"
#include "fn_luaC_step.h"
#include "fn_luaS_new.h"
#include "fn_luaS_newlstr.h"
#include "lua.h"

LUA_API const char *lua_pushlstring (lua_State *L, const char *s, size_t len) {
  TString *ts;
  ts = (len == 0) ? luaS_new(L, "") : luaS_newlstr(L, s, len);
  L->top->value_.gc = (GCObject *)ts;
  L->top->tt_ = ts->tt | 0x40;
  L->top++;
  if (L->l_G->GCdebt > 0)
    luaC_step(L);
  return (const char *)(ts->contents);
}
