#include "lstate.h"
#include "fn_luaD_growstack.h"
#include "lua.h"

LUA_API int lua_checkstack (lua_State *L, int n) {
  int res;
  CallInfo *ci;
  ci = L->ci;
  if (L->stack_last - L->top > n)
    res = 1;
  else
    res = luaD_growstack(L, n, 0);
  if (res && ci->top < L->top + n)
    ci->top = L->top + n;
  return res;
}
