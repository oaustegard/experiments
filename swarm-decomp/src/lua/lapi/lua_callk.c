#include "lstate.h"
#include "lua.h"

LUA_API void lua_callk (lua_State *L, int nargs, int nresults,
                        lua_KContext ctx, lua_KFunction k) {
  StkId func;
  func = L->top - (nargs+1);
  if (k != NULL && ((L->nCcalls & 0xffff0000) == 0)) {
    L->ci->u.c.k = k;
    L->ci->u.c.ctx = ctx;
    luaD_call(L, func, nresults);
  }
  else
    luaD_callnoyield(L, func, nresults);
  if (nresults <= LUA_MULTRET && L->ci->top < L->top)
    L->ci->top = L->top;
}
