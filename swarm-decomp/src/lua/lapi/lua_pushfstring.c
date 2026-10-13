#include "lstate.h"
#include "fn_luaC_step.h"
#include "fn_luaO_pushvfstring.h"
#include "lua.h"

LUA_API const char *lua_pushfstring (lua_State *L, const char *fmt, ...) {
  const char *ret;
  va_list argp;
  va_start(argp, fmt);
  ret = luaO_pushvfstring(L, fmt, argp);
  va_end(argp);
  if (L->l_G->GCdebt > 0)
    luaC_step(L);
  return ret;
}
