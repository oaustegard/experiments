#include <stdarg.h>
#include "lstate.h"
#include "lua.h"
#include "fn_luaO_pushvfstring.h"
#include "fn_luaC_step.h"

LUA_API const char *lua_pushvfstring (lua_State *L, const char *fmt, va_list argp) {
  const char *ret;
  ret = luaO_pushvfstring(L, fmt, argp);
  if (L->l_G->GCdebt > 0)
    luaC_step(L);
  return ret;
}
