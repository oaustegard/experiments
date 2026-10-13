#include "lstate.h"
#include "lua.h"
#include "fn_luaO_str2num.h"

LUA_API size_t lua_stringtonumber (lua_State *L, const char *s) {
  size_t sz = luaO_str2num(s, L->top);
  if (sz != 0)
    L->top++;
  return sz;
}
