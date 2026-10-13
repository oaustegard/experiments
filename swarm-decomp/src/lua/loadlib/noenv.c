#include "lua.h"
#include "lauxlib.h"

static int noenv (lua_State *L) {
  int b;
  lua_getfield(L, LUA_REGISTRYINDEX, "LUA_NOENV");
  b = lua_toboolean(L, -1);
  lua_settop(L, -2);
  return b;
}
