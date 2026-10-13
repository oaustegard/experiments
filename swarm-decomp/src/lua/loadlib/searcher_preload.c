#include "lua.h"
#include "lauxlib.h"

static int searcher_preload (lua_State *L) {
  const char *name = luaL_checkstring(L, 1);
  lua_getfield(L, LUA_REGISTRYINDEX, "_PRELOAD");
  if (lua_getfield(L, -1, name) == LUA_TNIL) {
    lua_pushfstring(L, "no field package.preload['%s']", name);
    return 1;
  }
  else {
    lua_pushstring(L, ":preload:");
    return 2;
  }
}
