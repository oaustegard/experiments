#include <dlfcn.h>
#include "lua.h"
#include "lauxlib.h"

static int gctm (lua_State *L) {
  lua_Integer n = luaL_len(L, 1);
  for (; n >= 1; n--) {
    lua_rawgeti(L, 1, n);
    dlclose(lua_touserdata(L, -1));
    lua_settop(L, -2);
  }
  return 0;
}
