#include "lua_api.h"

static int luaB_next (lua_State *L) {
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_settop(L, 2);
  if (lua_next(L, 1) == 0) {
    lua_pushnil(L);
    return 1;
  }
  return 2;
}
