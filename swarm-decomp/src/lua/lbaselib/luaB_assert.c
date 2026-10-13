#include "lua_api.h"

static int luaB_error (lua_State *L);

static int luaB_assert (lua_State *L) {
  if (lua_toboolean(L, 1))
    return lua_gettop(L);
  else {
    luaL_checkany(L, 1);
    lua_remove(L, 1);
    lua_pushliteral(L, "assertion failed!");
    lua_settop(L, 1);
    return luaB_error(L);
  }
}
