#include "lua_api.h"

static int ipairsaux (lua_State *L);

static int luaB_ipairs (lua_State *L) {
  luaL_checkany(L, 1);
  lua_pushcfunction(L, ipairsaux);
  lua_pushvalue(L, 1);
  lua_pushinteger(L, 0);
  return 3;
}
