#include "lua_api.h"

static int luaB_warn (lua_State *L) {
  int n = lua_gettop(L);
  int i;
  luaL_checkstring(L, 1);
  for (i = 2; i <= n; i++)
    luaL_checkstring(L, i);
  for (i = 1; i < n; i++)
    lua_warning(L, lua_tostring(L, i), 1);
  lua_warning(L, lua_tostring(L, n), 0);
  return 0;
}
