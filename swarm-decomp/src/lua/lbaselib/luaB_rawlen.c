#include "lua_api.h"

static int luaB_rawlen (lua_State *L) {
  int t = lua_type(L, 1);
  luaL_argexpected(L, t == LUA_TTABLE || t == LUA_TSTRING, 1, "table or string");
  lua_pushinteger(L, lua_rawlen(L, 1));
  return 1;
}
