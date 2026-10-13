#include "lua_api.h"

static int finishpcall (lua_State *L, int status, lua_KContext extra) {
  if (status != LUA_OK && status != LUA_YIELD) {
    lua_pushboolean(L, 0);
    lua_pushvalue(L, -2);
    return 2;
  }
  return lua_gettop(L) - (int)extra;
}
