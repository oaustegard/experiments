#include "lua_api.h"

static int finishpcall (lua_State *L, int status, lua_KContext extra);

static int luaB_pcall (lua_State *L) {
  int status;
  luaL_checkany(L, 1);
  lua_pushboolean(L, 1);
  lua_rotate(L, 1, 1);
  status = lua_pcallk(L, lua_gettop(L) - 2, LUA_MULTRET, 0, 0, finishpcall);
  return finishpcall(L, status, 0);
}
